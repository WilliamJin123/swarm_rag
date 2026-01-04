from typing import Dict, List, Callable, Any, NotRequired, Set, TypedDict
from dataclasses import dataclass, field

import numpy as np

from swarm_rag.evolution.execution.fitness import FitnessResult

from .expressions import ExpressionNode
from ...core.heuristics import HeuristicContext, HeuristicRegistry
from .expressions import ExpressionNode

class SwarmParams(TypedDict):
    """
    Defines the contract for Swarm Hyperparameters.
    Acts as a Dictionary at runtime, but provides IDE autocompletion.
    """
    n_agents: int
    steps: int
    decay: float
    initial_pool_size: int
    start_subset: int
    
    # Future-proofing: You can add optional fields without breaking old code
    exploration_jitter: NotRequired[float]

DEFAULT_PARAMS: SwarmParams = {
    "n_agents": 20,
    "steps": 4,
    "decay": 0.5,
    "initial_pool_size": 30,
    "start_subset": 10
}

@dataclass 
class Genome:
    """
    A complete retrieval strategy with BOTH hyperparameters
    and expression trees in one genome.
    """

    params: SwarmParams = field(default_factory=lambda: DEFAULT_PARAMS.copy())
    strategies: Dict[str, ExpressionNode] = field(default_factory=dict)

    fitness: FitnessResult

    metrics: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    evaluated: bool = False

    _compiled_cache: Dict[str, Callable] = field(default_factory=dict, repr=False)

    def complexity(self) -> int:
        """Sum of the size of all expression trees."""
        return sum(tree.size() for tree in self.strategies.values())
    
    def clear_cache(self):
        """Must be called after any mutation."""
        self._compiled_cache.clear()

    def get_kwargs(self) -> Dict[str, Any]:
        """
        Returns a dictionary ready to be unpacked into SwarmRetriever.retrieve().
        Merges params and compiled strategies.
        """
        # This will fail if not compiled, handled by Compiler/Evaluator
        if not self._compiled_cache and self.strategies:
            raise RuntimeError(f"Genome {self.id} accessed before compilation.")
            
        return {**self.params}
    
    def copy(self) -> 'Genome':
        """
        Deep copy ensures mutations don't bleed into parents/elites.
        """
        # Deep copy the strategies (ExpressionNodes are mutable)
        new_strategies = {k: v.copy() for k, v in self.strategies.items()}
        
        # Params are simple types (int/float), shallow copy of dict is fine
        new_params = self.params.copy()

        return Genome(
            id=f"{self.id}_copy",
            params=new_params,
            strategies=new_strategies,
            fitness=self.fitness,
            metrics=self.metrics.copy(),
            latency_ms=self.latency_ms,
            evaluated=self.evaluated,
            # Do NOT copy the cache; the new genome might be mutated
            _compiled_cache={} 
        )

class GenomeCompiler:
    """
    Dynamically compiles a Genome's symbolic trees into executable logic.
    Decoupled from specific strategy names (works for 'movement', 'ranking', or anything new).
    """

    def compile(self, genome: Genome) -> Dict[str, Any]:
        """
        Converts a Genome into the specific kwargs required by SwarmRetriever.
        """
        compiled_kwargs = {}
        for name, expr_tree in genome.strategies.items():
            if name in genome._compiled_cache:
                compiled_func = genome._compiled_cache[name]
            else:
                compiled_func = self._compile_tree(expr_tree)
                genome._compiled_cache[name] = compiled_func
            
            # Convention: strategy 'movement' -> kwarg 'movement_strategies'
            kwarg_key = f"{name}_strategies"

            # SwarmRetriever expects: {'strategy_name': (function, weight)}
            compiled_kwargs[kwarg_key] = {
                f"evolved_{name}": (compiled_func, 1.0)
            }
        # Merge with hyperparameters
        # Result: {'n_agents': 10, 'movement_strategies': {...}, ...}
        return {**genome.params, **compiled_kwargs}

    def _compile_tree(self, expr_tree: ExpressionNode) -> Callable[[HeuristicContext], float]:
        """
        Compiles a single expression tree into a lambda.
        """
        # Extract feature dependencies
        required_features = sorted(list(self._extract_features(expr_tree)))

        getters = []

        for name in required_features:
            # Handle Context Attributes (Fast Path)
            if name == 'degree':
                getters.append(lambda ctx: ctx.node_degrees)
            elif name == 'pheromone':
                getters.append(lambda ctx: ctx.pheromone_values)
            elif name == 'max_pheromone':
                getters.append(lambda ctx : ctx.max_pheromone)
            elif name == 'avg_degree':
                getters.append(lambda ctx: ctx.avg_degree)
            elif name == 'votes':
                getters.append(lambda ctx: ctx.votes)
            else:
                try:
                    func = HeuristicRegistry.get(name) 
                    getters.append(func)
                except ValueError:
                    print(f"Warning: Feature '{name}' not in registry. Assuming raw data.")
                    getters.append(lambda ctx, n=name: 0.0)
        
        # Signature: lambda arg1, arg2: ...
        compiled_lambda = expr_tree.compile(arg_names=required_features)
        
        def strategy_wrapper(ctx: HeuristicContext) -> float:
            args = [getter(ctx) for getter in getters]
            raw_scores = compiled_lambda(*args)
            return np.nan_to_num(raw_scores, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return strategy_wrapper

    def _extract_features(self, node: ExpressionNode) -> Set[str]:
        features = set()
        if node.type == 'feature':
            features.add(node.value)
        for child in node.children:
            features.update(self._extract_features(child))
        return features