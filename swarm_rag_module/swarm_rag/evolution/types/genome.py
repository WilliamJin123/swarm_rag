from typing import Dict, Callable, Any, List, Set, TypedDict
from dataclasses import dataclass, field
try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired
import numpy as np

from .fitness_results import FitnessResult

from .expressions import ExpressionNode
from ...core.heuristics import HeuristicContext, HeuristicRegistry
from .expressions import ExpressionNode

class CompiledStrategies(TypedDict, total=False):
    """
    Type definition for the compiled function cache.
    total=False means keys can be missing (e.g. if 'deposit' isn't compiled yet).
    """
    movement: Callable
    ranking: Callable
    deposit: Callable

class SwarmParams(TypedDict):
    """
    Defines the contract for Swarm Hyperparameters.
    Acts as a Dictionary at runtime, but provides IDE autocompletion.
    """
    n_agents: int
    steps: int
    drop_zone_inc: float
    decay: float
    initial_pool_size: int
    start_subset: int
    

DEFAULT_PARAMS: SwarmParams = {
    "n_agents": 20,
    "steps": 4,
    "decay": 0.5,
    "drop_zone_inc": 0.05,
    "initial_pool_size": 30,
    "start_subset": 10,
}

@dataclass 
class Genome:
    """
    A complete retrieval strategy with BOTH hyperparameters
    and expression trees in one genome.
    """
    id: str
    mutation_rate: float = 0.1

    params: SwarmParams = field(default_factory=lambda: DEFAULT_PARAMS.copy())
    group_ratios: Dict[str, float] = field(default_factory=dict)
    strategies: Dict[str, ExpressionNode] = field(default_factory=dict)

    fitness: FitnessResult = field(default_factory=lambda: FitnessResult())

    # Recall@20, Hit@1, Hit@5, MRR, etc.
    metrics: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    evaluated: bool = False

    _compiled_cache: CompiledStrategies = field(default_factory=dict, repr=False)

    def __hash__(self):
        """Allows Genome to be used in sets or as dict keys."""
        return hash(self.id)

    def __eq__(self, other):
        """Genomes with the same ID are considered the same object."""
        if not isinstance(other, Genome):
            return False
        return self.id == other.id

    def __getstate__(self):
        """
        Custom pickling logic.
        EXCLUDES the compiled cache (functions) because they cannot be pickled.
        We only save the 'strategies' (trees) and 'params' (data).
        """
        state = self.__dict__.copy()
        # Remove the un-picklable cache
        state['_compiled_cache'] = {} 
        return state

    def __setstate__(self, state):
        """
        Restores state and re-initializes the empty cache.
        """
        self.__dict__.update(state)
        # Ensure cache exists
        if '_compiled_cache' not in self.__dict__:
            self._compiled_cache = {}

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
        return Genome(
            id=f"{self.id}_copy",
            mutation_rate=self.mutation_rate,
            params=self.params.copy(),
            group_ratios=self.group_ratios.copy(),
            strategies={k: v.copy() for k, v in self.strategies.items()},
            fitness=self.fitness, 
            metrics=self.metrics.copy(),
            latency_ms=self.latency_ms,
            evaluated=self.evaluated,
            _compiled_cache={} 
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the Genome into a plain Python dictionary.
        
        This is essential for saving genomes to JSON, logging, or other
        forms of data interchange. It intelligently handles nested objects
        and excludes non-serializable components like the compiled cache.

        Returns:
            A dictionary representation of the genome, ready for JSON serialization.
        """
        d = self.__dict__.copy()
        # Exclude the non-serializable compiled cache
        d.pop('_compiled_cache', None)

        # Serialize nested, complex objects.
        if 'strategies' in d and d['strategies']:
            d['strategies'] = {key: node.to_dict() for key, node in d['strategies'].items()}
        
        if 'fitness' in d and hasattr(d['fitness'], 'to_dict'):
            d['fitness'] = d['fitness'].to_dict()
        
        return d

class GenomeCompiler:
    """
    Dynamically compiles a Genome's symbolic trees into executable logic.
    Decoupled from specific strategy names (works for 'movement', 'ranking', or anything new).
    """

    def compile(self, genome: Genome) -> Dict[str, Any]:
        """
        Converts a Genome into the specific kwargs required by SwarmRetriever.
        """
        valid_suffixes = {'movement', 'deposit', 'ranking'}
        for key in genome.strategies:
            if key == 'ranking': 
                continue
            
            # Split 'g0_movement' -> prefix='g0', suffix='movement'
            if "_" in key:
                prefix, suffix = key.rsplit('_', 1)
                if suffix not in valid_suffixes:
                    raise ValueError(f"Genome contains illegal strategy key: '{key}'. Suffix '{suffix}' not in {valid_suffixes}")
            else:
                raise ValueError(f"Genome contains malformed strategy key: '{key}' (expected format 'gN_type')")

        compiled_funcs = {}
        for name, expr_tree in genome.strategies.items():
            if name in genome._compiled_cache:
                compiled_func = genome._compiled_cache[name]
            else:
                compiled_func = self._compile_tree(expr_tree)
                genome._compiled_cache[name] = compiled_func
            compiled_funcs[name] = compiled_func

        ranking_strategies = {}
        if "ranking" in compiled_funcs:
             ranking_strategies["evolved_ranking"] = (compiled_funcs["ranking"], 1.0)    
        
        agent_groups = []
        total_agents = genome.params['n_agents']

        # Use the explicit group_ratios dict, sorted by key
        sorted_groups = sorted(genome.group_ratios.keys()) # ['g0', 'g1', ...]
        
        if sorted_groups:
            ratios = [genome.group_ratios[g] for g in sorted_groups]
            total_ratio = sum(ratios) 
            if total_ratio <= 1e-9:
                total_ratio = 1.0
            
            counts = [int(round(total_agents * (r / total_ratio))) for r in ratios]
            
            # Fix rounding remainder (dump into first group)
            current_sum = sum(counts)
            if current_sum < total_agents:
                # Add deficit to first group
                counts[0] += (total_agents - current_sum)
            elif current_sum > total_agents:
                # Remove surplus from first group (rare with round, but possible)
                counts[0] -= (current_sum - total_agents)
            
            for i, group_key in enumerate(sorted_groups):
                if counts[i] <= 0: continue
                
                # Extract index from "g0", "g1"
                idx_suffix = group_key[1:] 
                mov_key = f"{group_key}_movement"
                dep_key = f"{group_key}_deposit"
                
                group_config = {
                    "count": counts[i],
                    "movement_strategies": { f"evolved_mov_{idx_suffix}": (compiled_funcs.get(mov_key), 1.0) },
                    "deposit_strategies": { f"evolved_dep_{idx_suffix}": (compiled_funcs.get(dep_key), 1.0) },
                }
                agent_groups.append(group_config)

        # Return keyword arguments for retrieve()
        return {
            **genome.params, 
            "agent_groups": agent_groups,
            "ranking_strategies": ranking_strategies,
        }



    def _compile_tree(self, expr_tree: ExpressionNode) -> Callable[[HeuristicContext], float]:
        """
        Compiles a single expression tree into a lambda.
        """
        raw_features = self._extract_features(expr_tree)
        required_features = sorted([str(f.value) if hasattr(f, 'value') else str(f) for f in raw_features])

        getters: List[Callable[[HeuristicContext], Any]] = []
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