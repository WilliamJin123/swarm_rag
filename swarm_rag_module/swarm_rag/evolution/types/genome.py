from typing import Dict, List, Callable, Any, Set
from dataclasses import dataclass, field

import numpy as np
from .expressions import ExpressionNode
from ...core.heuristics import HeuristicContext, HeuristicRegistry
from .expressions import ExpressionNode

@dataclass 
class Genome:
    """
    A complete retrieval strategy with BOTH hyperparameters
    and expression trees in one genome.
    """

    # Hyperparameters
    n_agents: int
    steps: int
    decay: float
    initial_pool_size: int
    start_subset: int
    
    # Expression trees (the evolved heuristics)
    movement_expr: ExpressionNode
    ranking_expr: ExpressionNode
    deposit_expr: ExpressionNode

    # Cache for compiled strategies
    _compiled_movement: Any = field(default=None, repr=False)
    _compiled_ranking: Any = field(default=None, repr=False)
    _compiled_deposit: Any = field(default=None, repr=False)

    

    # Available features for each strategy (TO CHANGE)
    available_movement_features: List[str] = field(default_factory=lambda: [
        'semantic', 'centrality', 'diversity', 'jitter'
    ])
    available_ranking_features: List[str] = field(default_factory=lambda: [
        'semantic', 'votes', 'centrality'
    ])
    available_deposit_features: List[str] = field(default_factory=lambda: [
        'flat', 'semantic', 'hub', 'explorer'
    ])

    # Performance metrics
    fitness: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0

    evaluated: bool = False

    def clear_cache(self):
        """Call this after mutation/crossover"""
        self._compiled_movement = None
        self._compiled_ranking = None
        self._compiled_deposit = None

    def complexity(self) -> int:
        """Total complexity across all expressions."""
        c = 0
        if self.movement_expr: c += self.movement_expr.size()
        if self.ranking_expr: c += self.ranking_expr.size()
        if self.deposit_expr: c += self.deposit_expr.size()
        return c
    
    def get_metric(self, metric_name: str, default: float = 0.0) -> float:
        return self.metrics.get(metric_name, default)

    def copy(self) -> 'Genome':
        """
        Creates a deep copy of the Genome.
        Crucial for preventing mutations from affecting parents/elites.
        """
        return Genome(
            # 1. Primitives (Copy by value automatically)
            n_agents=self.n_agents,
            steps=self.steps,
            decay=self.decay,
            initial_pool_size=self.initial_pool_size,
            start_subset=self.start_subset,
            
            # 2. Expression Trees (MUST use their .copy() method)
            movement_expr=self.movement_expr.copy() if self.movement_expr else None,
            ranking_expr=self.ranking_expr.copy() if self.ranking_expr else None,
            deposit_expr=self.deposit_expr.copy() if self.deposit_expr else None,
            
            # 3. Mutable Lists/Dicts (Create new objects)
            metrics=self.metrics.copy(),
            available_movement_features=list(self.available_movement_features),
            available_ranking_features=list(self.available_ranking_features),
            available_deposit_features=list(self.available_deposit_features),
            
            # 4. State Flags
            fitness=self.fitness,
            latency_ms=self.latency_ms,
            evaluated=self.evaluated
        )

class GenomeCompiler:
    """
    Responsible for converting a Genome's symbolic expression trees
    into executable Python callables that the SwarmRetriever can consume.
    """

    def compile(self, genome: Genome) -> Dict[str, Any]:
        """
        Converts a Genome into the specific kwargs required by SwarmRetriever.
        """
        if genome._compiled_movement is None:
             genome._compiled_movement = self._compile_tree(genome.movement_expr)
        
        if genome._compiled_deposit is None:
             genome._compiled_deposit = self._compile_tree(genome.deposit_expr)

        if genome._compiled_ranking is None:
             genome._compiled_ranking = self._compile_tree(genome.ranking_expr)

        return {
            'n_agents': genome.n_agents,
            'steps': genome.steps,
            'decay': genome.decay,
            'initial_pool_size': genome.initial_pool_size,
            'start_subset': genome.start_subset,
            'movement_strategies': {
                'evolved_move': (genome._compiled_movement, 1.0)
            },
            'depoist_strategies': {
                'evolved_deposit': (genome._compiled_deposit, 1.0)
            },
            'ranking_strategies': {
                'evolved_rank': (genome._compiled_ranking, 1.0)
            },
        }

    def _compile_tree(self, expr_tree: ExpressionNode) -> Callable[[HeuristicContext], float]:
        """
        Compiles expression tree into a callable strategy.
        Now decoupled from the Engine logic.
        """
        # Extract feature dependencies
        required_features = sorted(list(self._extract_features(expr_tree)))
        resolved_funcs = {}

        # Signature becomes: func(arg1, arg2, arg3...)
        compiled_lambda = expr_tree.compile(arg_names=required_features)

        getters = []

        for name in required_features:
            # raw fields
            if name == 'degree':
                getters.append(lambda ctx: ctx.node_degrees)
            elif name == 'pheromone':
                getters.append(lambda ctx: ctx.pheromone_values)
            elif name == 'max_pheromone':
                getters.append(lambda ctx : ctx.max_pheromone)
            elif name == 'avg_degree':
                getters.append(lambda ctx: ctx.avg_degree)
            else:
                func = HeuristicRegistry.get(name) 
                getters.append(func)

        def strategy_wrapper(ctx: HeuristicContext) -> float:
            args = [getter(ctx) for getter in getters]
            raw_scores = compiled_lambda(*args)
            return np.logaddexp(0, np.clip(raw_scores, -10, 10))
        
        return strategy_wrapper

    def _extract_features(self, node: ExpressionNode) -> Set[str]:
        features = set()
        if node.type == 'feature':
            features.add(node.value)
        for child in node.children:
            features.update(self._extract_features(child))
        return features
    
    @classmethod
    def map_ctx_attr(key):
            """Map string key to actual context attribute"""
            mapping = {
                'degree': 'node_degrees', 
                'pheromone': 'pheromone_values',
                # others are the same
            }
            return mapping.get(key, key)