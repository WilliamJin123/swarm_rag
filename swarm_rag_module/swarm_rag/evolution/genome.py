from typing import Dict, List, Callable, Any, Set
from dataclasses import dataclass, field
from .expressions import ExpressionNode
from ..core.heuristics import HeuristicContext, HeuristicRegistry
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

    # TO CACHE
    # compiled_movement: Optional[Callable] = None
    # compiled_ranking: Optional[Callable] = None
    # compiled_deposit: Optional[Callable] = None


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

    def complexity(self) -> int:
        """Total complexity across all expressions."""
        return (
            self.movement_expr.size() +
            self.ranking_expr.size() +
            self.deposit_expr.size()
        )
    
    def get_metric(self, metric_name: str, default: float = 0.0) -> float:
        """Safely get a metric value."""
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
            movement_expr=self.movement_expr.copy(),
            ranking_expr=self.ranking_expr.copy(),
            deposit_expr=self.deposit_expr.copy(),
            
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
        return {
            'n_agents': genome.n_agents,
            'steps': genome.steps,
            'decay': genome.decay,
            'initial_pool_size': genome.initial_pool_size,
            'start_subset': genome.start_subset,
            'movement_strategies': {'evolved_move': (self._compile_tree(genome.movement_expr), 1.0)},
            'ranking_strategies': {'evolved_rank': (self._compile_tree(genome.ranking_expr), 1.0)},
            'deposit_strategies': {'evolved_deposit': (self._compile_tree(genome.deposit_expr), 1.0)}
        }
    
    def _compile_tree(self, expr_tree: ExpressionNode) -> Callable[[HeuristicContext], float]:
        """
        Compiles expression tree into a callable strategy.
        Now decoupled from the Engine logic.
        """
        # 1. Extract feature dependencies
        required_features = self._extract_features(expr_tree)
        
        # 2. Compile the lambda (CPU heavy operation)
        compiled_lambda = expr_tree.compile()
        
        # 3. Create the optimized wrapper
        def strategy_wrapper(ctx: HeuristicContext) -> float:
            feature_values = {name: HeuristicRegistry.get(name)(ctx) for name in required_features}
            score = compiled_lambda(feature_values)
            return max(0.0, min(1.0, score))
            
        return strategy_wrapper

    def _extract_features(self, node: ExpressionNode) -> Set[str]:
        features = set()
        if node.type == 'feature':
            features.add(node.value)
        for child in node.children:
            features.update(self._extract_features(child))
        return features