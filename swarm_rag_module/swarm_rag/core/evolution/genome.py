from typing import Dict, List
from dataclasses import dataclass, field
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