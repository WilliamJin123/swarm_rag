"""
Shared Types for SwarmRAG.

These types are used across both core and evolution modules.
Centralizing them here prevents circular imports and provides
a single source of truth for type definitions.

Usage:
    from swarm_rag.interfaces.shared_types import AgentGroupConfig, StrategyConfig
"""
from typing import Dict, Any, List, Tuple, Callable, TypedDict, Optional, Literal

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


TorchDeviceStr = Literal["cpu", "cuda", "mps"]

# =============================================================================
# Strategy Configuration Types
# =============================================================================

class StrategyConfig(TypedDict, total=False):
    """
    Configuration for a named strategy (movement, deposit, or ranking).

    Strategies can be defined as:
    1. Tuple of (heuristic_fn_or_name, weight)
    2. Tuple of (callable, weight) for custom strategies

    Example:
        movement_strategies: StrategyConfig = {
            "semantic": ("semantic_similarity", 0.5),
            "diversity": ("pheromone_repulsion", 0.3),
            "hub": (my_custom_fn, 0.2),
        }
    """
    pass  # Dict[str, Tuple[Any, float]]


class AgentGroupConfig(TypedDict):
    """
    Configuration for a specific sub-group of agents.

    Used to define heterogeneous agent populations where different
    groups can have different movement and deposit strategies.

    Example:
        agent_groups = [
            {
                "count": 10,
                "movement_strategies": {"semantic": (semantic_fn, 1.0)},
                "deposit_strategies": {"flat": (flat_fn, 1.0)},
            },
            {
                "count": 5,
                "movement_strategies": {"explorer": (explore_fn, 1.0)},
                "deposit_strategies": {"hub": (hub_fn, 1.0)},
            },
        ]
    """
    count: int  # How many agents of this type?
    movement_strategies: Dict[str, Any]  # {name: (fn_or_name, weight)}
    deposit_strategies: Dict[str, Any]   # {name: (fn_or_name, weight)}


# =============================================================================
# Retrieval Configuration Types
# =============================================================================

class RetrievalConfig(TypedDict, total=False):
    """
    Configuration for a single retrieval operation.

    Combines global parameters with strategy configurations.
    """
    # Global parameters
    n_agents: int
    steps: int
    decay: float
    drop_zone_inc: float
    initial_pool_size: int
    start_subset: int
    top_k: int

    # Strategy configurations (used when not using agent_groups)
    movement_strategies: Dict[str, Any]
    deposit_strategies: Dict[str, Any]
    ranking_strategies: Dict[str, Any]

    # Heterogeneous agent groups (overrides global strategies)
    agent_groups: List[AgentGroupConfig]


# =============================================================================
# Metric Types
# =============================================================================

class RetrievalMetrics(TypedDict, total=False):
    """
    Metrics computed from a retrieval evaluation.

    Standard metrics for comparing retrieval quality.
    """
    # Recall metrics
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float

    # Hit metrics
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    hit_at_20: float

    # Ranking metrics
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain

    # Efficiency metrics
    latency: float  # Seconds per query
    complexity: int  # Genome complexity (expression tree size)
    variance: float  # Metric variance across queries


class FitnessMetrics(TypedDict, total=False):
    """
    Fitness metrics for evolutionary optimization.

    These are the primary metrics used for selection and comparison.
    """
    quality_score: float   # Primary quality metric (higher is better)
    cost_score: float      # Cost/latency metric (lower is better)
    stability_score: float # Consistency metric (higher is better)


# =============================================================================
# Expression Tree Types
# =============================================================================

class ExpressionNodeDict(TypedDict, total=False):
    """
    Serialized representation of an expression tree node.

    Used for JSON serialization and reconstruction.
    """
    type: str  # 'op', 'feature', 'const', 'weighted'
    value: Any  # Operator name, feature name, or constant value
    children: List["ExpressionNodeDict"]
    weight: NotRequired[float]


# =============================================================================
# Heuristic Types
# =============================================================================

class HeuristicInfo(TypedDict):
    """
    Information about a registered heuristic.
    """
    name: str
    category: str  # 'movement', 'deposit', 'ranking'
    description: str
    is_vectorized: bool


