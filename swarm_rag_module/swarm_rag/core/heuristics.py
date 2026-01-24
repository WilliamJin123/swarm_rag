from typing import Any, Dict, List, Optional, Union, Callable, Literal
import torch
import math
from dataclasses import dataclass, field

from ..interfaces.enums import HeuristicKey
from ..interfaces.registry import _MovementRegistry, _RankingRegistry, _DepositRegistry
from ..interfaces.abstract_classes import GraphStore


def _dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute dot product using PyTorch."""
    return torch.matmul(a, b)



class HeuristicRegistry:
    """
    Public API that mirrors the original three-registry design but re-uses
    the generic implementation from '_BaseRegistry'.
    """
    # expose the three concrete registries as class attributes
    movement = _MovementRegistry
    ranking  = _RankingRegistry
    deposit  = _DepositRegistry

    _REGISTRY_MAP = {
        "movement": _MovementRegistry,
        "ranking": _RankingRegistry,
        "deposit": _DepositRegistry
    }

    @classmethod
    def register(cls, strategy_type: Literal["movement", "ranking", "deposit"], name: Union["HeuristicKey", str]):
        """Generic registration helper."""
        registry = cls._REGISTRY_MAP.get(strategy_type)
        if not registry:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        return registry.register(name)

    @classmethod
    def register_movement(cls, name: "HeuristicKey"):
        """Decorator (or direct call) for a movement heuristic."""
        return cls.movement.register(name)

    @classmethod
    def register_ranking(cls, name: "HeuristicKey"):
        """Decorator (or direct call) for a ranking heuristic."""
        return cls.ranking.register(name)

    @classmethod
    def register_deposit(cls, name: "HeuristicKey"):
        """Decorator (or direct call) for a deposit heuristic."""
        return cls.deposit.register(name)

    @classmethod
    def get_movement(cls, name: Union["HeuristicKey", str]) -> Callable:
        return cls.movement.get(name)

    @classmethod
    def get_ranking(cls, name: Union["HeuristicKey", str]) -> Callable:
        return cls.ranking.get(name)

    @classmethod
    def get_deposit(cls, name: Union["HeuristicKey", str]) -> Callable:
        return cls.deposit.get(name)

    @classmethod
    def get_by_type(cls, strategy_type: Literal["movement", "ranking", "deposit"], name: Union["HeuristicKey", str]) -> Callable:
        """Type-specific lookup."""
        registry = cls._REGISTRY_MAP.get(strategy_type)
        if not registry:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        return registry.get(name)

    @classmethod
    def get(cls, name: Union["HeuristicKey", str]) -> Callable:
        """
        Search all registries (movement, ranking, deposit) for name and
        return the first matching heuristic.
        """
        for registry in cls._REGISTRY_MAP.values():
            try:
                return registry.get(name)
            except KeyError:
                continue
        
        raise KeyError(f"Heuristic '{name}' is not registered "
                       f"in movement, ranking, or deposit registries.")

    @classmethod
    def all_movement(cls):
        return cls.movement.all()

    @classmethod
    def all_ranking(cls):
        return cls.ranking.all()

    @classmethod
    def all_deposit(cls):
        return cls.deposit.all()

    @classmethod
    def all(cls):
        """Merge the three dictionaries into one view."""
        res = {}
        for registry in cls._REGISTRY_MAP.values():
            res.update(registry.all())
        return res

@dataclass(slots=True)
class HeuristicContext:
    """
    A shared dataclass to hold context for heuristic functions.

    All tensor fields are PyTorch tensors for GPU acceleration.
    """
    query_vec: torch.Tensor  # Shape: (D,)
    target_vecs: Optional[torch.Tensor] = None
    target_ids: Union[List[int], torch.Tensor] = None

    pheromone_values: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    node_degrees: torch.Tensor = field(default_factory=lambda: torch.tensor([]))

    graph: Optional[GraphStore] = None
    max_pheromone: float = 1.0
    avg_degree: float = 1.0
    step_index: int = 0
    agent_index: int = 0
    votes: int = 0
    total_agents: int = 0

    extra_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_gpu(self) -> bool:
        """Check if context is using GPU tensors."""
        if isinstance(self.target_vecs, torch.Tensor):
            return self.target_vecs.is_cuda
        return False

    @property
    def device(self) -> str:
        """Get the device of the tensors."""
        if isinstance(self.target_vecs, torch.Tensor):
            return str(self.target_vecs.device)
        return "cpu"

class Heuristics:
    """
    A library of preset heuristics.
    Each function takes a `HeuristicContext` object and returns a float score, normalized to [0,1].

    All heuristics use PyTorch tensors for GPU acceleration.
    """

    # --- MOVEMENT HEURISTICS ---

    @staticmethod
    @HeuristicRegistry.register_movement(HeuristicKey.SEMANTIC_SIMILARITY)
    def semantic_similarity(ctx: HeuristicContext) -> torch.Tensor:
        """
        NORMALIZED Cosine Similarity: Maps [-1, 1] to [0, 1].
        - 0.0 = completely opposite direction (cosine = -1)
        - 0.5 = orthogonal/unrelated (cosine = 0)
        - 1.0 = perfect match (cosine = 1)
        """
        scores = _dot_product(ctx.target_vecs, ctx.query_vec)
        # Normalize from [-1, 1] to [0, 1]
        return (scores + 1.0) / 2.0

    @staticmethod
    @HeuristicRegistry.register_movement(HeuristicKey.SEMANTIC_SIMILARITY_UNNORMALIZED)
    def semantic_similarity_unnormalized(ctx: HeuristicContext) -> torch.Tensor:
        """
        RAW Cosine Similarity in [-1, 1] for ranking where negative scores are meaningful.
        """
        return _dot_product(ctx.target_vecs, ctx.query_vec)

    @staticmethod
    @HeuristicRegistry.register_movement(HeuristicKey.NODE_CENTRALITY)
    def node_centrality(ctx: HeuristicContext) -> torch.Tensor:
        """
        Normalized centrality that works with any GraphStore.
        Requires ctx.graph.degree and ctx.graph.avg_degree

        Range: [0, 1] (sigmoid normalization)
        """
        log_degree = torch.log(1 + ctx.node_degrees.float())
        log_avg = math.log(1 + ctx.avg_degree)
        return log_degree / (log_degree + log_avg + 1e-8)

    @staticmethod
    @HeuristicRegistry.register_movement(HeuristicKey.PHEROMONE_REPULSION)
    def pheromone_repulsion(ctx: HeuristicContext) -> torch.Tensor:
        """
        Inverse Pheromone frequency.
        Returns 1.0 if no one has been there, approaches 0.0 as traffic increases.
        """
        max_p = max(ctx.max_pheromone, 0.0001)
        return 1.0 - (ctx.pheromone_values / max_p)

    @staticmethod
    @HeuristicRegistry.register_movement(HeuristicKey.RANDOM_JITTER)
    def random_jitter(ctx: HeuristicContext) -> torch.Tensor:
        """
        Adds pure chaos to break loops.
        """
        count = len(ctx.target_ids) if ctx.target_ids is not None else 1
        device = ctx.target_vecs.device if isinstance(ctx.target_vecs, torch.Tensor) else "cpu"
        return torch.rand(count, device=device)

    # --- RANKING HEURISTICS (Final Consensus) ---

    @staticmethod
    @HeuristicRegistry.register_ranking(HeuristicKey.PERCENTAGE_VISITED)
    def percentage_visited(ctx: HeuristicContext) -> float:
        """Percentage of total agents that visited this node."""
        if ctx.total_agents == 0: return 0.0
        return ctx.votes / ctx.total_agents

    @staticmethod
    @HeuristicRegistry.register_ranking(HeuristicKey.SEMANTIC_RANK)
    def semantic_rank(ctx: HeuristicContext) -> float:
        """
        RAW semantic similarity for final ranking.
        Uses full [-1, 1] range since we want to distinguish good from bad.
        """
        val = Heuristics.semantic_similarity_unnormalized(ctx)
        if hasattr(val, 'item'):
            return val.item()
        return float(val)

    # --- DEPOSIT HEURISTICS ---

    @staticmethod
    @HeuristicRegistry.register_deposit(HeuristicKey.FLAT)
    def deposit_flat(ctx: HeuristicContext) -> torch.Tensor:
        """Returns tensor of 1.0s matching input size."""
        return torch.ones_like(ctx.pheromone_values)

    @staticmethod
    @HeuristicRegistry.register_deposit(HeuristicKey.HUB)
    def deposit_hub(ctx: HeuristicContext) -> torch.Tensor:
        """Hubs get more pheromones."""
        return Heuristics.node_centrality(ctx)

    @staticmethod
    @HeuristicRegistry.register_deposit(HeuristicKey.SEMANTIC)
    def deposit_semantic(ctx: HeuristicContext) -> torch.Tensor:
        """
        Semantic-weighted deposit using NORMALIZED similarity.
        Only deposits on positive matches (similarity > 0.5 in normalized space) with range 0-1.
        """
        normalized_sim = Heuristics.semantic_similarity(ctx)
        return torch.where(normalized_sim > 0.5, (normalized_sim - 0.5) * 2.0,
                           torch.zeros_like(normalized_sim))

    @staticmethod
    @HeuristicRegistry.register_deposit(HeuristicKey.SEMANTIC_UNNORMALIZED)
    def deposit_semantic_unnormalized(ctx: HeuristicContext) -> torch.Tensor:
        """
        Alternative: Uses unnormalized similarity and clamps to [0, 1].
        Deposits on any positive match.

        Range: [0, 1]
        """
        sim = Heuristics.semantic_similarity_unnormalized(ctx)
        return torch.clamp(sim, min=0.0)

    @staticmethod
    @HeuristicRegistry.register_deposit(HeuristicKey.EXPLORATION_BONUS)
    def deposit_exploration_bonus(
        ctx: HeuristicContext,
        base_deposit: float = 1.0,
        fresh_multiplier: float = 2.0,
        high_traffic_multiplier: float = 0.5
    ) -> torch.Tensor:
        """
        Encourages visiting new nodes (Exploration).

        Args:
            base_deposit: Base amount to deposit
            fresh_multiplier: Multiplier for completely unvisited nodes (default 2.0)
            high_traffic_multiplier: Multiplier for maximally visited nodes (default 0.5)

        Range: [base_deposit * high_traffic_multiplier, base_deposit * fresh_multiplier]
        """
        max_p = max(ctx.max_pheromone, 0.0001)
        traffic_ratio = ctx.pheromone_values / max_p
        traffic_ratio = torch.clamp(traffic_ratio, 0.0, 1.0)
        multiplier = fresh_multiplier - (fresh_multiplier - high_traffic_multiplier) * traffic_ratio
        return base_deposit * multiplier

    @staticmethod
    @HeuristicRegistry.register_deposit(HeuristicKey.COLLABORATIVE_AMP)
    def deposit_collaborative_amplification(
        ctx: HeuristicContext,
        base_deposit: float = 1.0,
        amplification_factor: float = 1.0,
        max_multiplier: float = 5.0
    ) -> torch.Tensor:
        """
        The more pheromone already present, the larger the new deposit (Exploitation).
        This creates a "rich get richer" effect.
        """
        multiplier = 1.0 + (amplification_factor * ctx.pheromone_values)
        return base_deposit * torch.minimum(multiplier, torch.tensor(max_multiplier, device=multiplier.device))
    
