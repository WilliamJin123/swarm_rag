
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
    # Expose the three concrete registries as class attributes
    movement = _MovementRegistry
    ranking = _RankingRegistry
    deposit = _DepositRegistry

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
        return cls.movement.register(name)

    @classmethod
    def register_ranking(cls, name: "HeuristicKey"):
        return cls.ranking.register(name)

    @classmethod
    def register_deposit(cls, name: "HeuristicKey"):
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
        registry = cls._REGISTRY_MAP.get(strategy_type)
        if not registry:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        return registry.get(name)

    @classmethod
    def get(cls, name: Union["HeuristicKey", str]) -> Callable:
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
    target_ids: Optional[torch.Tensor] = None

    pheromone_values: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    node_degrees: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    graph: Optional[GraphStore] = None
    max_pheromone: float = 1.0
    avg_degree: float = 1.0
    step_index: int = 0
    agent_index: int = 0
    votes: Optional[torch.Tensor] = None
    total_agents: int = 0

    extra_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_gpu(self) -> bool:
        """Check if context is using GPU tensors."""
        # .is_cuda is efficient and doesn't trigger synchronization for checks
        if isinstance(self.target_vecs, torch.Tensor):
            return self.target_vecs.is_cuda
        return False

    @property
    def device(self) -> torch.device:
        """
        Get the device of the tensors.
        Returns torch.device object for direct use in tensor creation, avoiding string parsing.
        """
        if isinstance(self.target_vecs, torch.Tensor):
            return self.target_vecs.device
        return torch.device("cpu")

class Heuristics:
    """
    A library of preset heuristics.
    Each function takes a `HeuristicContext` object and returns a float score/tensor.
    Optimized to avoid memory allocations and run efficiently on GPU/CPU.
    """

    # --- MOVEMENT HEURISTICS ---

    @staticmethod
    @HeuristicRegistry.register_movement(HeuristicKey.SEMANTIC_SIMILARITY)
    def semantic_similarity(ctx: HeuristicContext) -> torch.Tensor:
        """
        NORMALIZED Cosine Similarity: Maps [-1, 1] to [0, 1].
        Uses broadcasting for scalar addition/multiplication.
        """
        scores = _dot_product(ctx.target_vecs, ctx.query_vec)
        # Broadcasting: (1.0) is implicitly handled by PyTorch
        return (scores + 1.0) * 0.5

    @staticmethod
    @HeuristicRegistry.register_movement(HeuristicKey.SEMANTIC_SIMILARITY_UNNORMALIZED)
    def semantic_similarity_unnormalized(ctx: HeuristicContext) -> torch.Tensor:
        """RAW Cosine Similarity in [-1, 1]."""
        return _dot_product(ctx.target_vecs, ctx.query_vec)

    @staticmethod
    @HeuristicRegistry.register_movement(HeuristicKey.NODE_CENTRALITY)
    def node_centrality(ctx: HeuristicContext) -> torch.Tensor:
        """
        Normalized centrality.
        Range: [0, 1]
        """
        # Ensure float calculation to avoid integer division truncation
        log_degree = torch.log(1.0 + ctx.node_degrees.float())
        log_avg = math.log(1.0 + ctx.avg_degree)
        return log_degree / (log_degree + log_avg + 1e-8)

    @staticmethod
    @HeuristicRegistry.register_movement(HeuristicKey.PHEROMONE_REPULSION)
    def pheromone_repulsion(ctx: HeuristicContext) -> torch.Tensor:
        """
        Inverse Pheromone frequency.
        """
        max_p = max(ctx.max_pheromone, 1e-4)
        # Scalar division is broadcasted
        return 1.0 - (ctx.pheromone_values / max_p)

    @staticmethod
    @HeuristicRegistry.register_movement(HeuristicKey.RANDOM_JITTER)
    def random_jitter(ctx: HeuristicContext) -> torch.Tensor:
        """
        Adds pure chaos to break loops.
        Uses the context device directly to ensure GPU compatibility without device transfers.
        """
        # Determine batch size. If target_vecs is None, assume scalar or single element.
        if ctx.target_vecs is not None:
            count = ctx.target_vecs.shape[0]
        elif ctx.target_ids is not None:
            count = ctx.target_ids.shape[0]
        else:
            count = 1
            
        return torch.rand(count, device=ctx.device)

    # --- RANKING HEURISTICS ---

    @staticmethod
    @HeuristicRegistry.register_ranking(HeuristicKey.PERCENTAGE_VISITED)
    def percentage_visited(ctx: HeuristicContext) -> torch.Tensor:
        """
        Percentage of total agents that visited this node.
        OPTIMIZATION: Returns a zero-tensor matching shape if agents is 0, 
        preventing shape mismatch errors during aggregation.
        """
        if ctx.total_agents == 0:
            if ctx.votes is not None:
                # Returns zeros of same shape/device as votes, explicitly float
                return torch.zeros_like(ctx.votes, dtype=torch.float32)
            # Fallback for scalar case
            return torch.tensor(0.0)
        
        return ctx.votes.float() / ctx.total_agents

    @staticmethod
    @HeuristicRegistry.register_ranking(HeuristicKey.SEMANTIC_RANK)
    def semantic_rank(ctx: HeuristicContext) -> torch.Tensor:
        """RAW semantic similarity for final ranking."""
        return Heuristics.semantic_similarity_unnormalized(ctx)

    # --- DEPOSIT HEURISTICS ---

    @staticmethod
    @HeuristicRegistry.register_deposit(HeuristicKey.FLAT)
    def deposit_flat(ctx: HeuristicContext) -> torch.Tensor:
        """Returns tensor of 1.0s matching input size."""
        # ones_like preserves shape and device
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
        Only deposits on positive matches (similarity > 0.5 in normalized space).
        """
        normalized_sim = Heuristics.semantic_similarity(ctx)
        # Broadcasting 0.5 and 2.0
        return torch.where(normalized_sim > 0.5, (normalized_sim - 0.5) * 2.0,
                           torch.zeros_like(normalized_sim))

    @staticmethod
    @HeuristicRegistry.register_deposit(HeuristicKey.SEMANTIC_UNNORMALIZED)
    def deposit_semantic_unnormalized(ctx: HeuristicContext) -> torch.Tensor:
        """
        Uses unnormalized similarity and clamps to [0, 1].
        """
        sim = Heuristics.semantic_similarity_unnormalized(ctx)
        # Optimized clamp using float scalar
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
        Uses pure Python math for scalars, broadcasting for tensors.
        """
        max_p = max(ctx.max_pheromone, 1e-4)
        traffic_ratio = ctx.pheromone_values / max_p
        traffic_ratio = torch.clamp(traffic_ratio, 0.0, 1.0)
        
        # Scalar arithmetic outside of tensor ops where possible
        delta_multiplier = fresh_multiplier - high_traffic_multiplier
        multiplier = fresh_multiplier - (delta_multiplier * traffic_ratio)
        
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
        """
        multiplier = 1.0 + (amplification_factor * ctx.pheromone_values)
        # Optimized: Broadcasts max_multiplier float directly, no tensor creation
        clamped_multiplier = torch.clamp(multiplier, max=max_multiplier)
        return base_deposit * clamped_multiplier