from typing import Any, Dict, List, Optional, Union, Callable, Literal, TYPE_CHECKING
import numpy as np
import math
from dataclasses import dataclass, field

from ..interfaces.enums import HeuristicKey
from ..interfaces.registry import _MovementRegistry, _RankingRegistry, _DepositRegistry
from ..interfaces.abstract_classes import GraphStore

# Optional torch import for GPU operations
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:
    import torch


def _to_numpy(arr: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
    """Convert tensor to numpy array if needed."""
    if _TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _dot_product(a: Union[np.ndarray, "torch.Tensor"], b: Union[np.ndarray, "torch.Tensor"]) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Compute dot product that works with both numpy and torch tensors.
    Keeps computation on GPU if inputs are GPU tensors.
    """
    if _TORCH_AVAILABLE:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.matmul(a, b)
        elif isinstance(a, torch.Tensor):
            b_tensor = torch.from_numpy(np.asarray(b)).to(a.device)
            return torch.matmul(a, b_tensor)
        elif isinstance(b, torch.Tensor):
            a_tensor = torch.from_numpy(np.asarray(a)).to(b.device)
            return torch.matmul(a_tensor, b)

    # NumPy fallback
    return np.dot(np.asarray(a), np.asarray(b))



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

    Supports both numpy arrays and PyTorch tensors for GPU acceleration.
    When using GPU tensors, computations stay on GPU for maximum performance.
    """
    query_vec: Union[np.ndarray, "torch.Tensor"]  # Shape: (D,)
    target_vecs: Optional[Union[np.ndarray, "torch.Tensor"]] = None
    target_ids: Union[List[int], np.ndarray] = None

    pheromone_values: Union[np.ndarray, "torch.Tensor"] = field(default_factory=lambda: np.array([]))
    node_degrees: Union[np.ndarray, "torch.Tensor"] = field(default_factory=lambda: np.array([]))

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
        if _TORCH_AVAILABLE:
            if isinstance(self.target_vecs, torch.Tensor):
                return self.target_vecs.is_cuda
        return False

    @property
    def device(self) -> str:
        """Get the device of the tensors."""
        if _TORCH_AVAILABLE and isinstance(self.target_vecs, torch.Tensor):
            return str(self.target_vecs.device)
        return "cpu"

class Heuristics:
    """
    A library of preset heuristics.
    Each function takes a `HeuristicContext` object and returns a float score, normalized to [0,1].

    All heuristics support both numpy arrays and PyTorch tensors for GPU acceleration.
    When GPU tensors are provided, computations stay on GPU for maximum performance.
    """

    # --- MOVEMENT HEURISTICS ---

    @staticmethod
    @HeuristicRegistry.register_movement(HeuristicKey.SEMANTIC_SIMILARITY)
    def semantic_similarity(ctx: HeuristicContext) -> Union[float, Any]:
        """
        NORMALIZED Cosine Similarity: Maps [-1, 1] to [0, 1].
        - 0.0 = completely opposite direction (cosine = -1)
        - 0.5 = orthogonal/unrelated (cosine = 0)
        - 1.0 = perfect match (cosine = 1)

        Supports GPU acceleration when ctx.target_vecs is a CUDA tensor.
        """
        scores = _dot_product(ctx.target_vecs, ctx.query_vec)
        # Normalize from [-1, 1] to [0, 1]
        return (scores + 1.0) / 2.0

    @staticmethod
    @HeuristicRegistry.register_movement(HeuristicKey.SEMANTIC_SIMILARITY_UNNORMALIZED)
    def semantic_similarity_unnormalized(ctx: HeuristicContext) -> Union[float, Any]:
        """
        RAW Cosine Similarity in [-1, 1] for ranking where negative scores are meaningful.

        Supports GPU acceleration when ctx.target_vecs is a CUDA tensor.
        """
        return _dot_product(ctx.target_vecs, ctx.query_vec)
    
    @staticmethod
    @HeuristicRegistry.register_movement(HeuristicKey.NODE_CENTRALITY)
    def node_centrality(ctx: HeuristicContext) -> Any:
        """
        Normalized centrality that works with any GraphStore.
        Requires ctx.graph.degree and ctx.graph.avg_degree
        
        Range: [0, 1] (sigmoid normalization)
        """
        log_degree = np.log(1 + ctx.node_degrees)
        log_avg = math.log(1 + ctx.avg_degree)
        # Sigmoid normalization
        return log_degree / (log_degree + log_avg + 1e-8)

    @staticmethod
    @HeuristicRegistry.register_movement(HeuristicKey.PHEROMONE_REPULSION)
    def pheromone_repulsion(ctx: HeuristicContext) -> Any:
        """
        Inverse Pheromone frequency. 
        Returns 1.0 if no one has been there, approaches 0.0 as traffic increases.
        """
        max_p = max(ctx.max_pheromone, 0.0001)
        return 1.0 - (ctx.pheromone_values / max_p)

    @staticmethod
    @HeuristicRegistry.register_movement(HeuristicKey.RANDOM_JITTER)
    def random_jitter(ctx: HeuristicContext) -> Any:
        """Adds pure chaos to break loops."""
        count = len(ctx.target_ids) if ctx.target_ids is not None else 1
        return np.random.random(count)

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

        Supports GPU acceleration when ctx.target_vecs is a CUDA tensor.
        """
        val = Heuristics.semantic_similarity_unnormalized(ctx)
        # Handle both numpy arrays and torch tensors
        if hasattr(val, 'item'):
            return val.item()
        return float(val)

    # --- DEPOSIT HEURISTICS ---

    @staticmethod
    @HeuristicRegistry.register_deposit(HeuristicKey.FLAT)
    def deposit_flat(ctx: HeuristicContext) -> Any:
        """Returns array of 1.0s matching input size."""
        return np.ones_like(ctx.pheromone_values)

    @staticmethod
    @HeuristicRegistry.register_deposit(HeuristicKey.HUB)
    def deposit_hub(ctx: HeuristicContext) -> Any:
        """Hubs get more pheromones."""
        return Heuristics.node_centrality(ctx)
    
    @staticmethod
    @HeuristicRegistry.register_deposit(HeuristicKey.SEMANTIC)
    def deposit_semantic(ctx: HeuristicContext) -> Any:
        """
        Semantic-weighted deposit using NORMALIZED similarity.
        Only deposits on positive matches (similarity > 0.5 in normalized space) with range 0-1.

        Supports GPU acceleration when ctx.target_vecs is a CUDA tensor.
        """
        normalized_sim = Heuristics.semantic_similarity(ctx)
        # Handle both numpy and torch
        if _TORCH_AVAILABLE and isinstance(normalized_sim, torch.Tensor):
            return torch.where(normalized_sim > 0.5, (normalized_sim - 0.5) * 2.0,
                               torch.zeros_like(normalized_sim))
        return np.where(normalized_sim > 0.5, (normalized_sim - 0.5) * 2.0, 0.0)

    @staticmethod
    @HeuristicRegistry.register_deposit(HeuristicKey.SEMANTIC_UNNORMALIZED)
    def deposit_semantic_unnormalized(ctx: HeuristicContext) -> Any:
        """
        Alternative: Uses unnormalized similarity and clamps to [0, 1].
        Deposits on any positive match.

        Range: [0, 1]
        Supports GPU acceleration when ctx.target_vecs is a CUDA tensor.
        """
        sim = Heuristics.semantic_similarity_unnormalized(ctx)
        # Handle both numpy and torch
        if _TORCH_AVAILABLE and isinstance(sim, torch.Tensor):
            return torch.clamp(sim, min=0.0)
        return np.maximum(0.0, sim)
    
    @staticmethod
    @HeuristicRegistry.register_deposit(HeuristicKey.EXPLORATION_BONUS)
    def deposit_exploration_bonus(
        ctx: HeuristicContext,
        base_deposit: float = 1.0,
        fresh_multiplier: float = 2.0,
        high_traffic_multiplier: float = 0.5
    ) -> Any:
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
        traffic_ratio = np.clip(traffic_ratio, 0.0, 1.0)
            
        multiplier = fresh_multiplier - (fresh_multiplier - high_traffic_multiplier) * traffic_ratio
        return base_deposit * multiplier
    
    @staticmethod
    @HeuristicRegistry.register_deposit(HeuristicKey.COLLABORATIVE_AMP)
    def deposit_collaborative_amplification(
        ctx: HeuristicContext,
        base_deposit: float = 1.0,
        amplification_factor: float = 1.0,
        max_multiplier: float = 5.0
    ) -> float:
        """
        The more pheromone already present, the larger the new deposit (Exploitation).
        This creates a "rich get richer" effect.
        """
        multiplier = 1.0 + (amplification_factor * ctx.pheromone_values)
        return base_deposit * np.minimum(multiplier, max_multiplier)
    
