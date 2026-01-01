from typing import Any, Dict, List, Optional, Union
import numpy as np
import math
from dataclasses import dataclass, field
from ..interfaces.base import GraphStore


class HeuristicRegistry:
    _movement_registry = {}
    _ranking_registry = {}
    _deposit_registry = {}

    @classmethod
    def register_movement(cls, name: str):
        """Decorator for movement heuristics."""
        def decorator(fn):
            cls._movement_registry[name] = fn
            return fn
        return decorator

    @classmethod
    def register_ranking(cls, name: str):
        """Decorator for ranking heuristics."""
        def decorator(fn):
            cls._ranking_registry[name] = fn
            return fn
        return decorator

    @classmethod
    def register_deposit(cls, name: str):
        """Decorator for deposit heuristics."""
        def decorator(fn):
            cls._deposit_registry[name] = fn
            return fn
        return decorator

    @classmethod
    def get_movement(cls, name: str):
        """Retrieve a movement heuristic."""
        return cls._movement_registry[name]

    @classmethod
    def get_ranking(cls, name: str):
        """Retrieve a ranking heuristic."""
        return cls._ranking_registry[name]

    @classmethod
    def get_deposit(cls, name: str):
        """Retrieve a deposit heuristic."""
        return cls._deposit_registry[name]

    @classmethod
    def all_movement(cls):
        """Get all movement heuristics."""
        return cls._movement_registry

    @classmethod
    def all_ranking(cls):
        """Get all ranking heuristics."""
        return cls._ranking_registry

    @classmethod
    def all_deposit(cls):
        """Get all deposit heuristics."""
        return cls._deposit_registry
    
    @classmethod
    def all(cls):
        return {
            **cls._movement_registry,
            **cls._deposit_registry,
            **cls._ranking_registry
        }
    
    @classmethod
    def get(cls, name: str):
        """
        Allows EvolutionEngine to find a function by name 
        without knowing which specific registry it lives in.
        """
        if name in cls._movement_registry:
            return cls._movement_registry[name]
        if name in cls._ranking_registry:
            return cls._ranking_registry[name]
        if name in cls._deposit_registry:
            return cls._deposit_registry[name]
        
        raise ValueError(f"Heuristic '{name}' not found in any registry.")

@dataclass(slots=True)
class HeuristicContext:
    """A shared dataclass to hold context for heuristic functions."""
    query_vec: np.ndarray # Shape: (D,)
    target_vecs: Optional[np.ndarray] = None    
    target_ids: Union[List[int], np.ndarray] = None

    pheromone_values: np.ndarray = field(default_factory=lambda: np.array([]))
    node_degrees: np.ndarray = field(default_factory=lambda: np.array([]))
   
    graph: Optional[GraphStore] = None
    max_pheromone: float = 1.0
    avg_degree: float = 1.0
    step_index: int = 0
    agent_index: int = 0
    votes: int = 0
    total_agents: int = 0

    extra_data: Dict[str, Any] = field(default_factory=dict)

class Heuristics:
    """
    A library of preset heuristics. 
    Each function takes a `HeuristicContext` object and returns a float score, normalized to [0,1]
    """

    # --- MOVEMENT HEURISTICS ---
    
    @staticmethod
    @HeuristicRegistry.register_movement("semantic_similarity")
    def semantic_similarity(ctx: HeuristicContext) -> np.ndarray:
        """
        NORMALIZED Cosine Similarity: Maps [-1, 1] to [0, 1].
        - 0.0 = completely opposite direction (cosine = -1)
        - 0.5 = orthogonal/unrelated (cosine = 0)
        - 1.0 = perfect match (cosine = 1)
        """
        scores = np.dot(ctx.target_vecs, ctx.query_vec)
        # Normalize from [-1, 1] to [0, 1]
        return (scores + 1.0) / 2.0

    @staticmethod
    @HeuristicRegistry.register_movement("semantic_similarity_unnormalized")
    def semantic_similarity_unnormalized(ctx: HeuristicContext) -> np.ndarray:
        """
        RAW Cosine Similarity in [-1, 1] for ranking where negative scores are meaningful.
        """
        return np.dot(ctx.target_vecs, ctx.query_vec)
    
    @staticmethod
    @HeuristicRegistry.register_movement("node_centrality")
    def node_centrality(ctx: HeuristicContext) -> np.ndarray:
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
    @HeuristicRegistry.register_movement("pheromone_repulsion")
    def pheromone_repulsion(ctx: HeuristicContext) -> np.ndarray:
        """
        Inverse Pheromone frequency. 
        Returns 1.0 if no one has been there, approaches 0.0 as traffic increases.
        """
        max_p = max(ctx.max_pheromone, 0.0001)
        return 1.0 - (ctx.pheromone_values / max_p)

    @staticmethod
    @HeuristicRegistry.register_movement("random_jitter")
    def random_jitter(ctx: HeuristicContext) -> np.ndarray:
        """Adds pure chaos to break loops."""
        count = len(ctx.target_ids) if ctx.target_ids is not None else 1
        return np.random.random(count)

    # --- RANKING HEURISTICS (Final Consensus) ---

    @staticmethod
    @HeuristicRegistry.register_ranking("percentage_visited")
    def percentage_visited(ctx: HeuristicContext) -> float:
        """Percentage of total agents that visited this node."""
        if ctx.total_agents == 0: return 0.0
        return ctx.votes / ctx.total_agents
    
    @staticmethod
    @HeuristicRegistry.register_ranking("semantic_rank")
    def semantic_rank(ctx: HeuristicContext) -> float:
        """
        RAW semantic similarity for final ranking.
        Uses full [-1, 1] range since we want to distinguish good from bad.
        """
        val = Heuristics.semantic_similarity_unnormalized(ctx)
        return val.item() if isinstance(val, np.ndarray) else val

    # @staticmethod
    # def edge_type_preference(
    #     ctx: HeuristicContext,
    #     edge_type_dict: Optional[Dict[Any, Any]] = None,
    #     edge_weights: Optional[Dict[Any, float]] = None,
    #     default_weight: float = 0.5
    # ) -> float:
    #     """
    #     Weights neighbors based on the edge type connecting current node to target.
    #     Useful for STARK where edge types indicate relationship types.
        
    #     Args:
    #         edge_type_dict: Dict mapping edge_type_id -> edge_type_name
    #                     e.g. {0: "treats", 1: "metabolizes", 2: "indicates"}
    #         edge_weights: Dict mapping edge_type_name -> weight score [0, 1]
    #                     e.g. {"treats": 1.0, "metabolizes": 0.1, "indicates": 0.8}
    #         default_weight: Weight for edge types not in edge_weights dict (default 0.5)
        
    #     Returns:
    #         Weight in [0, 1] based on edge type relevance
        
    #     Example usage:
    #         # User provides edge type mapping
    #         edge_type_dict = {0: "treats", 1: "metabolizes", 2: "indicates"}
            
    #         # LLM determines relevant edges ONCE per query
    #         edge_weights = {"treats": 1.0, "indicates": 0.8, "metabolizes": 0.0}
    #     """
    #     if edge_type_dict is None or edge_weights is None:
    #         return default_weight
    #     edge_type_id = ctx.extra_data.get('edge_type_id')
    
    #     if edge_type_id is None:
    #         return default_weight
        
    #     # Map edge type ID to name
    #     edge_type_name = edge_type_dict.get(edge_type_id)
    
    #     if edge_type_name is None:
    #         return default_weight
        
    #     # Return weight for this edge type
    #     return edge_weights.get(edge_type_name, default_weight)

    # --- DEPOSIT HEURISTICS ---

    @staticmethod
    @HeuristicRegistry.register_deposit("flat")
    def deposit_flat(ctx: HeuristicContext) -> np.ndarray:
        """Returns array of 1.0s matching input size."""
        return np.ones_like(ctx.pheromone_values)

    @staticmethod
    @HeuristicRegistry.register_deposit("hub")
    def deposit_hub(ctx: HeuristicContext) -> np.ndarray:
        """Hubs get more pheromones."""
        return Heuristics.node_centrality(ctx)
    
    @staticmethod
    @HeuristicRegistry.register_deposit("semantic")
    def deposit_semantic(ctx: HeuristicContext) -> np.ndarray:
        """
        Semantic-weighted deposit using NORMALIZED similarity.
        Only deposits on positive matches (similarity > 0.5 in normalized space) with range 0-1.
        """
        normalized_sim = Heuristics.semantic_similarity(ctx)
        return np.where(normalized_sim > 0.5, (normalized_sim - 0.5) * 2.0, 0.0)

    @staticmethod
    @HeuristicRegistry.register_deposit("semantic_unnormalized")
    def deposit_semantic_unnormalized(ctx: HeuristicContext) -> np.ndarray:
        """
        Alternative: Uses unnormalized similarity and clamps to [0, 1].
        Deposits on any positive match.
        
        Range: [0, 1]
        """
        sim = Heuristics.semantic_similarity_unnormalized(ctx)
        return np.maximum(0.0, sim)
    
    @staticmethod
    @HeuristicRegistry.register_deposit("exploration_bonus")
    def deposit_exploration_bonus(
        ctx: HeuristicContext,
        base_deposit: float = 1.0,
        fresh_multiplier: float = 2.0,
        high_traffic_multiplier: float = 0.5
    ) -> np.ndarray:
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
    @HeuristicRegistry.register_deposit("collaborative_amp")
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
    
