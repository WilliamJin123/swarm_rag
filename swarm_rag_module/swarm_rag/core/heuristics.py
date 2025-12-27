from typing import Any, Dict, Optional
import numpy as np
import math
from dataclasses import dataclass, field
from ..interfaces.base import GraphStore

@dataclass
class HeuristicContext:
    """A shared dataclass to hold context for heuristic functions."""
    query_vec: np.ndarray
    target_vec: np.ndarray
    target_id: int
    graph: GraphStore
    pheromones: Dict[int, float] = field(default_factory=dict)
    
    # Optional fields, with default values
    current_id: Optional[int] = None
    max_pheromone: float = 1.0
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

    # --- MOVEMENT HEURISTICS (Agent Step Decision) ---
    
    @staticmethod
    def semantic_similarity(ctx: HeuristicContext) -> float:
        """
        NORMALIZED Cosine Similarity: Maps [-1, 1] to [0, 1].
        - 0.0 = completely opposite direction (cosine = -1)
        - 0.5 = orthogonal/unrelated (cosine = 0)
        - 1.0 = perfect match (cosine = 1)
        """
        q = ctx.query_vec
        t = ctx.target_vec
        cos_sim = np.dot(q, t) / (np.linalg.norm(q) * np.linalg.norm(t) + 1e-8)
        
        # Map [-1, 1] → [0, 1]
        return (cos_sim + 1.0) / 2.0

    @staticmethod
    def semantic_similarity_unnormalized(ctx: HeuristicContext) -> float:
        """
        RAW Cosine Similarity in [-1, 1] for ranking where negative scores are meaningful.
        """
        q = ctx.query_vec
        t = ctx.target_vec
        return np.dot(q, t) / (np.linalg.norm(q) * np.linalg.norm(t) + 1e-8)

    @staticmethod
    def node_centrality_unnormalized(ctx: HeuristicContext) -> float:
        """
        This version requires ctx.graph.degree which may not exist.
        """
        return math.log(1 + ctx.graph.degree[ctx.target_id])
    
    @staticmethod
    def node_centrality(ctx: HeuristicContext) -> float:
        """
        Normalized centrality that works with any GraphStore.
        Requires ctx.graph.degree and ctx.graph.avg_degree
        
        Range: [0, 1] (sigmoid normalization)
        """
        log_degree = math.log(1 + ctx.graph.degree[ctx.target_id])
        avg = math.log(1 + ctx.graph.avg_degree)
        # Sigmoid normalization
        return log_degree / (log_degree + avg)

    @staticmethod
    def pheromone_repulsion(ctx: HeuristicContext) -> float:
        """
        Inverse Pheromone frequency. 
        Returns 1.0 if no one has been there, approaches 0.0 as traffic increases.
        """
        p_val = ctx.pheromones.get(ctx.target_id, 0.0)
        max_p = max(ctx.max_pheromone, 0.0001)
        # Avoid division by zero or extremely small numbers
        return 1.0 - (p_val / max_p)

    @staticmethod
    def random_jitter(ctx):
        """Adds pure chaos to break loops."""
        return np.random.random()

    # --- RANKING HEURISTICS (Final Consensus) ---

    @staticmethod
    def percentage_visited(ctx: HeuristicContext) -> float:
        """Percentage of total agents that visited this node."""
        return ctx.votes / ctx.total_agents
    
    @staticmethod
    def semantic_rank(ctx: HeuristicContext) -> float:
        """
        RAW semantic similarity for final ranking.
        Uses full [-1, 1] range since we want to distinguish good from bad.
        """
        return Heuristics.semantic_similarity_unnormalized(ctx)

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
    def deposit_flat(ctx: HeuristicContext) -> float:
        """Standard Ant Colony: Leave a constant amount (1.0)."""
        return 1.0

    @staticmethod
    def deposit_hub(ctx: HeuristicContext) -> float:
        """Hubs get more pheromones."""
        return Heuristics.node_centrality(ctx)
    
    def deposit_hub_unnormalized(ctx: HeuristicContext) -> float:
        """Hubs get more pheromones. (UNNORMALIZED)"""
        return Heuristics.node_centrality_unnormalized(ctx)
    
    @staticmethod
    def deposit_semantic(ctx: HeuristicContext) -> float:
        """
        Semantic-weighted deposit using NORMALIZED similarity.
        Only deposits on positive matches (similarity > 0.5 in normalized space) with range 0-1.
        """
        normalized_sim = Heuristics.semantic_similarity(ctx)
        if normalized_sim > 0.5:
            return (normalized_sim - 0.5) * 2.0  # Maps [0.5, 1] → [0, 1]
        else:
            return 0.0

    @staticmethod
    def deposit_semantic_unnormalized(ctx: HeuristicContext) -> float:
        """
        Alternative: Uses unnormalized similarity and clamps to [0, 1].
        Deposits on any positive match.
        
        Range: [0, 1]
        """
        unnormalized_sim = Heuristics.semantic_similarity_unnormalized(ctx)
        return max(0.0, unnormalized_sim)
    
    @staticmethod
    def deposit_explorer_bonus(
        ctx: HeuristicContext,
        base_deposit: float = 1.0,
        fresh_multiplier: float = 2.0,
        high_traffic_multiplier: float = 0.5
    ) -> float:
        """
        Args:
        base_deposit: Base amount to deposit
        fresh_multiplier: Multiplier for completely unvisited nodes (default 2.0)
        trafficked_multiplier: Multiplier for maximally visited nodes (default 0.5)
    
        Range: [base_deposit * trafficked_multiplier, base_deposit * fresh_multiplier]
        Default: [0.5, 2.0]
        """
        max_p = max(ctx.max_pheromone, 0.0001)
        current_pheromone = ctx.pheromones.get(ctx.target_id, 0.0)
        
        traffic_ratio = current_pheromone / max_p
            
        multiplier = fresh_multiplier - (fresh_multiplier - high_traffic_multiplier) * traffic_ratio
    
        return base_deposit * multiplier
    
    @staticmethod
    def deposit_collaborative_amplification(
        ctx: HeuristicContext,
        base_deposit: float = 1.0,
        amplification_factor: float = 1.0,
        max_multiplier: float = 5.0
    ) -> float:
        """
        The more pheromone already present, the larger the new deposit.
        This creates a "rich get richer" effect.
        
        Args:
            base_deposit: Base amount to deposit
            amplification_factor: How strongly to amplify existing pheromones (default 1.0)
            max_multiplier: Maximum multiplier cap (default 5.0)
        
        Range: [base_deposit, base_deposit * max_multiplier]
        Default: [1.0, 5.0]
        """
        current_pheromone = ctx.pheromones.get(ctx.target_id, 0.0)
        multiplier = 1.0 + (amplification_factor * current_pheromone)
        return base_deposit * min(multiplier, max_multiplier)
    
