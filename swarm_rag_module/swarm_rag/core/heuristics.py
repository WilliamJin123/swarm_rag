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
    Each function takes a `HeuristicContext` object and returns a float score.
    """

    # --- MOVEMENT HEURISTICS (Agent Step Decision) ---
    
    @staticmethod
    def semantic_similarity(ctx: HeuristicContext) -> float:
        """Standard Cosine Similarity between Query and Target Node."""
        q = ctx.query_vec
        t = ctx.target_vec
        return np.dot(q, t) / (
            np.linalg.norm(q) * np.linalg.norm(t) + 1e-8
        )

    @staticmethod
    def node_centrality(ctx: HeuristicContext) -> float:
        """Log-scaled degree centrality. Good for finding 'hubs'. NOTE: Your GraphStore object must have a self.degree attribute or this will throw."""
        # ctx.graph is the NetworkX graph, ctx.target_id is the candidate node
        degree = ctx.graph.degree[ctx.target_id]
        return math.log(1 + degree)

    @staticmethod
    def pheromone_repulsion(ctx: HeuristicContext) -> float:
        """
        Inverse Pheromone frequency. 
        Returns 1.0 if no one has been there, approaches 0.0 as traffic increases.
        """
        p_val = ctx.pheromones.get(ctx.target_id, 0.0)
        max_p = ctx.max_pheromone
        # Avoid division by zero or extremely small numbers
        max_p = max(max_p, 0.0001) 
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
        """Semantic similarity for the final ranking phase."""
        return Heuristics.semantic_similarity(ctx)

    # --- DEPOSIT HEURISTICS ---

    @staticmethod
    def deposit_flat(ctx: HeuristicContext) -> float:
        """Standard Ant Colony: Leave a constant amount (1.0)."""
        return 1.0

    @staticmethod
    def deposit_hub(ctx: HeuristicContext) -> float:
        """Original behavior: Hubs get more pheromones."""
        return Heuristics.node_centrality(ctx)
    
    @staticmethod
    def deposit_semantic(ctx: HeuristicContext) -> float:
        """
        'Glow' strategy: High semantic match = stronger pheromone trail.
        Good for guiding other agents to relevant pockets.
        """
        return Heuristics.semantic_similarity(ctx)

    @staticmethod
    def deposit_explorer_bonus(ctx: HeuristicContext) -> float:
        """
        Rewards agents for visiting nodes with low pheromone traffic.
        """
        base_deposit = 1.0
        max_p = ctx.max_pheromone
        current_pheromone = ctx.pheromones.get(ctx.target_id, 0.0)
        
        # If no one has been there, get a full bonus. As traffic increases, bonus decreases.
        traffic_ratio = current_pheromone / max_p if max_p > 0 else 0
        
        # The deposit is scaled by how "fresh" the location is.
        return base_deposit * (1.0 - traffic_ratio)
    
    @staticmethod
    def deposit_collaborative_amplification(ctx: HeuristicContext) -> float:
        """
        The more pheromone already present, the larger the new deposit.
        This creates a "rich get richer" effect.
        """
        base_deposit = 1.0
        current_pheromone = ctx.pheromones.get(ctx.target_id, 0.0)
        return base_deposit * min(1.0 + current_pheromone, 5.0)  # Cap at 5x
    
