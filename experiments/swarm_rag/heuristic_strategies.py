import numpy as np
import networkx as nx
import math

class Heuristics:
    """
    A library of preset heuristics. 
    Each function takes a `ctx` dictionary and returns a float score.
    """

    # --- MOVEMENT HEURISTICS (Agent Step Decision) ---
    
    @staticmethod
    def semantic_similarity(ctx):
        """Standard Cosine Similarity between Query and Target Node."""
        q = ctx['query_vec']
        t = ctx['target_vec']
        if t is None: return 0.0
        norm_q = np.linalg.norm(q)
        norm_t = np.linalg.norm(t)
        return np.dot(q, t) / (norm_q * norm_t) if norm_q and norm_t else 0.0

    @staticmethod
    def node_authority(ctx):
        """Log-scaled degree centrality. Good for finding 'hubs'."""
        # ctx['graph'] is the NetworkX graph, ctx['target_id'] is the candidate node
        degree = ctx['graph'].degree[ctx['target_id']]
        return math.log(1 + degree)

    @staticmethod
    def pheromone_repulsion(ctx):
        """
        Inverse Pheromone frequency. 
        Returns 1.0 if no one has been there, approaches 0.0 as traffic increases.
        """
        p_val = ctx['pheromones'].get(ctx['target_id'], 0.0)
        max_p = ctx.get('max_pheromone', 1.0)
        # Avoid division by zero or extremely small numbers
        max_p = max(max_p, 0.0001) 
        return 1.0 - (p_val / max_p)

    @staticmethod
    def random_jitter(ctx):
        """Adds pure chaos to break loops."""
        return np.random.random()

    # --- RANKING HEURISTICS (Final Consensus) ---

    @staticmethod
    def consensus_vote(ctx):
        """Percentage of total agents that visited this node."""
        return ctx['votes'] / ctx['total_agents']
    
    @staticmethod
    def semantic_rank(ctx):
        """Semantic similarity for the final ranking phase."""
        # Re-use the logic, but the keys might differ in the final context
        return Heuristics.semantic_similarity(ctx)

    @staticmethod
    def deposit_flat(ctx):
        """Standard Ant Colony: Leave a constant amount (1.0)."""
        return 1.0

    @staticmethod
    def deposit_authority(ctx):
        """Original behavior: Hubs get more pheromones."""
        degree = ctx['graph'].degree[ctx['target_id']]
        return math.log(1 + degree)
    
    @staticmethod
    def deposit_semantic(ctx):
        """
        'Glow' strategy: High semantic match = stronger pheromone trail.
        Good for guiding other agents to relevant pockets.
        """
        return Heuristics.semantic_similarity(ctx)



# Dictionary for string-based lookup (optional, for easy config files)
PRESET_REGISTRY = {
    "semantic": Heuristics.semantic_similarity,
    "authority": Heuristics.node_authority,
    "diversity": Heuristics.pheromone_repulsion,
    "jitter": Heuristics.random_jitter,
    "consensus": Heuristics.consensus_vote
}