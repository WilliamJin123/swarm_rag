from enum import Enum

class HeuristicKey(Enum):
    """Enum for all CORE, built-in heuristic function names."""
    # --- MOVEMENT HEURISTICS ---
    SEMANTIC_SIMILARITY = "semantic_similarity"
    SEMANTIC_SIMILARITY_UNNORMALIZED = "semantic_similarity_unnormalized"
    NODE_CENTRALITY = "node_centrality"
    PHEROMONE_REPULSION = "pheromone_repulsion"
    RANDOM_JITTER = "random_jitter"

    # --- RANKING HEURISTICS ---
    PERCENTAGE_VISITED = "percentage_visited"
    SEMANTIC_RANK = "semantic_rank"

    # --- DEPOSIT HEURISTICS ---
    FLAT = "flat"
    HUB = "hub"
    SEMANTIC = "semantic"
    SEMANTIC_UNNORMALIZED = "semantic_unnormalized"
    EXPLORATION_BONUS = "exploration_bonus"
    COLLABORATIVE_AMP = "collaborative_amplification"
    
    # --- INTEGRATIONS ---
    STARK_CENTRALITY = "stark_centrality"


class GeneticKey(Enum):
    """Enum for all CORE, built-in genetic operator names."""
    # --- SELECTION ---
    TOURNAMENT = "tournament"
    BOLTZMANN = "boltzmann"

    # --- CROSSOVER ---
    UNIFORM_PARAMETER_MIX = "uniform_parameter_mix"
    SUBTREE_CROSSOVER = "subtree_crossover"
    ROOT_MIX_CROSSOVER = "root_mix_crossover"

    # --- MUTATION ---
    EXPRESSION_TREE_MUTATION = "expression_tree_mutation"
    AGGRESSIVE_MUTATION = "aggressive_mutation"
    GUIDED_MUTATION = "guided_mutation"
    LLM_MUTATION = "llm_mutation"
    FOCUSED_MUTATION = "focused_mutation"

    # --- CREATION ---
    STANDARD_INITIALIZATION = "standard_initialization"
    SHALLOW_GROWTH_INITIALIZATION = "shallow_growth_initialization"
    SEEDED_INITIALIZATION = "seeded_initialization"
    BASELINE_SEEDED_INITIALIZATION = "baseline_seeded_initialization"