"""
Known good genome configurations for warm-starting evolution.

These configurations provide a strong baseline and reduce wasted
generations discovering basic effective strategies. Based on empirical
results showing these configurations consistently perform well.
"""
from typing import Dict, Any, List

from .types.genome import Genome, FIXED_PARAMS
from .types.expressions import ExpressionNode


# =============================================================================
# Seed Genome Configurations
# =============================================================================

# =============================================================================
# Seed Genome Configurations
# Target metrics: Hit@1>60%, Hit@5>80%, MRR>80%, Recall@20>85%
# =============================================================================
SEED_GENOMES: List[Dict[str, Any]] = [
    # --- PRECISION-FOCUSED (Hit@1, MRR) ---
    # High semantic similarity for precise top-1 results
    {
        "name": "precision_semantic",
        "n_agents": 20,
        "steps": 6,
        "decay": 0.6,
        "initial_pool_size": 30,
        "movement_tree": "ADD(ADD(MUL(semantic_similarity_unnormalized, 0.6), MUL(stark_centrality, 0.2)), MUL(node_centrality, 0.2))",
        "deposit_tree": "ADD(ADD(MUL(semantic_unnormalized, 0.4), MUL(hub, 0.3)), MUL(collaborative_amplification, 0.3))",
        "ranking_tree": "ADD(MUL(semantic_rank, 0.8), MUL(percentage_visited, 0.2))",
    },
    # Hub highways - fast paths to relevant content
    {
        "name": "precision_hub_highways",
        "n_agents": 25,
        "steps": 5,
        "decay": 0.5,
        "initial_pool_size": 35,
        "movement_tree": "ADD(ADD(MUL(stark_centrality, 0.35), MUL(semantic_similarity_unnormalized, 0.35)), MUL(node_centrality, 0.3))",
        "deposit_tree": "ADD(MUL(hub, 0.5), MUL(collaborative_amplification, 0.5))",
        "ranking_tree": "ADD(MUL(semantic_rank, 0.7), MUL(percentage_visited, 0.3))",
    },
    # Consensus-driven - multiple agents agree = high quality
    {
        "name": "precision_consensus",
        "n_agents": 30,
        "steps": 5,
        "decay": 0.55,
        "initial_pool_size": 35,
        "movement_tree": "ADD(ADD(MUL(semantic_similarity_unnormalized, 0.45), MUL(stark_centrality, 0.25)), MUL(random_jitter, 0.3))",
        "deposit_tree": "ADD(MUL(collaborative_amplification, 0.5), MUL(semantic_unnormalized, 0.5))",
        "ranking_tree": "ADD(MUL(percentage_visited, 0.5), MUL(semantic_rank, 0.5))",
    },

    # --- RECALL-FOCUSED (Recall@20) ---
    # High exploration - maximize coverage
    {
        "name": "recall_explorer",
        "n_agents": 40,
        "steps": 4,
        "decay": 0.4,
        "initial_pool_size": 50,
        "movement_tree": "ADD(ADD(MUL(pheromone_repulsion, 0.3), MUL(random_jitter, 0.3)), MUL(semantic_similarity_unnormalized, 0.4))",
        "deposit_tree": "ADD(MUL(exploration_bonus, 0.5), MUL(flat, 0.5))",
        "ranking_tree": "ADD(MUL(percentage_visited, 0.4), MUL(semantic_rank, 0.6))",
    },
    # Wide swarm - many agents, shallow depth
    {
        "name": "recall_wide_swarm",
        "n_agents": 50,
        "steps": 3,
        "decay": 0.35,
        "initial_pool_size": 60,
        "movement_tree": "ADD(ADD(MUL(semantic_similarity_unnormalized, 0.35), MUL(random_jitter, 0.35)), MUL(pheromone_repulsion, 0.3))",
        "deposit_tree": "ADD(MUL(exploration_bonus, 0.4), MUL(flat, 0.6))",
        "ranking_tree": "ADD(MUL(semantic_rank, 0.6), MUL(percentage_visited, 0.4))",
    },
    # Repulsion-driven - spread agents across graph
    {
        "name": "recall_repulsion",
        "n_agents": 35,
        "steps": 4,
        "decay": 0.3,
        "initial_pool_size": 45,
        "movement_tree": "ADD(ADD(MUL(pheromone_repulsion, 0.4), MUL(semantic_similarity_unnormalized, 0.35)), MUL(stark_centrality, 0.25))",
        "deposit_tree": "ADD(MUL(exploration_bonus, 0.6), MUL(hub, 0.4))",
        "ranking_tree": "ADD(MUL(semantic_rank, 0.7), MUL(percentage_visited, 0.3))",
    },

    # --- BALANCED VARIANTS ---
    # Balanced baseline with all new features
    {
        "name": "balanced_baseline",
        "n_agents": 25,
        "steps": 5,
        "decay": 0.5,
        "initial_pool_size": 35,
        "movement_tree": "ADD(ADD(ADD(MUL(semantic_similarity_unnormalized, 0.4), MUL(stark_centrality, 0.25)), MUL(pheromone_repulsion, 0.2)), MUL(random_jitter, 0.15))",
        "deposit_tree": "ADD(ADD(MUL(semantic_unnormalized, 0.35), MUL(exploration_bonus, 0.35)), MUL(hub, 0.3))",
        "ranking_tree": "ADD(MUL(semantic_rank, 0.75), MUL(percentage_visited, 0.25))",
    },
    # Deep search - fewer agents, more steps
    {
        "name": "balanced_deep",
        "n_agents": 18,
        "steps": 8,
        "decay": 0.7,
        "initial_pool_size": 25,
        "movement_tree": "ADD(ADD(MUL(semantic_similarity_unnormalized, 0.5), MUL(stark_centrality, 0.3)), MUL(node_centrality, 0.2))",
        "deposit_tree": "ADD(MUL(semantic_unnormalized, 0.5), MUL(collaborative_amplification, 0.5))",
        "ranking_tree": "semantic_rank",
    },
    # STaRK-graph optimized
    {
        "name": "balanced_stark_graph",
        "n_agents": 28,
        "steps": 5,
        "decay": 0.5,
        "initial_pool_size": 40,
        "movement_tree": "ADD(ADD(MUL(stark_centrality, 0.4), MUL(semantic_similarity_unnormalized, 0.35)), MUL(node_centrality, 0.25))",
        "deposit_tree": "ADD(ADD(MUL(hub, 0.4), MUL(semantic_unnormalized, 0.3)), MUL(exploration_bonus, 0.3))",
        "ranking_tree": "ADD(MUL(semantic_rank, 0.8), MUL(percentage_visited, 0.2))",
    },
    # Hybrid precision-recall
    {
        "name": "balanced_hybrid",
        "n_agents": 32,
        "steps": 5,
        "decay": 0.45,
        "initial_pool_size": 40,
        "movement_tree": "ADD(ADD(ADD(MUL(semantic_similarity_unnormalized, 0.35), MUL(stark_centrality, 0.25)), MUL(pheromone_repulsion, 0.2)), MUL(random_jitter, 0.2))",
        "deposit_tree": "ADD(ADD(MUL(semantic_unnormalized, 0.3), MUL(exploration_bonus, 0.3)), MUL(collaborative_amplification, 0.4))",
        "ranking_tree": "ADD(MUL(semantic_rank, 0.65), MUL(percentage_visited, 0.35))",
    },
]


# =============================================================================
# Tree Parsing Functions
# =============================================================================

def _parse_simple_tree(expr_str: str) -> ExpressionNode:
    """
    Parse a simple expression string into an ExpressionNode tree.

    Supported formats:
    - "feature_name" -> feature node
    - "0.5" -> constant node
    - "ADD(x, y)" -> operator node
    - "MUL(x, y)" -> operator node

    Args:
        expr_str: Expression string like "ADD(MUL(x, 0.5), y)"

    Returns:
        Root ExpressionNode of the parsed tree
    """
    expr_str = expr_str.strip()

    # Check for function/operator call: NAME(args)
    if "(" in expr_str and expr_str.endswith(")"):
        # Find the function name
        paren_idx = expr_str.index("(")
        func_name = expr_str[:paren_idx].upper()
        args_str = expr_str[paren_idx + 1 : -1]

        # Map function names to operator values
        op_map = {
            "ADD": "+",
            "MUL": "*",
            "SUB": "-",
            "DIV": "/",
            "MAX": "max",
            "MIN": "min",
        }

        # Parse arguments (handle nested parentheses)
        args = _split_args(args_str)

        if func_name in op_map:
            children = [_parse_simple_tree(arg) for arg in args]
            return ExpressionNode(type="op", value=op_map[func_name], children=children)
        else:
            # Unknown function, treat as unary function
            children = [_parse_simple_tree(args[0])] if args else []
            return ExpressionNode(type="func", value=func_name.lower(), children=children)

    # Try to parse as a number (constant)
    try:
        value = float(expr_str)
        return ExpressionNode(type="const", value=value)
    except ValueError:
        pass

    # Otherwise, treat as a feature name
    return ExpressionNode(type="feature", value=expr_str)


def _split_args(args_str: str) -> List[str]:
    """
    Split comma-separated arguments, respecting nested parentheses.

    Args:
        args_str: String like "MUL(x, 0.5), y"

    Returns:
        List of argument strings: ["MUL(x, 0.5)", "y"]
    """
    args = []
    current = ""
    depth = 0

    for char in args_str:
        if char == "(":
            depth += 1
            current += char
        elif char == ")":
            depth -= 1
            current += char
        elif char == "," and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += char

    if current.strip():
        args.append(current.strip())

    return args


# =============================================================================
# Seed Genome Creation
# =============================================================================

def create_seed_genome(seed_config: Dict[str, Any]) -> Genome:
    """
    Create a Genome from a seed configuration.

    Args:
        seed_config: Dictionary with seed parameters and tree strings

    Returns:
        Initialized Genome with fixed params and seed configuration
    """
    # Start with fixed params
    params = dict(FIXED_PARAMS)

    # Add evolvable params from seed
    params["n_agents"] = seed_config["n_agents"]
    params["steps"] = seed_config["steps"]
    params["decay"] = seed_config["decay"]
    params["initial_pool_size"] = seed_config["initial_pool_size"]

    # Parse expression trees
    strategies = {}
    strategies["ranking"] = _parse_simple_tree(seed_config.get("ranking_tree", "semantic_rank"))

    # Parse movement and deposit trees for group 0 (single group for seeds)
    movement_tree = _parse_simple_tree(seed_config.get("movement_tree", "semantic_similarity"))
    deposit_tree = _parse_simple_tree(seed_config.get("deposit_tree", "flat"))

    strategies["g0_movement"] = movement_tree
    strategies["g0_deposit"] = deposit_tree

    # Create genome with unique ID based on seed name
    seed_name = seed_config.get("name", "seed")
    genome_id = f"seed_{seed_name}"

    return Genome(
        id=genome_id,
        params=params,
        strategies=strategies,
        group_ratios={"g0": 1.0},
        evaluated=False,
    )


def get_all_seed_genomes() -> List[Genome]:
    """
    Create and return all seed genomes.

    Returns:
        List of Genome objects created from SEED_GENOMES configurations
    """
    return [create_seed_genome(config) for config in SEED_GENOMES]
