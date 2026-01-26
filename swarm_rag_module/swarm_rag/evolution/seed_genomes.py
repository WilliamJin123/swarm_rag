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

SEED_GENOMES: List[Dict[str, Any]] = [
    # High-semantic config with balanced exploration
    {
        "name": "semantic_balanced",
        "n_agents": 25,
        "steps": 4,
        "decay": 0.5,
        "initial_pool_size": 30,
        "movement_tree": "ADD(MUL(semantic_similarity, 0.7), MUL(node_centrality, 0.3))",
        "deposit_tree": "semantic_similarity",
        "ranking_tree": "ADD(MUL(percentage_visited, 0.4), MUL(semantic_rank, 0.6))",
    },
    # Hub-explorer config - emphasizes graph structure
    {
        "name": "hub_explorer",
        "n_agents": 30,
        "steps": 5,
        "decay": 0.6,
        "initial_pool_size": 40,
        "movement_tree": "ADD(ADD(MUL(node_centrality, 0.5), MUL(semantic_similarity, 0.3)), MUL(pheromone_repulsion, 0.2))",
        "deposit_tree": "node_centrality",
        "ranking_tree": "ADD(MUL(pheromone_level, 0.3), MUL(semantic_rank, 0.7))",
    },
    # Diversity-focused config - avoids clustering
    {
        "name": "diversity_focused",
        "n_agents": 20,
        "steps": 4,
        "decay": 0.4,
        "initial_pool_size": 35,
        "movement_tree": "ADD(MUL(pheromone_repulsion, 0.4), MUL(semantic_similarity, 0.6))",
        "deposit_tree": "MUL(semantic_similarity, pheromone_repulsion)",
        "ranking_tree": "ADD(MUL(percentage_visited, 0.5), MUL(pheromone_level, 0.5))",
    },
    # Conservative config - fewer agents, more steps
    {
        "name": "conservative_deep",
        "n_agents": 18,
        "steps": 6,
        "decay": 0.7,
        "initial_pool_size": 25,
        "movement_tree": "ADD(MUL(semantic_similarity, 0.8), MUL(node_centrality, 0.2))",
        "deposit_tree": "semantic_similarity",
        "ranking_tree": "semantic_rank",
    },
    # Aggressive exploration config
    {
        "name": "aggressive_explorer",
        "n_agents": 45,
        "steps": 3,
        "decay": 0.35,
        "initial_pool_size": 50,
        "movement_tree": "ADD(MUL(pheromone_repulsion, 0.5), MUL(semantic_similarity, 0.5))",
        "deposit_tree": "ADD(semantic_similarity, node_centrality)",
        "ranking_tree": "ADD(MUL(percentage_visited, 0.6), MUL(semantic_rank, 0.4))",
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
