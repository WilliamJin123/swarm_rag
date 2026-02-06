"""
Known good genome configurations for warm-starting evolution.

These configurations provide a strong baseline and reduce wasted
generations discovering basic effective strategies. Based on empirical
results showing these configurations consistently perform well.

This module uses the unified seed configurations from seed_configs.py
and converts them to expression tree format for expression_tree mode.
"""
from typing import Dict, Any, List

from ..genome import Genome, FIXED_PARAMS
from ..expressions import ExpressionNode
from .seed_configs import (
    SEED_CONFIGS,
    get_all_expression_tree_configs,
)


# =============================================================================
# Seed Genome Configurations (Expression Tree Format)
# =============================================================================
# Generated from unified SEED_CONFIGS in seed_configs.py
# Expression tree strings are converted from weight dictionaries

SEED_GENOMES: List[Dict[str, Any]] = get_all_expression_tree_configs()


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
