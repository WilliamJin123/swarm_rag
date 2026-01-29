"""
Unified Seed Genome Configurations

Single source of truth for seed configurations used by both:
- expression_tree mode (SEED_GENOMES in seed_genomes.py)
- weighted_sum mode (SEED_VARIANTS in weighted_sum.py)

Configurations use weight dictionaries which can be:
- Used directly for weighted_sum mode
- Converted to expression trees for expression_tree mode

Target metrics: Hit@1>60%, Hit@5>80%, MRR>80%, Recall@20>85%
"""
from typing import Dict, Any, List, Optional


# =============================================================================
# Baseline Configuration
# =============================================================================

# Baseline hyperparameters shared across seeds
BASELINE_HYPERPARAMS: Dict[str, Any] = {
    "n_agents": 25,
    "steps": 5,
    "decay": 0.5,
    "initial_pool_size": 30,
}

# Baseline weights - balanced starting point
BASELINE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "movement_weights": {
        "semantic_similarity_unnormalized": 0.40,
        "stark_centrality": 0.20,
        "node_centrality": 0.15,
        "pheromone_repulsion": 0.15,
        "random_jitter": 0.10,
    },
    "deposit_weights": {
        "flat": 0.30,
        "semantic_unnormalized": 0.25,
        "exploration_bonus": 0.20,
        "hub": 0.15,
        "collaborative_amplification": 0.10,
    },
    "ranking_weights": {
        "semantic_rank": 0.80,
        "percentage_visited": 0.20,
    },
}


# =============================================================================
# Seed Configurations - Single Source of Truth
# =============================================================================
# Each seed specifies:
# - name: Unique identifier
# - hyperparams: Override BASELINE_HYPERPARAMS (optional)
# - weights: Override BASELINE_WEIGHTS (optional)
# - notes: Human-readable description (optional)

SEED_CONFIGS: List[Dict[str, Any]] = [
    # --- BASELINE ---
    {
        "name": "baseline_balanced",
        "notes": "Balanced configuration - good all-around performance",
    },

    # --- PRECISION-FOCUSED (Hit@1, MRR) ---
    {
        "name": "precision_semantic",
        "notes": "High semantic similarity for precise top-1 results",
        "hyperparams": {"n_agents": 20, "steps": 6, "decay": 0.6},
        "weights": {
            "movement_weights": {
                "semantic_similarity_unnormalized": 0.60,
                "stark_centrality": 0.15,
                "node_centrality": 0.10,
                "pheromone_repulsion": 0.10,
                "random_jitter": 0.05,
            },
            "deposit_weights": {
                "flat": 0.10,
                "semantic_unnormalized": 0.40,
                "exploration_bonus": 0.10,
                "hub": 0.20,
                "collaborative_amplification": 0.20,
            },
        },
    },
    {
        "name": "precision_hub_highways",
        "notes": "Hub highways - fast paths to relevant content",
        "hyperparams": {"steps": 5, "decay": 0.5, "initial_pool_size": 35},
        "weights": {
            "movement_weights": {
                "semantic_similarity_unnormalized": 0.35,
                "stark_centrality": 0.30,
                "node_centrality": 0.20,
                "pheromone_repulsion": 0.10,
                "random_jitter": 0.05,
            },
            "deposit_weights": {
                "flat": 0.10,
                "semantic_unnormalized": 0.20,
                "exploration_bonus": 0.10,
                "hub": 0.35,
                "collaborative_amplification": 0.25,
            },
            "ranking_weights": {
                "semantic_rank": 0.70,
                "percentage_visited": 0.30,
            },
        },
    },
    {
        "name": "precision_consensus",
        "notes": "Consensus-driven - multiple agents agree = high quality",
        "hyperparams": {"n_agents": 30, "steps": 5, "decay": 0.55, "initial_pool_size": 35},
        "weights": {
            "movement_weights": {
                "semantic_similarity_unnormalized": 0.45,
                "stark_centrality": 0.20,
                "node_centrality": 0.15,
                "pheromone_repulsion": 0.05,
                "random_jitter": 0.15,
            },
            "deposit_weights": {
                "flat": 0.05,
                "semantic_unnormalized": 0.25,
                "exploration_bonus": 0.10,
                "hub": 0.20,
                "collaborative_amplification": 0.40,
            },
            "ranking_weights": {
                "semantic_rank": 0.50,
                "percentage_visited": 0.50,
            },
        },
    },

    # --- RECALL-FOCUSED (Recall@20) ---
    {
        "name": "recall_explorer",
        "notes": "High exploration - maximize coverage",
        "hyperparams": {"n_agents": 40, "steps": 4, "decay": 0.4, "initial_pool_size": 50},
        "weights": {
            "movement_weights": {
                "semantic_similarity_unnormalized": 0.25,
                "stark_centrality": 0.15,
                "node_centrality": 0.15,
                "pheromone_repulsion": 0.25,
                "random_jitter": 0.20,
            },
            "deposit_weights": {
                "flat": 0.20,
                "semantic_unnormalized": 0.15,
                "exploration_bonus": 0.40,
                "hub": 0.15,
                "collaborative_amplification": 0.10,
            },
            "ranking_weights": {
                "semantic_rank": 0.60,
                "percentage_visited": 0.40,
            },
        },
    },
    {
        "name": "recall_wide_swarm",
        "notes": "Wide swarm - many agents, shallow depth",
        "hyperparams": {"n_agents": 50, "steps": 3, "decay": 0.35, "initial_pool_size": 60},
        "weights": {
            "movement_weights": {
                "semantic_similarity_unnormalized": 0.30,
                "stark_centrality": 0.20,
                "node_centrality": 0.10,
                "pheromone_repulsion": 0.20,
                "random_jitter": 0.20,
            },
            "deposit_weights": {
                "flat": 0.30,
                "semantic_unnormalized": 0.20,
                "exploration_bonus": 0.30,
                "hub": 0.10,
                "collaborative_amplification": 0.10,
            },
            "ranking_weights": {
                "semantic_rank": 0.60,
                "percentage_visited": 0.40,
            },
        },
    },
    {
        "name": "recall_repulsion",
        "notes": "Repulsion-driven - spread agents across graph",
        "hyperparams": {"n_agents": 35, "steps": 4, "decay": 0.3, "initial_pool_size": 45},
        "weights": {
            "movement_weights": {
                "semantic_similarity_unnormalized": 0.30,
                "stark_centrality": 0.15,
                "node_centrality": 0.10,
                "pheromone_repulsion": 0.35,
                "random_jitter": 0.10,
            },
            "deposit_weights": {
                "flat": 0.15,
                "semantic_unnormalized": 0.15,
                "exploration_bonus": 0.45,
                "hub": 0.15,
                "collaborative_amplification": 0.10,
            },
            "ranking_weights": {
                "semantic_rank": 0.70,
                "percentage_visited": 0.30,
            },
        },
    },

    # --- BALANCED VARIANTS ---
    {
        "name": "balanced_deep",
        "notes": "Deep search - fewer agents, more steps",
        "hyperparams": {"n_agents": 18, "steps": 7, "decay": 0.7, "initial_pool_size": 25},
        "weights": {
            "movement_weights": {
                "semantic_similarity_unnormalized": 0.50,
                "stark_centrality": 0.25,
                "node_centrality": 0.15,
                "pheromone_repulsion": 0.05,
                "random_jitter": 0.05,
            },
            "deposit_weights": {
                "flat": 0.20,
                "semantic_unnormalized": 0.35,
                "exploration_bonus": 0.10,
                "hub": 0.15,
                "collaborative_amplification": 0.20,
            },
        },
    },
    {
        "name": "balanced_stark_graph",
        "notes": "STaRK-graph optimized - leverage graph structure",
        "hyperparams": {"n_agents": 28, "steps": 5, "decay": 0.5, "initial_pool_size": 40},
        "weights": {
            "movement_weights": {
                "semantic_similarity_unnormalized": 0.30,
                "stark_centrality": 0.35,
                "node_centrality": 0.15,
                "pheromone_repulsion": 0.10,
                "random_jitter": 0.10,
            },
            "deposit_weights": {
                "flat": 0.20,
                "semantic_unnormalized": 0.20,
                "exploration_bonus": 0.20,
                "hub": 0.25,
                "collaborative_amplification": 0.15,
            },
            "ranking_weights": {
                "semantic_rank": 0.80,
                "percentage_visited": 0.20,
            },
        },
    },
    {
        "name": "balanced_hybrid",
        "notes": "Hybrid precision-recall",
        "hyperparams": {"n_agents": 32, "steps": 5, "decay": 0.45, "initial_pool_size": 40},
        "weights": {
            "movement_weights": {
                "semantic_similarity_unnormalized": 0.35,
                "stark_centrality": 0.20,
                "node_centrality": 0.15,
                "pheromone_repulsion": 0.15,
                "random_jitter": 0.15,
            },
            "deposit_weights": {
                "flat": 0.15,
                "semantic_unnormalized": 0.25,
                "exploration_bonus": 0.25,
                "hub": 0.20,
                "collaborative_amplification": 0.15,
            },
            "ranking_weights": {
                "semantic_rank": 0.65,
                "percentage_visited": 0.35,
            },
        },
    },
]


# =============================================================================
# Utility Functions
# =============================================================================

def get_resolved_config(seed_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve a seed config by merging with baseline defaults.

    Args:
        seed_config: Seed configuration with optional overrides

    Returns:
        Fully resolved configuration with all fields populated
    """
    # Start with baseline
    resolved = {
        "name": seed_config["name"],
        **BASELINE_HYPERPARAMS.copy(),
    }

    # Deep copy baseline weights
    resolved_weights = {}
    for key, weights in BASELINE_WEIGHTS.items():
        resolved_weights[key] = weights.copy()

    # Apply hyperparameter overrides
    if "hyperparams" in seed_config:
        resolved.update(seed_config["hyperparams"])

    # Apply weight overrides (deep merge)
    if "weights" in seed_config:
        for key, weights in seed_config["weights"].items():
            if key in resolved_weights:
                resolved_weights[key].update(weights)
            else:
                resolved_weights[key] = weights.copy()

    resolved.update(resolved_weights)
    return resolved


def weights_to_expression_tree(weights: Dict[str, float], strategy_type: str = "movement") -> str:
    """
    Convert weight dictionary to expression tree string.

    Generates nested ADD(MUL(feature, weight), ...) expressions.

    Args:
        weights: Dictionary mapping feature names to weights
        strategy_type: Type of strategy for feature name mapping

    Returns:
        Expression tree string like "ADD(MUL(semantic_similarity_unnormalized, 0.6), MUL(...))"
    """
    # Filter out zero weights
    non_zero = {k: v for k, v in weights.items() if v > 0.001}

    if not non_zero:
        return "flat" if strategy_type == "deposit" else "semantic_similarity_unnormalized"

    if len(non_zero) == 1:
        name, weight = next(iter(non_zero.items()))
        if abs(weight - 1.0) < 0.001:
            return name
        return f"MUL({name}, {weight})"

    # Build nested ADD tree
    items = list(non_zero.items())

    def build_tree(items):
        if len(items) == 1:
            name, weight = items[0]
            return f"MUL({name}, {weight})"
        elif len(items) == 2:
            left = f"MUL({items[0][0]}, {items[0][1]})"
            right = f"MUL({items[1][0]}, {items[1][1]})"
            return f"ADD({left}, {right})"
        else:
            mid = len(items) // 2
            left = build_tree(items[:mid])
            right = build_tree(items[mid:])
            return f"ADD({left}, {right})"

    return build_tree(items)


def config_to_expression_tree_format(seed_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a resolved seed config to expression tree format.

    Args:
        seed_config: Resolved seed configuration

    Returns:
        Configuration with expression tree strings instead of weight dicts
    """
    resolved = get_resolved_config(seed_config)

    return {
        "name": resolved["name"],
        "n_agents": resolved["n_agents"],
        "steps": resolved["steps"],
        "decay": resolved["decay"],
        "initial_pool_size": resolved["initial_pool_size"],
        "movement_tree": weights_to_expression_tree(
            resolved["movement_weights"], "movement"
        ),
        "deposit_tree": weights_to_expression_tree(
            resolved["deposit_weights"], "deposit"
        ),
        "ranking_tree": weights_to_expression_tree(
            resolved["ranking_weights"], "ranking"
        ),
    }


def config_to_weighted_sum_format(seed_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a seed config to weighted_sum format (for SEED_VARIANTS).

    Args:
        seed_config: Seed configuration

    Returns:
        Configuration in SEED_VARIANTS format with 'changes' dict
    """
    changes = {}

    # Add hyperparameter overrides
    if "hyperparams" in seed_config:
        changes.update(seed_config["hyperparams"])

    # Add weight overrides
    if "weights" in seed_config:
        changes.update(seed_config["weights"])

    return {
        "name": seed_config["name"],
        "changes": changes,
    }


def get_all_expression_tree_configs() -> List[Dict[str, Any]]:
    """Get all seed configs in expression tree format."""
    return [config_to_expression_tree_format(cfg) for cfg in SEED_CONFIGS]


def get_all_weighted_sum_configs() -> List[Dict[str, Any]]:
    """Get all seed configs in weighted_sum format."""
    return [config_to_weighted_sum_format(cfg) for cfg in SEED_CONFIGS]


__all__ = [
    "SEED_CONFIGS",
    "BASELINE_HYPERPARAMS",
    "BASELINE_WEIGHTS",
    "get_resolved_config",
    "weights_to_expression_tree",
    "config_to_expression_tree_format",
    "config_to_weighted_sum_format",
    "get_all_expression_tree_configs",
    "get_all_weighted_sum_configs",
]
