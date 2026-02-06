"""
Mutation operators for the evolutionary algorithm.

Contains expression tree mutation, aggressive mutation, guided mutation,
focused mutation, and supporting helper functions for parameter/ratio mutation.
"""
import logging
import math
import random
from typing import Dict, List

from ...types.config import EvolutionContext, SwarmParamRanges
from ...types.expressions import ExpressionEvolution, ExpressionNode
from ....interfaces.enums import GeneticKey
from ...types.genome import Genome, DEFAULT_PARAMS, SwarmParams, FIXED_PARAMS
from .focused_mutation import apply_focused_mutation

from .registry import GeneticRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mutate_params_standard(genome: Genome, ctx: EvolutionContext, rate: float):
    """Standard parameter mutation (Smart Jitter).

    Skips fixed parameters (FIXED_PARAMS) and uses tightened ranges
    from SwarmParamRanges (single source of truth) for evolvable parameters.
    """
    evolvable_ranges = SwarmParamRanges().to_evolvable_dict()
    for key, val in genome.params.items():
        # Skip fixed parameters - they should never be mutated
        if key in FIXED_PARAMS:
            # Ensure fixed params have correct values
            genome.params[key] = FIXED_PARAMS[key]
            continue

        if random.random() < rate:
            # 80% chance: Fine-tuning (Small Gaussian jitter)
            if random.random() < 0.8:
                if isinstance(val, int):
                    delta = int(round(random.gauss(0, 1.5)))  # +/- 1 or 2 usually
                    new_val = max(1, val + delta)
                    # Clamp to evolvable range if defined
                    if key in evolvable_ranges:
                        min_v, max_v = evolvable_ranges[key]
                        new_val = max(min_v, min(max_v, new_val))
                    genome.params[key] = new_val
                elif isinstance(val, float):
                    # +/- 10% relative change
                    factor = random.gauss(1.0, 0.1)
                    new_val = val * factor
                    # Clamp to evolvable range if defined
                    if key in evolvable_ranges:
                        min_v, max_v = evolvable_ranges[key]
                        new_val = max(min_v, min(max_v, new_val))
                    else:
                        new_val = max(0.001, min(0.999, new_val))
                    genome.params[key] = new_val

            # 20% chance: Exploration (Re-sample from range)
            else:
                # Prefer SwarmParamRanges, fallback to config ranges
                if key in evolvable_ranges:
                    min_v, max_v = evolvable_ranges[key]
                    if isinstance(min_v, int):
                        genome.params[key] = random.randint(min_v, max_v)
                    else:
                        genome.params[key] = random.uniform(min_v, max_v)
                else:
                    ranges = ctx.config.genetic.param_ranges
                    if hasattr(ranges, key):
                        min_v, max_v = getattr(ranges, key)
                        if isinstance(min_v, int):
                            genome.params[key] = random.randint(min_v, max_v)
                        else:
                            genome.params[key] = random.uniform(min_v, max_v)


def _mutate_ratios_standard(genome: Genome, rate: float):
    """Standard group ratio mutation (Smart Jitter)."""
    for key, val in genome.group_ratios.items():
        if random.random() < rate:
            # Jitter ratio
            genome.group_ratios[key] = max(0.05, min(1.0, val * random.uniform(0.8, 1.2)))


def _randomize_all_params(ctx: EvolutionContext) -> SwarmParams:
    """Helper to fully randomize evolvable parameters within tightened ranges.

    Uses FIXED_PARAMS for fixed parameters and SwarmParamRanges (single source
    of truth) for evolvable parameters. Falls back to ctx.config.genetic.param_ranges
    for parameters not defined in either.
    """
    # Start with default params
    params = DEFAULT_PARAMS.copy()

    # Apply fixed parameters (never randomized)
    params.update(FIXED_PARAMS)

    # Randomize evolvable parameters from SwarmParamRanges (single source of truth)
    evolvable_ranges = SwarmParamRanges().to_evolvable_dict()
    for key, (min_v, max_v) in evolvable_ranges.items():
        if isinstance(min_v, int):
            params[key] = random.randint(min_v, max_v)
        else:
            params[key] = random.uniform(min_v, max_v)

    # Fallback for any params not in evolvable_ranges or FIXED_PARAMS
    ranges = ctx.config.genetic.param_ranges
    for key in params.keys():
        if key not in FIXED_PARAMS and key not in evolvable_ranges:
            if hasattr(ranges, key):
                min_v, max_v = getattr(ranges, key)
                if isinstance(min_v, int):
                    params[key] = random.randint(int(min_v), int(max_v))
                else:
                    params[key] = random.uniform(min_v, max_v)

    return params


def _randomize_ratios(ctx: EvolutionContext, n_groups: int) -> Dict[str, float]:
    """Helper to fully randomize group ratios."""
    # Default range for group ratios (not in SwarmParamRanges)
    min_r, max_r = 0.1, 1.0
    return {f"g{i}": random.uniform(min_r, max_r) for i in range(n_groups)}


def _resolve_feature_list(key: str, ctx: EvolutionContext) -> List[str]:
    """
    Resolves the appropriate feature list for a strategy key.

    Handles patterns like 'ranking', 'gN_movement', 'gN_deposit'.

    Args:
        key: Strategy key (e.g., "g0_movement", "ranking")
        ctx: Evolution context containing expression_features

    Returns:
        List of valid feature names for this strategy type
    """
    # Direct match first
    feature_list = ctx.expression_features.get(key)
    if feature_list:
        return feature_list

    # Pattern match for group strategies
    if key.endswith("_movement") or "movement" in key:
        return ctx.expression_features.get("movement", [])
    elif key.endswith("_deposit") or "deposit" in key:
        return ctx.expression_features.get("deposit", [])
    elif key == "ranking" or "ranking" in key:
        return ctx.expression_features.get("ranking", [])

    # Fallback: return empty list
    return []


# ---------------------------------------------------------------------------
# Registered mutation operators
# ---------------------------------------------------------------------------

@GeneticRegistry.register_mutation(GeneticKey.EXPRESSION_TREE_MUTATION)
def expression_tree_mutation(genome: Genome, ctx: EvolutionContext) -> Genome:
    tau = 0.2
    # Optimized: Use python math/random for scalar operations instead of torch to avoid kernel launch overhead
    log_mutation_factor = tau * random.gauss(0, 1)
    genome.mutation_rate *= math.exp(log_mutation_factor)
    genome.mutation_rate = max(0.01, min(0.5, genome.mutation_rate))
    rate = genome.mutation_rate * ctx.global_mutation_multiplier

    # 1. Parameter Mutation (Smart Jitter)
    _mutate_params_standard(genome, ctx, rate)

    # 2. Group Ratio Mutation
    _mutate_ratios_standard(genome, rate)

    # 3. Strategy Tree Mutation
    for key, tree in genome.strategies.items():
        if random.random() < rate:
            feature_list = _resolve_feature_list(key, ctx)

            # Structural Mutations
            mut_choice = random.random()

            if mut_choice < 0.1 and tree.type == 'op':
                # Hoist: Replace current node with one of its children (Simplification)
                if tree.children:
                    genome.strategies[key] = random.choice(tree.children).copy()

            elif mut_choice < 0.2:
                # Wrapper: Wrap current tree in a unary function (Complexity)
                func = random.choice(['log', 'sigmoid', 'tanh'])
                new_root = ExpressionNode(type='func', value=func, children=[tree.copy()])
                genome.strategies[key] = new_root

            else:
                # Standard Node/Subtree Mutation
                mutated_tree = ExpressionEvolution.mutate_tree(
                    tree,
                    features=feature_list,
                    mutation_rate=rate,
                    inplace=True
                )
                genome.strategies[key] = mutated_tree

            # Occasional Simplification/Pruning to prevent bloat
            if random.random() < 0.1:
                genome.strategies[key] = ExpressionEvolution.simplify_tree(genome.strategies[key], max_size=30)

    genome.clear_cache()
    return genome


@GeneticRegistry.register_mutation(GeneticKey.AGGRESSIVE_MUTATION)
def aggressive_mutation(genome: Genome, ctx: EvolutionContext) -> Genome:
    """
    High-impact mutation strategy.
    - Higher base rate.
    - Parameters are often re-sampled from global ranges instead of jittered.
    - Tree mutations prefer 'subtree' replacement.
    """
    # Boost rate
    genome.mutation_rate = 0.4  # Lock to high rate
    rate = genome.mutation_rate * ctx.global_mutation_multiplier
    ranges = ctx.config.genetic.param_ranges

    # Aggressive Parameter Resampling
    for key in genome.params.keys():
        if random.random() < rate:
            if hasattr(ranges, key):
                # 50% chance to purely resample from global range (Big Jump)
                if random.random() < 0.5:
                    min_v, max_v = getattr(ranges, key)
                    if isinstance(min_v, int):
                        genome.params[key] = random.randint(int(min_v), int(max_v))
                    else:
                        genome.params[key] = random.uniform(min_v, max_v)
                else:
                    # 50% chance of large jitter (+/- 30%), clamped to bounds
                    val = genome.params[key]
                    min_v, max_v = getattr(ranges, key)
                    if isinstance(val, int):
                        new_val = val + random.randint(-5, 5)
                        genome.params[key] = max(int(min_v), min(int(max_v), new_val))
                    else:
                        new_val = val * random.uniform(0.7, 1.3)
                        genome.params[key] = max(min_v, min(max_v, new_val))

    # Aggressive Group Ratio Mutation
    for key, val in genome.group_ratios.items():
        if random.random() < rate:
            # Larger Jitter for aggressive
            genome.group_ratios[key] = max(0.05, min(1.0, val * random.uniform(0.6, 1.4)))

    # Aggressive Tree Mutation
    for key, tree in genome.strategies.items():
        if random.random() < rate:
            feature_list = _resolve_feature_list(key, ctx)

            # Force a subtree replacement (structural change)
            # rather than just changing a node value
            # We do this by manually generating a new random subtree and swapping
            if random.random() < 0.7:
                genome.strategies[key] = ExpressionEvolution.random_tree(feature_list, max_depth=3)
            else:
                # Fallback to standard mutation
                genome.strategies[key] = ExpressionEvolution.mutate_tree(
                    tree, features=feature_list, mutation_rate=1.0, inplace=True
                )

    genome.clear_cache()
    return genome


@GeneticRegistry.register_mutation(GeneticKey.GUIDED_MUTATION)
def guided_mutation(genome: Genome, ctx: EvolutionContext) -> Genome:
    """
    Smart mutation that encourages known good patterns:
    - Ensures semantic similarity features are present in movement.
    - Ensures 'pheromone_repulsion' (diversity) is occasionally injected.
    - Protects critical features from being lost during mutation.

    Critical features to protect:
    - semantic_similarity / semantic_similarity_unnormalized
    - stark_centrality (when present)
    - pheromone_repulsion
    """
    tau = 0.2
    # Optimized: Use python math/random for scalar operations instead of torch
    genome.mutation_rate *= math.exp(tau * random.gauss(0, 1))
    genome.mutation_rate = max(0.01, min(0.5, genome.mutation_rate))
    rate = genome.mutation_rate * ctx.global_mutation_multiplier

    # Critical feature names (protect these during mutation)
    SEMANTIC_FEATURES = {'semantic_similarity', 'semantic_similarity_unnormalized'}
    CENTRALITY_FEATURES = {'stark_centrality', 'node_centrality'}
    DIVERSITY_FEATURES = {'pheromone_repulsion'}

    # Standard Parameter Jitter (Same as expression_tree_mutation)
    _mutate_params_standard(genome, ctx, rate)

    # Group Ratio Mutation (Standard Jitter)
    _mutate_ratios_standard(genome, rate)

    # Guided Tree Mutation
    for key, tree in genome.strategies.items():
        if random.random() < rate:
            feature_list = _resolve_feature_list(key, ctx)

            # Check for critical features
            all_nodes = ExpressionEvolution.get_all_nodes(tree)
            node_values = {n.value for n in all_nodes}

            has_semantic = bool(node_values & SEMANTIC_FEATURES)
            has_centrality = bool(node_values & CENTRALITY_FEATURES)
            has_pheromone = bool(node_values & DIVERSITY_FEATURES)

            # A) Injection Logic (If missing critical features, inject them)
            injected = False
            if "movement" in key:
                # If missing semantic signal, 50% chance to inject
                if not has_semantic and random.random() < 0.5:
                    # Prefer unnormalized for baseline compatibility
                    semantic_feature = "semantic_similarity_unnormalized"
                    if semantic_feature not in feature_list:
                        semantic_feature = "semantic_similarity"

                    # Wrap: (Current + semantic * 0.5) / 1.5
                    new_node = ExpressionNode("op", "+", [
                        tree.copy(),
                        ExpressionNode("op", "*", [
                            ExpressionNode("feature", semantic_feature),
                            ExpressionNode("const", 0.5)
                        ])
                    ])
                    genome.strategies[key] = ExpressionNode("op", "/", [
                         new_node,
                         ExpressionNode("const", 1.5)
                    ])
                    injected = True

                # If missing diversity, 30% chance to inject
                elif not has_pheromone and random.random() < 0.3:
                    # Wrap: Current * (0.7 + 0.3 * pheromone_repulsion)
                    pheromone_term = ExpressionNode("op", "+", [
                        ExpressionNode("const", 0.7),
                        ExpressionNode("op", "*", [
                            ExpressionNode("const", 0.3),
                            ExpressionNode("feature", "pheromone_repulsion")
                        ])
                    ])
                    genome.strategies[key] = ExpressionNode("op", "*", [
                        tree.copy(),
                        pheromone_term
                    ])
                    injected = True

                # If has centrality but missing semantic, 40% chance to add both
                elif has_centrality and not has_semantic and random.random() < 0.4:
                    semantic_feature = "semantic_similarity_unnormalized"
                    if semantic_feature not in feature_list:
                        semantic_feature = "semantic_similarity"

                    # Add semantic weighted sum
                    genome.strategies[key] = ExpressionNode("op", "+", [
                        ExpressionNode("op", "*", [
                            tree.copy(),
                            ExpressionNode("const", 0.4)
                        ]),
                        ExpressionNode("op", "*", [
                            ExpressionNode("feature", semantic_feature),
                            ExpressionNode("const", 0.6)
                        ])
                    ])
                    injected = True

            # B) Standard Mutation (if not injected)
            if not injected:
                # Save original tree for potential revert
                original_tree = tree.copy()

                mutated_tree = ExpressionEvolution.mutate_tree(
                    tree, features=feature_list, mutation_rate=rate, inplace=True
                )

                # Verification: Check if we lost critical features
                new_nodes = ExpressionEvolution.get_all_nodes(mutated_tree)
                new_values = {n.value for n in new_nodes}

                # Revert if we lost semantic features (very important!)
                if has_semantic and not (new_values & SEMANTIC_FEATURES):
                    if random.random() < 0.85:  # 85% chance to revert
                        genome.strategies[key] = original_tree
                        continue

                # Revert if we lost centrality (moderately important)
                if has_centrality and not (new_values & CENTRALITY_FEATURES):
                    if random.random() < 0.6:  # 60% chance to revert
                        genome.strategies[key] = original_tree
                        continue

                genome.strategies[key] = mutated_tree

    genome.clear_cache()
    return genome


@GeneticRegistry.register_mutation(GeneticKey.FOCUSED_MUTATION)
def focused_mutation(genome: Genome, ctx: EvolutionContext) -> Genome:
    """
    Metric-aware mutation that targets the weakest metric.

    Analyzes the genome's fitness profile to identify which metric
    (recall, MRR, precision) is weakest, then focuses mutations on
    parameters most likely to improve that metric.

    This provides more directed evolution than random parameter
    mutation, potentially accelerating convergence.
    """
    # Get fitness if available
    fitness = genome.fitness if hasattr(genome, 'fitness') and genome.fitness else None

    # Apply focused mutation using the dedicated module
    genome = apply_focused_mutation(
        genome=genome,
        ctx=ctx,
        fitness=fitness,
        mutation_rate=genome.mutation_rate * ctx.global_mutation_multiplier,
    )

    # Also apply some tree mutation for exploration
    rate = genome.mutation_rate * ctx.global_mutation_multiplier * 0.5  # Reduced rate

    for key, tree in genome.strategies.items():
        if random.random() < rate:
            feature_list = _resolve_feature_list(key, ctx)
            mutated_tree = ExpressionEvolution.mutate_tree(
                tree, features=feature_list, mutation_rate=rate, inplace=True
            )
            genome.strategies[key] = mutated_tree

    genome.clear_cache()
    return genome
