"""
Crossover operators for the evolutionary algorithm.

Contains uniform parameter mix, subtree crossover, root mix crossover,
and supporting helper functions.
"""
import logging
import random

from ...types.config import EvolutionContext, SwarmParamRanges
from ...types.expressions import ExpressionEvolution, ExpressionNode
from ....interfaces.enums import GeneticKey
from ...types.genome import Genome

from .registry import GeneticRegistry

logger = logging.getLogger(__name__)

# Crossover constants
CROSSOVER_BIAS = 0.7        # Probability of attempting subtree/root mixing vs inheriting whole tree
PARENT_SELECTION_PROB = 0.5  # Probability threshold for selecting between parent1 and parent2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mix_params(child: Genome, parent2: Genome):
    """Helper to mix scalar parameters and group ratios uniformly."""
    ranges = SwarmParamRanges()
    for key in child.params:
        if key in parent2.params and random.random() > PARENT_SELECTION_PROB:
            child.params[key] = parent2.params[key]
        # Clamp to bounds after mixing
        if hasattr(ranges, key):
            min_v, max_v = getattr(ranges, key)
            val = child.params[key]
            if isinstance(min_v, int):
                child.params[key] = max(int(min_v), min(int(max_v), int(val)))
            else:
                child.params[key] = max(min_v, min(max_v, val))

    for key in child.group_ratios:
        if key in parent2.group_ratios and random.random() > PARENT_SELECTION_PROB:
            child.group_ratios[key] = parent2.group_ratios[key]


def _crossover_preamble(parent1: Genome, parent2: Genome) -> Genome:
    """Common setup for crossover operators: copy, average mutation_rate, mix params."""
    child = parent1.copy()
    child.mutation_rate = (parent1.mutation_rate + parent2.mutation_rate) / 2.0
    _mix_params(child, parent2)
    return child


def _pick_parent_tree(p1_tree, p2_tree):
    """Randomly pick one parent's tree (used when crossover is skipped)."""
    if random.random() > PARENT_SELECTION_PROB:
        return p2_tree.copy()
    return p1_tree.copy()


# ---------------------------------------------------------------------------
# Registered crossover operators
# ---------------------------------------------------------------------------

@GeneticRegistry.register_crossover(GeneticKey.UNIFORM_PARAMETER_MIX)
def uniform_parameter_mix(parent1: Genome, parent2: Genome, ctx: EvolutionContext) -> Genome:
    """Mixes traits 50/50. """
    child = parent1.copy()
    child.mutation_rate = (parent1.mutation_rate + parent2.mutation_rate) / 2.0

    _mix_params(child, parent2)

    for key in child.strategies:
        if random.random() > PARENT_SELECTION_PROB:
            child.strategies[key] = parent2.strategies[key].copy()

    child.clear_cache()
    return child


@GeneticRegistry.register_crossover(GeneticKey.SUBTREE_CROSSOVER)
def subtree_crossover(parent1: Genome, parent2: Genome, ctx: EvolutionContext) -> Genome:
    """
    GP-style Subtree Crossover.
    1. Mixes scalar parameters uniformly.
    2. For strategy trees, attempts to swap random subtrees between parents.
    """
    child = _crossover_preamble(parent1, parent2)

    for key in child.strategies:
        p1_tree = parent1.strategies[key]
        p2_tree = parent2.strategies[key]

        # Chance to perform subtree swap vs just inheriting whole tree
        if random.random() < CROSSOVER_BIAS:  # 70% chance to try mixing
            try:
                # We need deep copies to avoid modifying parents
                new_tree = p1_tree.copy()
                donor_tree = p2_tree.copy()

                # Get all nodes (flatten)
                p1_nodes = ExpressionEvolution.get_all_nodes(new_tree)
                p2_nodes = ExpressionEvolution.get_all_nodes(donor_tree)

                if p1_nodes and p2_nodes:
                    # Pick crossover points
                    target_node = random.choice(p1_nodes)
                    source_node = random.choice(p2_nodes)

                    # Swap content (type, value, children)
                    # We do this by modifying target_node in-place to become source_node
                    target_node.type = source_node.type
                    target_node.value = source_node.value
                    target_node.children = [c.copy() for c in source_node.children]

                    child.strategies[key] = new_tree
                else:
                    # Fallback
                    child.strategies[key] = p1_tree.copy()
            except Exception:
                # Safety fallback - log the failure for diagnostics
                logger.debug("Subtree crossover failed for strategy '%s', falling back to parent1 tree", key, exc_info=True)
                child.strategies[key] = p1_tree.copy()
        else:
            child.strategies[key] = _pick_parent_tree(p1_tree, p2_tree)

    child.clear_cache()
    return child


@GeneticRegistry.register_crossover(GeneticKey.ROOT_MIX_CROSSOVER)
def root_mix_crossover(parent1: Genome, parent2: Genome, ctx: EvolutionContext) -> Genome:
    """
    Swaps top-level branches of strategy trees.
    This is less destructive than random subtree crossover as it preserves
    the high-level logic (the operator) if both share it, or swaps whole approaches.
    """
    child = _crossover_preamble(parent1, parent2)

    for key in child.strategies:
        p1_tree = parent1.strategies[key]
        p2_tree = parent2.strategies[key]

        # 70% chance to mix, 30% chance to clone one parent
        if random.random() < CROSSOVER_BIAS:
            # If both are operators with children (e.g. A * B, C + D)
            if p1_tree.type == 'op' and p2_tree.type == 'op' and p1_tree.children and p2_tree.children:
                 # Create new root using Parent 1's operator
                 new_tree = ExpressionNode(type='op', value=p1_tree.value, children=[])

                 # Take one child from P1 and one from P2
                 # (Assumes binary operators for simplicity, or takes first child)
                 c1 = p1_tree.children[0].copy()
                 # If P2 has children, take one, otherwise take P2 itself
                 c2 = p2_tree.children[-1].copy() if len(p2_tree.children) > 1 else p2_tree.children[0].copy()

                 # Randomly swap order
                 if random.random() > PARENT_SELECTION_PROB:
                     new_tree.children = [c1, c2]
                 else:
                     new_tree.children = [c2, c1]

                 child.strategies[key] = new_tree
            else:
                # If structures don't match well, just swap the whole tree
                child.strategies[key] = p2_tree.copy()
        else:
            child.strategies[key] = _pick_parent_tree(p1_tree, p2_tree)

    child.clear_cache()
    return child
