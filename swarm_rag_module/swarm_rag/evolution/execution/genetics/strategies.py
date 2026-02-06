"""
Genetic operators for the evolutionary algorithm.

Strategies are split into category files:
- mutations.py: Mutation operators
- crossovers.py: Crossover operators
- selections.py: Selection operators
- initialization.py: Population initialization strategies

This module provides GeneticRegistry (operator registry) and
GeneticStrategies (backward-compatible facade).

GeneticRegistry lives in genetic_registry.py to avoid circular imports;
it is re-exported here so that existing ``from .strategies import
GeneticRegistry`` continues to work.
"""

# Re-export GeneticRegistry (canonical home: genetic_registry.py)
from .registry import GeneticRegistry  # noqa: F401

# Import category modules to trigger @GeneticRegistry.register_* decorators
from . import mutations, crossovers, selections, initialization  # noqa: F401


class GeneticStrategies:
    """Backward-compatible facade for genetic operators.

    Every public method delegates to the corresponding module-level
    function so that ``GeneticStrategies.expression_tree_mutation(...)``
    keeps working for existing call-sites.
    """

    # --- Mutations ---
    expression_tree_mutation = staticmethod(mutations.expression_tree_mutation)
    aggressive_mutation = staticmethod(mutations.aggressive_mutation)
    guided_mutation = staticmethod(mutations.guided_mutation)
    focused_mutation = staticmethod(mutations.focused_mutation)

    # --- Crossovers ---
    uniform_parameter_mix = staticmethod(crossovers.uniform_parameter_mix)
    subtree_crossover = staticmethod(crossovers.subtree_crossover)
    root_mix_crossover = staticmethod(crossovers.root_mix_crossover)

    # --- Selections ---
    tournament_selection = staticmethod(selections.tournament_selection)
    boltzmann_selection = staticmethod(selections.boltzmann_selection)

    # --- Initializations ---
    standard_initialization = staticmethod(initialization.standard_initialization)
    shallow_growth_initialization = staticmethod(initialization.shallow_growth_initialization)
    seeded_initialization = staticmethod(initialization.seeded_initialization)
    baseline_seeded_initialization = staticmethod(initialization.baseline_seeded_initialization)

    # --- Helpers (preserved for internal/test use) ---
    _mix_params = staticmethod(crossovers._mix_params)
    _crossover_preamble = staticmethod(crossovers._crossover_preamble)
    _pick_parent_tree = staticmethod(crossovers._pick_parent_tree)
    _mutate_params_standard = staticmethod(mutations._mutate_params_standard)
    _mutate_ratios_standard = staticmethod(mutations._mutate_ratios_standard)
    _randomize_all_params = staticmethod(mutations._randomize_all_params)
    _randomize_ratios = staticmethod(mutations._randomize_ratios)
    _resolve_feature_list = staticmethod(mutations._resolve_feature_list)
