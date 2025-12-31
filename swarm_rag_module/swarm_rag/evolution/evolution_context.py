from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .genome import Genome

@dataclass
class EvolutionConfig:
    """
    Central type-safe configuration for the Evolution Engine.
    """
    # --- Loop Control ---
    n_generations: int = 20
    population_size: int = 30
    elite_fraction: float = 0.1
    
    # --- Genetic Probabilities ---
    mutation_rate: float = 0.2
    crossover_rate: float = 0.6
    
    # --- Strategy Names (Must match Registry keys) ---
    selection_strategy: str = "tournament"
    crossover_strategy: str = "uniform_parameter_mix"
    mutation_strategy: str = "expression_tree_mutation"
    
    # --- Strategy-Specific Hyperparameters ---
    # We flatten these for transparency (instead of a opaque 'params' dict)
    selection_k: int = 3
    mutation_max_expr_size: int = 25

@dataclass
class EvolutionContext:
    """
    Shared context passed to all genetic operators (Selection, Crossover, Mutation).
    Replaces the messy **kwargs passing.
    """
    # Current State
    population: List[Genome]
    generation: int
    config: EvolutionConfig
    # Registry Data (What features can we mutate into?)
    available_features: List[str] = None