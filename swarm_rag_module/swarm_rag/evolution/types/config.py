from dataclasses import dataclass, field
from typing import List, Dict 
from .genome import Genome

@dataclass
class EvolutionConfig:
    """
    Central type-safe configuration for the Evolution Engine.
    """
    # --- Resource Management ---
    global_max_threads: int = 16
    concurrent_evaluations: int = 4

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
    # --- Strategy Hyperparameters ---
    selection_k: int = 3
    mutation_max_expr_size: int = 25

    # --- Genome Hyperparameter RANGES (FLAT, TYPED) ---
    n_agents_min: int = 5
    n_agents_max: int = 30

    steps_min: int = 5
    steps_max: int = 20

    decay_min: float = 0.85
    decay_max: float = 0.99

    initial_pool_size_min: int = 10
    initial_pool_size_max: int = 50

    start_subset_min: int = 5
    start_subset_max: int = 15

    # --- Expression Initialization ---
    expr_max_depth: int = 5
    
    # --- Validation & Tracking ---
    validation_frequency: int = 5   # Run validation every N generations
    log_file: str = "evo_log.jsonl"
    plot_file: str = "evo_plot.png"

@dataclass
class EvolutionContext:
    """
    Shared context passed to all genetic operators (Selection, Crossover, Mutation).
    Replaces the messy **kwargs passing.
    """
    # Current State
    population: List[Genome] = field(default_factory=list)
    generation: int
    config: EvolutionConfig = None
    # Registry Data (What features can we mutate into?)
    available_features: List[str] = field(default_factory=list)
    expression_features: Dict[str, List[str]] = field(default_factory=dict)
