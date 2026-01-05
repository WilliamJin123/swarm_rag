from dataclasses import dataclass, field
import os
from typing import List, Dict, Tuple, TypedDict 
from .genome import Genome


class EvolutionConfigDict(TypedDict):
    """
    Configuration for the Evolutionary Engine itself.
    """
    # Resource Management
    concurrent_evaluations: int
    global_max_threads: int

    # Loop Control
    n_generations: int
    population_size: int
    elite_fraction: float
    
    # Genetic Probabilities
    mutation_rate: float
    crossover_rate: float
    
    # Strategy Names
    selection_strategy: str
    crossover_strategy: str
    mutation_strategy: str
    
    # Strategy-Specific Settings
    selection_k: int
    mutation_max_expr_size: int
    expr_max_depth: int
    n_agent_groups: int

    # Ranges for Continuous/Integer SwarmParams
    # format: 'param_name': (min, max)
    param_ranges: Dict[str, Tuple[float, float]]

    # Validation & Logging
    validation_frequency: int
    log_path: str
    plot_path: str
    plot_title: str
    checkpoint_frequency: int
    checkpoint_path: str
    resume_from_checkpoint: bool

DEFAULT_EVO_CONFIG: EvolutionConfigDict = {
    "concurrent_evaluations": 4,
    "global_max_threads": 8,
    "n_generations": 20,
    "population_size": 30,
    "elite_fraction": 0.1,
    "mutation_rate": 0.2,
    "crossover_rate": 0.6,
    "selection_strategy": "tournament",
    "crossover_strategy": "uniform_parameter_mix",
    "mutation_strategy": "expression_tree_mutation",
    "selection_k": 3,
    "mutation_max_expr_size": 25,
    "expr_max_depth": 5,
    "n_agent_groups": 2,
    "param_ranges": {
        "n_agents": (5, 30),
        "steps": (4, 12),
        "decay": (0.85, 0.99),
        "initial_pool_size": (10, 50),
        "start_subset": (5, 15),
    },
    "validation_frequency": 5,
    "log_path": "evolution_log.jsonl",
    "plot_path": "evolution_progress.png",
    "plot_title": "Evolutionary Progress",
    "checkpoint_frequency": 5,
    "checkpoint_path": "evo_checkpoint.pkl",
    "resume_from_checkpoint": True
}

@dataclass
class EvolutionContext:
    """
    Shared context passed to all genetic operators (Selection, Crossover, Mutation).
    Replaces the messy **kwargs passing.
    """
    # Current State
    current_mutation_rate: float = None
    population: List[Genome] = field(default_factory=list)
    generation: int = 0
    config: EvolutionConfigDict = field(default_factory=lambda: DEFAULT_EVO_CONFIG.copy())
    
    # Registry Data (What features can we mutate into?)
    available_features: List[str] = field(default_factory=list)
    expression_features: Dict[str, List[str]] = field(default_factory=dict)

    

