from dataclasses import dataclass, field
import os
from typing import List, Dict, Tuple, TypedDict 
from .genome import Genome
try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

class SwarmRAGParamsRange(TypedDict):
    n_agents: Tuple[int, int]
    steps: Tuple[int, int]
    decay: Tuple[float, float]
    initial_pool_size: Tuple[int, int]
    start_subset: Tuple[int, int]
    drop_zone_inc: Tuple[float, float]

class EvolutionConfigDict(TypedDict):
    """
    Configuration for the Evolutionary Engine itself.
    """
    # Resource Management
    concurrent_evaluations: int
    max_workers_per_retrieval: int

    # Loop Control
    n_generations: int
    population_size: int
    elite_fraction: float
    
    # Genetic Probabilities
    base_mutation_rate: float
    crossover_rate: float
    
    # Strategy Names
    creation_strategy: str
    selection_strategy: str
    crossover_strategy: str
    mutation_strategy: str
    fitness_strategy: str
    
    # Strategy-Specific Settings
    phased_switch_gen: NotRequired[int]
    selection_k: NotRequired[int]
    boltzmann_temperature: NotRequired[float]
    boltzmann_alpha: NotRequired[float]
    boltzmann_min_temp: NotRequired[float]
    boltzmann_max_temp: NotRequired[float]
    boltzmann_adaptive: NotRequired[bool]

    mutation_max_expr_size: int
    expr_max_depth: int
    n_agent_groups: int

    # Ranges for Continuous/Integer SwarmParams
    # format: 'param_name': (min, max)
    swarmrag_param_ranges: SwarmRAGParamsRange

    # Validation & Logging
    validation_frequency: int
    log_path: str
    plot_path: str
    plot_title: str
    checkpoint_frequency: int
    checkpoint_path: str
    resume_from_checkpoint: bool

    # LLM Evolution
    use_llm_evolution: NotRequired[bool]
    llm_model: NotRequired[str]
    llm_provider: NotRequired[str]
    llm_concurrency: NotRequired[int]

    # MAP-Elites Configuration
    map_elites_enabled: NotRequired[bool]
    map_elites_dims: NotRequired[List[str]] # Names of descriptors
    map_elites_bins: NotRequired[List[int]] # Number of bins per dimension
    map_elites_ranges: NotRequired[List[Tuple[float, float]]] # Min/Max for each dim
    map_elites_initial_fill: NotRequired[int] # Initial random population to seed archive

DEFAULT_EVO_CONFIG: EvolutionConfigDict = {
    "concurrent_evaluations": 4,
    "max_workers_per_retrieval": 1,
    "n_generations": 20,
    "population_size": 30,
    "elite_fraction": 0.1,
    "base_mutation_rate": 0.2,
    "crossover_rate": 0.6,
    "creation_strategy": "standard_initialization",
    "selection_strategy": "boltzmann",
    "crossover_strategy": "uniform_parameter_mix",
    "mutation_strategy": "expression_tree_mutation",
    "fitness_strategy": "lexicographic",
    "phased_switch_gen": 10,
    "selection_k": 3,
    "boltzmann_temperature": 1.0,
    "boltzmann_alpha": 0.95,
    "boltzmann_min_temp": 0.1,
    "boltzmann_max_temp": 5.0,
    "boltzmann_adaptive": True,
    "boltzmann_diversity_threshold": 0.05,
    "mutation_max_expr_size": 25,
    "expr_max_depth": 5,
    "n_agent_groups": 3,
    "swarmrag_param_ranges": {
        "n_agents": (5, 30),
        "steps": (4, 12),
        "decay": (0.85, 0.99),
        "initial_pool_size": (10, 50),
        "start_subset": (5, 15),
        "drop_zone_inc": (0.05, 0.2)
    },
    "validation_frequency": 5,
    "log_path": "evolution_run/evolution_log.jsonl",
    "plot_path": "evolution_run/evolution_progress.png",
    "plot_title": "Evolutionary Progress",
    "checkpoint_frequency": 5,
    "checkpoint_path": "evolution_run/evo_checkpoint.pkl",
    "resume_from_checkpoint": True,
    
    # LLM Defaults
    "use_llm_evolution": False,
    "llm_provider": "cerebras",
    "llm_model": "zai-glm-4.7",
    "llm_concurrency": 50,

    # MAP-Elites Defaults
    "map_elites_enabled": False,
    "map_elites_dims": ["complexity", "n_agents"],
    "map_elites_bins": [10, 10],
    "map_elites_ranges": [(0, 100), (5, 30)],
    "map_elites_initial_fill": 100
}

@dataclass
class EvolutionContext:
    """
    Shared context passed to all genetic operators (Selection, Crossover, Mutation).
    Replaces the messy **kwargs passing.
    """
    # Current State
    global_mutation_multiplier: float = 1.0
    population: List[Genome] = field(default_factory=list)
    generation: int = 0
    config: EvolutionConfigDict = field(default_factory=lambda: DEFAULT_EVO_CONFIG.copy())
    
    # Registry Data (What features can we mutate into?)
    available_features: List[str] = field(default_factory=list)
    expression_features: Dict[str, List[str]] = field(default_factory=dict)
    
    # State for Adaptive Strategies
    current_temperature: float = 1.0

    

