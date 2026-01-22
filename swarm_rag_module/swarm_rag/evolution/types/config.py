"""
Evolution Configuration System

Organized nested dataclasses replacing the flat 50+ field TypedDict.
MAP-Elites is the default and only evolution paradigm.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, TYPE_CHECKING
import os

if TYPE_CHECKING:
    from .genome import Genome


def _get_optimal_concurrency() -> int:
    """
    Calculate optimal concurrency based on available CPU cores.

    Uses half the CPU count (to leave room for other processes),
    capped at 16 to avoid memory issues.
    """
    try:
        cpu_count = os.cpu_count() or 4
        return min(cpu_count // 2, 16)
    except Exception:
        return 4


# =============================================================================
# Config Dataclasses (Organized by Concern)
# =============================================================================

@dataclass
class ResourceConfig:
    """
    Concurrency and worker settings.

    Defaults are dynamically scaled based on available CPU cores.
    """
    concurrent_evaluations: int = field(default_factory=_get_optimal_concurrency)
    max_workers_per_retrieval: int = 4

    # Dynamic batch sizing based on archive state
    enable_dynamic_batch_size: bool = True
    base_batch_size: int = 30
    min_batch_size: int = 15
    max_batch_size: int = 50


@dataclass
class MapElitesConfig:
    """MAP-Elites archive and breeding settings."""
    dimensions: List[str] = field(default_factory=lambda: ["aggressiveness", "complexity"])
    bins: List[int] = field(default_factory=lambda: [15, 12])
    ranges: List[Tuple[float, float]] = field(default_factory=lambda: [(10.0, 150.0), (5.0, 60.0)])
    initial_fill: int = 100
    batch_size: int = 30  # offspring per generation


@dataclass
class SwarmParamRanges:
    """Valid ranges for SwarmRAG parameters."""
    n_agents: Tuple[int, int] = (5, 30)
    steps: Tuple[int, int] = (4, 12)
    decay: Tuple[float, float] = (0.85, 0.99)
    initial_pool_size: Tuple[int, int] = (10, 50)
    start_subset: Tuple[int, int] = (5, 15)
    drop_zone_inc: Tuple[float, float] = (0.05, 0.2)

    def to_dict(self) -> Dict[str, Tuple]:
        """Convert to dict format for backwards compatibility."""
        return {
            "n_agents": self.n_agents,
            "steps": self.steps,
            "decay": self.decay,
            "initial_pool_size": self.initial_pool_size,
            "start_subset": self.start_subset,
            "drop_zone_inc": self.drop_zone_inc,
        }


@dataclass
class BoltzmannConfig:
    """Boltzmann selection parameters."""
    temperature: float = 1.0
    alpha: float = 0.95  # cooling factor
    min_temp: float = 0.1
    max_temp: float = 5.0
    adaptive: bool = True
    diversity_threshold: float = 0.05


@dataclass
class GeneticConfig:
    """Genetic operator settings."""
    # Strategy names (registered in GeneticRegistry)
    creation_strategy: str = "standard_initialization"
    selection_strategy: str = "boltzmann"
    crossover_strategy: str = "uniform_parameter_mix"
    mutation_strategy: str = "guided_mutation"

    # Rates
    base_mutation_rate: float = 0.25
    crossover_rate: float = 0.6

    # Expression tree limits
    expr_max_depth: int = 5
    mutation_max_expr_size: int = 25
    n_agent_groups: int = 3

    # Selection settings
    selection_k: int = 3  # tournament size

    # Boltzmann selection
    boltzmann: BoltzmannConfig = field(default_factory=BoltzmannConfig)

    # Parameter ranges
    param_ranges: SwarmParamRanges = field(default_factory=SwarmParamRanges)


@dataclass
class LLMConfig:
    """LLM provider settings for LLM-guided mutations."""
    enabled: bool = False
    provider: str = "cerebras"
    model: str = "zai-glm-4.7"
    env_path: str = ".env"


@dataclass
class CreativeModeConfig:
    """
    Configuration for LLM creative mode (custom expression generation).

    Creative mode allows the LLM to generate custom heuristic expressions
    beyond predefined templates, while maintaining safety through validation.

    Trigger conditions (any of these can activate creative mode):
    - Stagnation: No improvement for `trigger_stagnation` generations
    - Fill rate: Archive fill rate below `trigger_fill_rate`
    - Top fitness unchanged: Best fitness hasn't improved for 3+ generations
    - Periodic: Every `periodic_interval` generations
    """
    # Global enable flag
    enabled: bool = False

    # Trigger thresholds
    trigger_stagnation: int = 5          # Generations without improvement
    trigger_fill_rate: float = 0.3       # Archive fill rate threshold
    periodic_interval: int = 10          # Periodic experimentation interval

    # Limits
    max_creative_per_generation: int = 3  # Max creative mutations per generation
    complexity_limit: int = 30            # Max expression nodes

    # Behavior
    fallback_on_failure: bool = True     # Use template if creative fails
    track_performance: bool = True       # Compare creative vs template

    # Circuit breaker (auto-disable after failures)
    max_consecutive_failures: int = 5    # Disable after N consecutive failures


@dataclass
class CheckpointConfig:
    """Logging and checkpointing settings."""
    log_path: str = "evolution_run/evolution_log.jsonl"
    plot_path: str = "evolution_run/evolution_progress.png"
    plot_title: str = "MAP-Elites Evolution"
    checkpoint_path: str = "evolution_run/evo_checkpoint.pkl"
    checkpoint_frequency: int = 5
    validation_frequency: int = 5


@dataclass
class EvolutionConfig:
    """
    Top-level evolution configuration.

    Replaces the flat 50+ field EvolutionConfigDict with organized nested configs.
    MAP-Elites is the default and only evolution paradigm.

    Example:
        config = EvolutionConfig(
            n_generations=100,
            map_elites=MapElitesConfig(bins=[20, 15]),
            llm=LLMConfig(enabled=True)
        )
    """
    # Core loop settings
    n_generations: int = 50
    fitness_strategy: str = "lexicographic"  # lexicographic, pareto, phased
    phased_switch_gen: int = 25  # Only used if fitness_strategy == "phased"

    # Nested configs
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    map_elites: MapElitesConfig = field(default_factory=MapElitesConfig)
    genetic: GeneticConfig = field(default_factory=GeneticConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    creative_mode: CreativeModeConfig = field(default_factory=CreativeModeConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert to flat dict format for backwards compatibility.

        This allows gradual migration of code that uses config["key"] access.
        Can be removed once all callsites are updated.
        """
        return {
            # Resource Management
            "concurrent_evaluations": self.resources.concurrent_evaluations,
            "max_workers_per_retrieval": self.resources.max_workers_per_retrieval,

            # Loop Control
            "n_generations": self.n_generations,
            "population_size": self.map_elites.batch_size,  # mapped for compatibility

            # Strategy Names
            "creation_strategy": self.genetic.creation_strategy,
            "selection_strategy": self.genetic.selection_strategy,
            "crossover_strategy": self.genetic.crossover_strategy,
            "mutation_strategy": self.genetic.mutation_strategy,
            "fitness_strategy": self.fitness_strategy,

            # Genetic Settings
            "base_mutation_rate": self.genetic.base_mutation_rate,
            "crossover_rate": self.genetic.crossover_rate,
            "expr_max_depth": self.genetic.expr_max_depth,
            "mutation_max_expr_size": self.genetic.mutation_max_expr_size,
            "n_agent_groups": self.genetic.n_agent_groups,
            "selection_k": self.genetic.selection_k,
            "phased_switch_gen": self.phased_switch_gen,

            # Boltzmann Settings
            "boltzmann_temperature": self.genetic.boltzmann.temperature,
            "boltzmann_alpha": self.genetic.boltzmann.alpha,
            "boltzmann_min_temp": self.genetic.boltzmann.min_temp,
            "boltzmann_max_temp": self.genetic.boltzmann.max_temp,
            "boltzmann_adaptive": self.genetic.boltzmann.adaptive,
            "boltzmann_diversity_threshold": self.genetic.boltzmann.diversity_threshold,

            # Parameter Ranges
            "swarmrag_param_ranges": self.genetic.param_ranges.to_dict(),

            # Validation & Logging
            "validation_frequency": self.checkpoint.validation_frequency,
            "log_path": self.checkpoint.log_path,
            "plot_path": self.checkpoint.plot_path,
            "plot_title": self.checkpoint.plot_title,
            "checkpoint_frequency": self.checkpoint.checkpoint_frequency,
            "checkpoint_path": self.checkpoint.checkpoint_path,

            # LLM Settings
            "llm_provider": self.llm.provider,
            "llm_model": self.llm.model,
            "llm_env_path": self.llm.env_path,

            # MAP-Elites (always enabled)
            "map_elites_enabled": True,
            "map_elites_dims": self.map_elites.dimensions,
            "map_elites_bins": self.map_elites.bins,
            "map_elites_ranges": self.map_elites.ranges,
            "map_elites_initial_fill": self.map_elites.initial_fill,

            # Creative Mode Settings
            "creative_mode_enabled": self.creative_mode.enabled,
            "creative_mode_trigger_stagnation": self.creative_mode.trigger_stagnation,
            "creative_mode_trigger_fill_rate": self.creative_mode.trigger_fill_rate,
            "creative_mode_periodic_interval": self.creative_mode.periodic_interval,
            "creative_mode_max_per_generation": self.creative_mode.max_creative_per_generation,
            "creative_mode_complexity_limit": self.creative_mode.complexity_limit,
            "creative_mode_fallback_on_failure": self.creative_mode.fallback_on_failure,
        }

    @classmethod
    def from_flat_dict(cls, flat: Dict[str, Any]) -> "EvolutionConfig":
        """
        Create EvolutionConfig from legacy flat dict.

        Useful for loading old presets or checkpoint files.
        """
        config = cls()

        # Core settings
        config.n_generations = flat.get("n_generations", config.n_generations)
        config.fitness_strategy = flat.get("fitness_strategy", config.fitness_strategy)
        config.phased_switch_gen = flat.get("phased_switch_gen", config.phased_switch_gen)

        # Resources
        config.resources.concurrent_evaluations = flat.get(
            "concurrent_evaluations", config.resources.concurrent_evaluations
        )
        config.resources.max_workers_per_retrieval = flat.get(
            "max_workers_per_retrieval", config.resources.max_workers_per_retrieval
        )

        # MAP-Elites
        config.map_elites.dimensions = flat.get("map_elites_dims", config.map_elites.dimensions)
        config.map_elites.bins = flat.get("map_elites_bins", config.map_elites.bins)
        config.map_elites.ranges = flat.get("map_elites_ranges", config.map_elites.ranges)
        config.map_elites.initial_fill = flat.get("map_elites_initial_fill", config.map_elites.initial_fill)
        config.map_elites.batch_size = flat.get("population_size", config.map_elites.batch_size)

        # Genetic
        config.genetic.creation_strategy = flat.get("creation_strategy", config.genetic.creation_strategy)
        config.genetic.selection_strategy = flat.get("selection_strategy", config.genetic.selection_strategy)
        config.genetic.crossover_strategy = flat.get("crossover_strategy", config.genetic.crossover_strategy)
        config.genetic.mutation_strategy = flat.get("mutation_strategy", config.genetic.mutation_strategy)
        config.genetic.base_mutation_rate = flat.get("base_mutation_rate", config.genetic.base_mutation_rate)
        config.genetic.crossover_rate = flat.get("crossover_rate", config.genetic.crossover_rate)
        config.genetic.expr_max_depth = flat.get("expr_max_depth", config.genetic.expr_max_depth)
        config.genetic.mutation_max_expr_size = flat.get("mutation_max_expr_size", config.genetic.mutation_max_expr_size)
        config.genetic.n_agent_groups = flat.get("n_agent_groups", config.genetic.n_agent_groups)
        config.genetic.selection_k = flat.get("selection_k", config.genetic.selection_k)

        # Boltzmann
        config.genetic.boltzmann.temperature = flat.get("boltzmann_temperature", config.genetic.boltzmann.temperature)
        config.genetic.boltzmann.alpha = flat.get("boltzmann_alpha", config.genetic.boltzmann.alpha)
        config.genetic.boltzmann.min_temp = flat.get("boltzmann_min_temp", config.genetic.boltzmann.min_temp)
        config.genetic.boltzmann.max_temp = flat.get("boltzmann_max_temp", config.genetic.boltzmann.max_temp)
        config.genetic.boltzmann.adaptive = flat.get("boltzmann_adaptive", config.genetic.boltzmann.adaptive)
        config.genetic.boltzmann.diversity_threshold = flat.get(
            "boltzmann_diversity_threshold", config.genetic.boltzmann.diversity_threshold
        )

        # Param ranges
        ranges = flat.get("swarmrag_param_ranges", {})
        if ranges:
            config.genetic.param_ranges.n_agents = ranges.get("n_agents", config.genetic.param_ranges.n_agents)
            config.genetic.param_ranges.steps = ranges.get("steps", config.genetic.param_ranges.steps)
            config.genetic.param_ranges.decay = ranges.get("decay", config.genetic.param_ranges.decay)
            config.genetic.param_ranges.initial_pool_size = ranges.get(
                "initial_pool_size", config.genetic.param_ranges.initial_pool_size
            )
            config.genetic.param_ranges.start_subset = ranges.get(
                "start_subset", config.genetic.param_ranges.start_subset
            )
            config.genetic.param_ranges.drop_zone_inc = ranges.get(
                "drop_zone_inc", config.genetic.param_ranges.drop_zone_inc
            )

        # Checkpoint
        config.checkpoint.log_path = flat.get("log_path", config.checkpoint.log_path)
        config.checkpoint.plot_path = flat.get("plot_path", config.checkpoint.plot_path)
        config.checkpoint.plot_title = flat.get("plot_title", config.checkpoint.plot_title)
        config.checkpoint.checkpoint_path = flat.get("checkpoint_path", config.checkpoint.checkpoint_path)
        config.checkpoint.checkpoint_frequency = flat.get("checkpoint_frequency", config.checkpoint.checkpoint_frequency)
        config.checkpoint.validation_frequency = flat.get("validation_frequency", config.checkpoint.validation_frequency)

        # LLM
        config.llm.provider = flat.get("llm_provider", config.llm.provider)
        config.llm.model = flat.get("llm_model", config.llm.model)
        config.llm.env_path = flat.get("llm_env_path", config.llm.env_path)
        # Enable LLM if mutation strategy is llm_mutation
        if config.genetic.mutation_strategy == "llm_mutation":
            config.llm.enabled = True

        # Creative Mode
        config.creative_mode.enabled = flat.get(
            "creative_mode_enabled", config.creative_mode.enabled
        )
        config.creative_mode.trigger_stagnation = flat.get(
            "creative_mode_trigger_stagnation", config.creative_mode.trigger_stagnation
        )
        config.creative_mode.trigger_fill_rate = flat.get(
            "creative_mode_trigger_fill_rate", config.creative_mode.trigger_fill_rate
        )
        config.creative_mode.periodic_interval = flat.get(
            "creative_mode_periodic_interval", config.creative_mode.periodic_interval
        )
        config.creative_mode.max_creative_per_generation = flat.get(
            "creative_mode_max_per_generation", config.creative_mode.max_creative_per_generation
        )
        config.creative_mode.complexity_limit = flat.get(
            "creative_mode_complexity_limit", config.creative_mode.complexity_limit
        )
        config.creative_mode.fallback_on_failure = flat.get(
            "creative_mode_fallback_on_failure", config.creative_mode.fallback_on_failure
        )

        return config


# =============================================================================
# Default Config Instance
# =============================================================================

DEFAULT_CONFIG = EvolutionConfig()


# =============================================================================
# Evolution Context (Runtime State)
# =============================================================================

@dataclass
class EvolutionContext:
    """
    Shared context passed to all genetic operators (Selection, Crossover, Mutation).

    Contains both static config and dynamic runtime state.
    """
    # Configuration (new dataclass-based)
    config: EvolutionConfig = field(default_factory=EvolutionConfig)

    # Current State
    generation: int = 0
    population: List["Genome"] = field(default_factory=list)
    global_mutation_multiplier: float = 1.0

    # Registry Data (What features can we mutate into?)
    available_features: List[str] = field(default_factory=list)
    expression_features: Dict[str, List[str]] = field(default_factory=dict)

    # State for Adaptive Strategies
    current_temperature: float = 1.0

    # LLM Integration (single provider interface)
    llm_provider: Optional[Any] = None

    # Creative Mode State (runtime tracking)
    stagnation_count: int = 0              # Generations without improvement
    archive_fill_rate: float = 0.0         # Current archive fill rate
    top_fitness_unchanged: int = 0         # Generations where top fitness unchanged
    creative_mutations_this_gen: int = 0   # Creative mutations used this generation
    creative_success_count: int = 0        # Successful creative mutations
    creative_failure_count: int = 0        # Failed creative mutations

    def get_flat_config(self) -> Dict[str, Any]:
        """Get config as flat dict for backwards compatibility."""
        return self.config.to_flat_dict()

    def reset_creative_gen_count(self):
        """Reset the per-generation creative mutation counter."""
        self.creative_mutations_this_gen = 0

    def can_use_creative_mode(self) -> bool:
        """Check if creative mode can be used this generation."""
        if not self.config.creative_mode.enabled:
            return False
        max_per_gen = self.config.creative_mode.max_creative_per_generation
        return self.creative_mutations_this_gen < max_per_gen


# =============================================================================
# Backwards Compatibility Layer
# =============================================================================

# For code that still imports these (will be removed in future)
EvolutionConfigDict = Dict[str, Any]  # Type alias for gradual migration

def get_default_flat_config() -> Dict[str, Any]:
    """Get default config as flat dict (backwards compatibility)."""
    return DEFAULT_CONFIG.to_flat_dict()

# Alias for old DEFAULT_EVO_CONFIG usage
DEFAULT_EVO_CONFIG = get_default_flat_config()


# =============================================================================
# Dynamic Configuration Helpers
# =============================================================================

def get_dynamic_batch_size(
    archive_fill_rate: float,
    config: ResourceConfig
) -> int:
    """
    Calculate dynamic batch size based on archive state.

    Larger batches when archive is sparse (explore more),
    smaller batches when full (refine existing).

    Args:
        archive_fill_rate: Fraction of archive cells occupied (0.0-1.0)
        config: ResourceConfig with batch size settings

    Returns:
        Batch size adjusted for current archive state
    """
    if not config.enable_dynamic_batch_size:
        return config.base_batch_size

    if archive_fill_rate < 0.3:
        # Sparse archive: explore more
        batch_size = int(config.base_batch_size * 1.5)
    elif archive_fill_rate > 0.7:
        # Full archive: refine existing
        batch_size = int(config.base_batch_size * 0.6)
    else:
        # Mid-range: standard batch
        batch_size = config.base_batch_size

    # Clamp to configured limits
    return max(config.min_batch_size, min(config.max_batch_size, batch_size))


def get_dynamic_evaluation_config(
    archive_fill_rate: float,
    generation: int,
    config: EvolutionConfig
) -> Dict[str, Any]:
    """
    Get dynamically adjusted evaluation parameters.

    Adjusts evaluation intensity based on evolutionary progress.

    Args:
        archive_fill_rate: Fraction of archive cells occupied
        generation: Current generation number
        config: Full evolution config

    Returns:
        Dict with adjusted evaluation parameters
    """
    base_queries = 100

    # Early generations: lighter evaluation for faster exploration
    if generation < 10:
        query_fraction = 0.6
        decision_sample_rate = 0.05  # Light tracking
    # Mid generations: standard evaluation
    elif generation < config.n_generations * 0.7:
        query_fraction = 0.8
        decision_sample_rate = 0.1
    # Late generations: full evaluation for refinement
    else:
        query_fraction = 1.0
        decision_sample_rate = 0.15

    # Adjust based on archive state
    if archive_fill_rate > 0.8:
        # Archive is full, need discriminative evaluation
        query_fraction = min(1.0, query_fraction * 1.2)

    return {
        "max_queries": int(base_queries * query_fraction),
        "decision_sample_rate": decision_sample_rate,
        "batch_size": get_dynamic_batch_size(archive_fill_rate, config.resources),
    }
