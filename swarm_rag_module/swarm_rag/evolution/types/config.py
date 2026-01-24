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
