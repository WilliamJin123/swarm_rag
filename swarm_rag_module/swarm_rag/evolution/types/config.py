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

    # Optimization flags for shared pre-computation
    enable_shared_precompute: bool = True  # Pre-compute query embeddings and initial pools once per generation
    enable_cross_genome_metric_batch: bool = True  # Batch metric computation across all genomes


@dataclass
class MapElitesConfig:
    """MAP-Elites archive and breeding settings."""
    dimensions: List[str] = field(default_factory=lambda: ["aggressiveness", "complexity"])
    bins: List[int] = field(default_factory=lambda: [15, 12])
    ranges: List[Tuple[float, float]] = field(default_factory=lambda: [(10.0, 150.0), (5.0, 60.0)])
    initial_fill: int = 100
    batch_size: int = 30  # offspring per generation
    comparison_mode: str = "quality_only"  # quality_only, weighted_composite, metric_threshold, lexicographic


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
class StorageConfig:
    """
    Unified storage configuration with dataset-first organization.

    Directory structure:
        {base_dir}/{dataset}/{run_id}/
            config.json         # Full experiment config snapshot
            checkpoints/
                latest.pkl      # Most recent checkpoint
                gen_000.pkl     # Per-generation checkpoints
            logs/
                evolution.jsonl # Main evolution log
            plots/
                progress.png    # Evolution progress plot
            results/
                best_genome.json    # Best genome parameters
                final_metrics.json  # Final evaluation metrics
    """
    base_dir: str = "runs"
    dataset: str = "prime"
    run_id: Optional[str] = None  # Auto-generated if None
    use_gpu: str = "auto"  # "auto", "always", "never"

    checkpoint_frequency: int = 5
    validation_frequency: int = 5
    keep_n_checkpoints: int = 10  # 0 = keep all
    plot_title: str = "MAP-Elites Evolution"

    # Computed paths (set in __post_init__)
    run_dir: str = field(default="", init=False)
    checkpoint_dir: str = field(default="", init=False)
    log_dir: str = field(default="", init=False)
    plot_dir: str = field(default="", init=False)
    results_dir: str = field(default="", init=False)

    def __post_init__(self):
        if self.run_id is None:
            from datetime import datetime
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._resolve_paths()

    def _resolve_paths(self):
        """Compute all directory paths from base_dir, dataset, and run_id."""
        self.run_dir = os.path.join(self.base_dir, self.dataset, self.run_id)
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.log_dir = os.path.join(self.run_dir, "logs")
        self.plot_dir = os.path.join(self.run_dir, "plots")
        self.results_dir = os.path.join(self.run_dir, "results")

    @property
    def latest_checkpoint_path(self) -> str:
        """Path to latest.pkl checkpoint file."""
        return os.path.join(self.checkpoint_dir, "latest.pkl")

    @property
    def log_path(self) -> str:
        """Path to main evolution log (JSONL format)."""
        return os.path.join(self.log_dir, "evolution.jsonl")

    @property
    def plot_path(self) -> str:
        """Path to evolution progress plot."""
        return os.path.join(self.plot_dir, "progress.png")

    @property
    def best_genome_path(self) -> str:
        """Path to best genome JSON file."""
        return os.path.join(self.results_dir, "best_genome.json")

    @property
    def config_snapshot_path(self) -> str:
        """Path to config snapshot JSON file."""
        return os.path.join(self.run_dir, "config.json")

    def checkpoint_path_for_gen(self, generation: int) -> str:
        """Get checkpoint path for a specific generation."""
        return os.path.join(self.checkpoint_dir, f"gen_{generation:03d}.pkl")

    def ensure_directories(self):
        """Create all required directories for the run."""
        for d in [self.checkpoint_dir, self.log_dir, self.plot_dir, self.results_dir]:
            os.makedirs(d, exist_ok=True)


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
    fitness_strategy: str = "lexicographic"  # lexicographic, pareto

    # Nested configs
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    map_elites: MapElitesConfig = field(default_factory=MapElitesConfig)
    genetic: GeneticConfig = field(default_factory=GeneticConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    creative_mode: CreativeModeConfig = field(default_factory=CreativeModeConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)



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

    @property
    def device(self) -> str:
        """
        Get resolved device string from storage config.

        Returns:
            Device string: "cuda" or "cpu"
        """
        from ...utils.device import get_device
        use_gpu = self.config.storage.use_gpu
        if use_gpu == "never":
            return "cpu"
        elif use_gpu == "always":
            return "cuda"
        return get_device()  # auto-detect


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
