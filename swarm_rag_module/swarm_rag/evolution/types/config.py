"""
Evolution Configuration System

Organized nested dataclasses replacing the flat 50+ field TypedDict.
MAP-Elites is the default and only evolution paradigm.
Supports dual-mode evolution: weighted_sum (linear) and expression_tree (symbolic).
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, TYPE_CHECKING, Literal
from enum import Enum
import os
import math
import torch

if TYPE_CHECKING:
    from .genome import Genome


# =============================================================================
# Genome Mode Enum (defined here to avoid circular imports)
# =============================================================================

class GenomeMode(Enum):
    """
    Genome evolution modes.

    - WEIGHTED_SUM: Linear heuristic combinations (fast, GPU-optimized)
    - EXPRESSION_TREE: Nonlinear symbolic expressions (expressive, default)
    """
    WEIGHTED_SUM = "weighted_sum"
    EXPRESSION_TREE = "expression_tree"


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
# Dual-Mode Evolution Data Structures
# =============================================================================

def _get_default_device() -> str:
    """Get device for default tensor creation."""
    from ...utils.device import get_device
    return get_device()


@dataclass
class WeightTensors:
    """
    GPU-friendly weight storage for heterogeneous agent groups.

    All weights stored as contiguous tensors for GPU batch operations.
    Used in weighted_sum genome mode for fast linear heuristic combinations.
    """
    # Movement: (n_groups, n_movement_features)
    movement_weights: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 4, device=_get_default_device()))
    movement_biases: torch.Tensor = field(default_factory=lambda: torch.zeros(1, device=_get_default_device()))

    # Deposit: (n_groups, n_deposit_features)
    deposit_weights: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 3, device=_get_default_device()))
    deposit_biases: torch.Tensor = field(default_factory=lambda: torch.zeros(1, device=_get_default_device()))

    # Ranking: shared across groups (n_ranking_features,)
    ranking_weights: torch.Tensor = field(default_factory=lambda: torch.zeros(2, device=_get_default_device()))
    ranking_bias: float = 0.0

    def to_device(self, device: str) -> "WeightTensors":
        """Move all tensors to specified device."""
        return WeightTensors(
            movement_weights=self.movement_weights.to(device),
            movement_biases=self.movement_biases.to(device),
            deposit_weights=self.deposit_weights.to(device),
            deposit_biases=self.deposit_biases.to(device),
            ranking_weights=self.ranking_weights.to(device),
            ranking_bias=self.ranking_bias,
        )

    def copy(self) -> "WeightTensors":
        """Create a deep copy of all tensors."""
        return WeightTensors(
            movement_weights=self.movement_weights.clone(),
            movement_biases=self.movement_biases.clone(),
            deposit_weights=self.deposit_weights.clone(),
            deposit_biases=self.deposit_biases.clone(),
            ranking_weights=self.ranking_weights.clone(),
            ranking_bias=self.ranking_bias,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for checkpointing."""
        return {
            "movement_weights": self.movement_weights.cpu().tolist(),
            "movement_biases": self.movement_biases.cpu().tolist(),
            "deposit_weights": self.deposit_weights.cpu().tolist(),
            "deposit_biases": self.deposit_biases.cpu().tolist(),
            "ranking_weights": self.ranking_weights.cpu().tolist(),
            "ranking_bias": self.ranking_bias,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], device: str = None) -> "WeightTensors":
        """Deserialize from dictionary. Device defaults to auto-detection if not provided."""
        from ...utils.device import get_device
        target_device = device if device is not None else get_device()
        return cls(
            movement_weights=torch.as_tensor(d["movement_weights"], dtype=torch.float32, device=target_device),
            movement_biases=torch.as_tensor(d["movement_biases"], dtype=torch.float32, device=target_device),
            deposit_weights=torch.as_tensor(d["deposit_weights"], dtype=torch.float32, device=target_device),
            deposit_biases=torch.as_tensor(d["deposit_biases"], dtype=torch.float32, device=target_device),
            ranking_weights=torch.as_tensor(d["ranking_weights"], dtype=torch.float32, device=target_device),
            ranking_bias=d["ranking_bias"],
        )

    @property
    def n_groups(self) -> int:
        """Number of agent groups."""
        return self.movement_weights.shape[0]

    @property
    def total_params(self) -> int:
        """Total number of parameters."""
        return (
            self.movement_weights.numel() +
            self.movement_biases.numel() +
            self.deposit_weights.numel() +
            self.deposit_biases.numel() +
            self.ranking_weights.numel() +
            1  # ranking_bias
        )


@dataclass
class MutationSigmas:
    """
    Self-adaptive mutation parameters - these also evolve (ES-style).

    Each genome carries its own mutation strengths, allowing evolution
    to discover optimal mutation rates for different regions of the search space.
    """
    weight_sigma: float = 0.10       # Weight perturbation strength
    bias_sigma: float = 0.05         # Bias perturbation strength
    ratio_sigma: float = 0.10        # Group ratio shift strength
    hyperparam_sigma: float = 0.15   # Hyperparameter mutation strength

    # Meta-parameter for sigma evolution
    tau: float = 0.1  # Learning rate for sigma adaptation

    # Bounds to prevent collapse or explosion
    min_sigma: float = 0.01
    max_sigma: float = 0.50

    def adapt(self) -> "MutationSigmas":
        """
        ES-style sigma adaptation: sigma' = sigma * exp(tau * N(0,1))

        Returns a new MutationSigmas with adapted values.
        """
        import random

        def _adapt_single(sigma: float) -> float:
            new_sigma = sigma * math.exp(self.tau * random.gauss(0, 1))
            return max(self.min_sigma, min(self.max_sigma, new_sigma))

        return MutationSigmas(
            weight_sigma=_adapt_single(self.weight_sigma),
            bias_sigma=_adapt_single(self.bias_sigma),
            ratio_sigma=_adapt_single(self.ratio_sigma),
            hyperparam_sigma=_adapt_single(self.hyperparam_sigma),
            tau=self.tau,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
        )

    def copy(self) -> "MutationSigmas":
        """Create a copy of the sigmas."""
        return MutationSigmas(
            weight_sigma=self.weight_sigma,
            bias_sigma=self.bias_sigma,
            ratio_sigma=self.ratio_sigma,
            hyperparam_sigma=self.hyperparam_sigma,
            tau=self.tau,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
        )

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return {
            "weight_sigma": self.weight_sigma,
            "bias_sigma": self.bias_sigma,
            "ratio_sigma": self.ratio_sigma,
            "hyperparam_sigma": self.hyperparam_sigma,
            "tau": self.tau,
            "min_sigma": self.min_sigma,
            "max_sigma": self.max_sigma,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "MutationSigmas":
        """Deserialize from dictionary."""
        return cls(**d)


@dataclass
class HeuristicFeatureConfig:
    """
    Configurable feature sets for evolution runs.

    Specifies which heuristic features are available for each strategy type.
    This allows dataset-specific customization of the feature space.
    """
    movement: List[str] = field(default_factory=lambda: [
        "semantic_similarity_unnormalized",
        "node_centrality",
        "pheromone_repulsion",
    ])

    deposit: List[str] = field(default_factory=lambda: [
        "flat",
        "semantic_unnormalized",
        "exploration_bonus",
    ])

    ranking: List[str] = field(default_factory=lambda: [
        "semantic_rank",
        "percentage_visited",
    ])

    def to_dict(self) -> Dict[str, List[str]]:
        """Serialize to dictionary."""
        return {
            "movement": self.movement.copy(),
            "deposit": self.deposit.copy(),
            "ranking": self.ranking.copy(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, List[str]]) -> "HeuristicFeatureConfig":
        """Deserialize from dictionary."""
        return cls(
            movement=d.get("movement", cls().movement),
            deposit=d.get("deposit", cls().deposit),
            ranking=d.get("ranking", cls().ranking),
        )


# =============================================================================
# Predefined Feature Configurations
# =============================================================================

# STaRK-specific config (includes stark_centrality)
# Optimized for target metrics: Hit@1>60%, Hit@5>80%, MRR>80%, Recall@20>85%
STARK_FEATURES = HeuristicFeatureConfig(
    movement=[
        "semantic_similarity_unnormalized",  # Core relevance signal
        "stark_centrality",                  # STaRK graph structure
        "node_centrality",                   # Hub navigation
        "pheromone_repulsion",               # Avoid over-visited paths
        "random_jitter",                     # Escape local optima (improves Recall@20)
    ],
    deposit=[
        "flat",                              # Baseline uniform deposit
        "semantic_unnormalized",             # Relevance-weighted deposit
        "exploration_bonus",                 # Reward visiting new nodes
        "hub",                               # Reinforce hub highways (improves Hit@1/MRR)
        "collaborative_amplification",       # Amplify consensus paths (improves Hit@1/MRR)
    ],
    ranking=[
        "semantic_rank",                     # Final relevance ordering
        "percentage_visited",                # Swarm consensus signal
    ],
)

# General graph features (no dataset-specific heuristics)
GENERAL_FEATURES = HeuristicFeatureConfig(
    movement=[
        "semantic_similarity_unnormalized",
        "node_centrality",
        "pheromone_repulsion",
        "random_jitter",
    ],
    deposit=[
        "flat",
        "semantic_unnormalized",
        "exploration_bonus",
        "hub",
    ],
    ranking=[
        "semantic_rank",
        "percentage_visited",
    ],
)


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

    # Early exit threshold at halfway checkpoint (quality score threshold)
    early_exit_threshold: float = 0.30

    # Retrieval execution mode (for SwarmRetriever)
    # "batched" (default): Multi-query GPU batching for throughput
    # "sequential": Process queries one at a time
    run_mode: Literal["batched", "sequential"] = "batched"
    # Number of queries per GPU batch (only used when run_mode="batched")
    # Benchmark results: batch_size=64 gives ~47 q/s, batch_size=100 gives ~52 q/s
    run_batch_size: int = 100

    # Profiler settings (for StepProfiler in SwarmRetriever)
    # Maximum samples per profiler section before rolling window kicks in
    # Can also be set via SWARM_PROFILE_SAMPLES environment variable
    profiler_max_samples: int = 1000


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
    """Valid ranges for SwarmRAG parameters.

    These are the authoritative ranges for all evolvable parameters.
    Values are tightened based on empirical analysis of effective configurations.
    """
    n_agents: Tuple[int, int] = (15, 50)           # Was (5, 30) - fewer agents = faster, too many = redundant
    steps: Tuple[int, int] = (3, 7)                 # Was (4, 12) - most signal captured in 3-6 steps
    decay: Tuple[float, float] = (0.3, 0.8)         # Was (0.85, 0.99) - keep dynamic, tightened range
    initial_pool_size: Tuple[int, int] = (20, 60)   # Was (10, 50) - tight range around optimal
    start_subset: Tuple[int, int] = (5, 15)         # Keep unchanged
    drop_zone_inc: Tuple[float, float] = (0.05, 0.2)  # Keep unchanged

    def to_evolvable_dict(self) -> Dict[str, Tuple[float, float]]:
        """Return evolvable parameters as dict (excludes fixed params like start_subset, drop_zone_inc)."""
        return {
            "n_agents": self.n_agents,
            "steps": self.steps,
            "decay": self.decay,
            "initial_pool_size": self.initial_pool_size,
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

    # Parallel mutation settings
    parallel_mutation_workers: int = 4  # Number of workers for parallel offspring generation

    # === Weighted Sum Mode Specific ===
    n_agent_groups_range: Tuple[int, int] = (1, 5)  # Min/max groups for evolution
    self_adaptive_mutation: bool = True  # Enable ES-style sigma evolution

    # Initial sigma values for weighted sum mode
    initial_weight_sigma: float = 0.10
    initial_bias_sigma: float = 0.05
    initial_ratio_sigma: float = 0.10
    initial_hyperparam_sigma: float = 0.15
    sigma_tau: float = 0.1  # Sigma learning rate

    # === Mutation Operator Probabilities (weighted sum mode) ===
    # These control how often each mutation type is applied
    # Must sum to 1.0
    mutation_prob_weight: float = 0.60      # Weight perturbation
    mutation_prob_bias: float = 0.15        # Bias perturbation
    mutation_prob_ratio: float = 0.10       # Group ratio shift
    mutation_prob_hyperparam: float = 0.10  # Hyperparameter mutation
    mutation_prob_group_change: float = 0.05  # Add/remove agent groups


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
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"

    checkpoint_frequency: int = 5
    validation_frequency: int = 5
    keep_n_checkpoints: int = 10  # 0 = keep all
    async_checkpoints: bool = True  # Enable async checkpoint writing
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

    Supports dual-mode evolution:
    - GenomeMode.EXPRESSION_TREE: Nonlinear symbolic expressions (current system, expressive)
    - GenomeMode.WEIGHTED_SUM: Linear heuristic combinations (fast, interpretable)

    Example:
        # Weighted sum mode for fast evolution
        config = EvolutionConfig(
            genome_mode=GenomeMode.WEIGHTED_SUM,
            heuristic_features=STARK_FEATURES,
            n_generations=1000,
        )

        # Expression tree mode (default, existing behavior)
        config = EvolutionConfig(
            genome_mode=GenomeMode.EXPRESSION_TREE,
            n_generations=100,
        )
    """
    # === Mode Selection ===
    genome_mode: GenomeMode = GenomeMode.EXPRESSION_TREE

    # === Feature Configuration (weighted_sum mode) ===
    heuristic_features: HeuristicFeatureConfig = field(
        default_factory=HeuristicFeatureConfig
    )

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

    # Resolved device (computed on init)
    resolved_device: str = field(default="", init=False)

    def __post_init__(self):
        """Initialize computed fields."""
        if not self.resolved_device:
            from ...utils.device import resolve_device
            self.resolved_device = resolve_device(self.config.storage.device)

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
        Get resolved device string.

        Returns:
            Device string: "cuda", "mps", or "cpu"
        """
        return self.resolved_device


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
