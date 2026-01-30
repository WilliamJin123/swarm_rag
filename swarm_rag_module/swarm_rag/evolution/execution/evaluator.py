"""
Population Evaluator with Single-Checkpoint Early Exit.

Implements simplified evaluation with a single quarter checkpoint for early exit.
Poor-performing genomes are filtered at the 25% point, while promising genomes
get full evaluation.

Optimization features:
- Shared pre-computation: Query embeddings and initial pools computed once per generation
- Cross-genome metric batching: Single GPU call for metrics across all genomes
- CPU thread pool capping: Prevents worker over-subscription
- Single quarter checkpoint: Filters bad genomes 3x faster than halfway
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
import torch
from typing import List, Any, Optional, Dict, Tuple, Set
from dataclasses import dataclass, field

from swarm_rag.interfaces.protocols import RetrievalBackend
from ...eval.metrics import Evaluator
from .memory_guard import MemoryGuard, MemoryThresholdExceeded
from .fitness import FitnessCalculator
from ..types.genome import GenomeCompiler, Genome
from ..types.config import HeuristicFeatureConfig, GenomeMode
from .shared_precompute import (
    SharedPrecomputeContext,
    prepare_shared_context,
    get_unique_pool_sizes,
    BatchedRetrievalResults
)
from .fitness_cache import FitnessCache, CacheStats
from .embedding_cache import EmbeddingCacheProvider


logger = logging.getLogger(__name__)


# Default early exit threshold at quarter checkpoint (25% of queries)
# DEPRECATED: Use ResourceConfig.early_exit_threshold instead for new code.
# This constant is kept for backward compatibility with existing tests and imports.
DEFAULT_EARLY_EXIT_THRESHOLD: float = 0.30


@dataclass
class EvaluationStats:
    """Statistics about evaluation efficiency."""
    total_genomes: int = 0
    tier_exits: Dict[str, int] = field(default_factory=dict)
    avg_queries_per_genome: float = 0.0
    time_saved_estimate: float = 0.0


@dataclass
class EvaluatorConfig:
    """
    Configuration for PopulationEvaluator.

    Groups related parameters for cleaner initialization.
    Use PopulationEvaluatorBuilder for fluent construction.
    """
    # Required dependencies (no defaults)
    retriever: Optional[RetrievalBackend] = None
    evaluator: Optional[Evaluator] = None
    fitness_calc: Optional[FitnessCalculator] = None

    # Data
    queries: List[str] = field(default_factory=list)
    ground_truth: List[List[Any]] = field(default_factory=list)

    # Concurrency
    concurrent_evaluations: int = 4
    max_workers_per_retrieval: int = 1

    # Decision tracking
    track_decisions: bool = False
    decision_sample_rate: float = 1.0

    # Early exit
    early_exit_threshold: float = DEFAULT_EARLY_EXIT_THRESHOLD
    enable_adaptive: bool = True

    # Device and optimization
    device: Optional[Any] = None
    enable_shared_precompute: bool = True
    enable_cross_genome_metric_batch: bool = True

    # Feature configuration
    heuristic_features: Optional[HeuristicFeatureConfig] = None

    # Execution mode
    run_mode: str = "batched"
    run_batch_size: int = 100


class PopulationEvaluatorBuilder:
    """
    Builder for PopulationEvaluator with fluent interface.

    Simplifies construction when many parameters need customization.

    Example:
        evaluator = (PopulationEvaluatorBuilder()
            .with_retriever(retriever)
            .with_evaluator(evaluator)
            .with_fitness_calc(fitness_calc)
            .with_queries(queries, ground_truth)
            .with_device("cuda")
            .enable_early_exit(threshold=0.25)
            .build())
    """

    def __init__(self):
        """Initialize builder with default configuration."""
        self._config = EvaluatorConfig()

    def with_retriever(self, retriever: RetrievalBackend) -> "PopulationEvaluatorBuilder":
        """Set the retrieval backend."""
        self._config.retriever = retriever
        return self

    def with_evaluator(self, evaluator: Evaluator) -> "PopulationEvaluatorBuilder":
        """Set the metrics evaluator."""
        self._config.evaluator = evaluator
        return self

    def with_fitness_calc(self, fitness_calc: FitnessCalculator) -> "PopulationEvaluatorBuilder":
        """Set the fitness calculator."""
        self._config.fitness_calc = fitness_calc
        return self

    def with_queries(
        self, queries: List[str], ground_truth: List[List[Any]]
    ) -> "PopulationEvaluatorBuilder":
        """Set queries and ground truth data."""
        self._config.queries = queries
        self._config.ground_truth = ground_truth
        return self

    def with_concurrency(
        self, concurrent_evaluations: int = 4, max_workers_per_retrieval: int = 1
    ) -> "PopulationEvaluatorBuilder":
        """Set concurrency parameters."""
        self._config.concurrent_evaluations = concurrent_evaluations
        self._config.max_workers_per_retrieval = max_workers_per_retrieval
        return self

    def with_decision_tracking(
        self, enabled: bool = True, sample_rate: float = 1.0
    ) -> "PopulationEvaluatorBuilder":
        """Enable decision tracking for LLM context."""
        self._config.track_decisions = enabled
        self._config.decision_sample_rate = sample_rate
        return self

    def enable_early_exit(
        self, threshold: float = DEFAULT_EARLY_EXIT_THRESHOLD
    ) -> "PopulationEvaluatorBuilder":
        """Enable adaptive early exit at quarter checkpoint."""
        self._config.enable_adaptive = True
        self._config.early_exit_threshold = threshold
        return self

    def disable_early_exit(self) -> "PopulationEvaluatorBuilder":
        """Disable adaptive early exit (full evaluation only)."""
        self._config.enable_adaptive = False
        return self

    def with_device(self, device: Any) -> "PopulationEvaluatorBuilder":
        """Set the target device (cuda, mps, cpu, or torch.device)."""
        self._config.device = device
        return self

    def with_optimization(
        self, shared_precompute: bool = True, cross_genome_batch: bool = True
    ) -> "PopulationEvaluatorBuilder":
        """Configure optimization flags."""
        self._config.enable_shared_precompute = shared_precompute
        self._config.enable_cross_genome_metric_batch = cross_genome_batch
        return self

    def with_heuristic_features(
        self, features: HeuristicFeatureConfig
    ) -> "PopulationEvaluatorBuilder":
        """Set heuristic features for weighted sum mode."""
        self._config.heuristic_features = features
        return self

    def with_run_mode(
        self, mode: str = "batched", batch_size: int = 100
    ) -> "PopulationEvaluatorBuilder":
        """Set execution mode (batched or sequential)."""
        self._config.run_mode = mode
        self._config.run_batch_size = batch_size
        return self

    def build(self) -> "PopulationEvaluator":
        """
        Build the PopulationEvaluator.

        Raises:
            ValueError: If required dependencies are not set.
        """
        if self._config.retriever is None:
            raise ValueError("retriever is required - use with_retriever()")
        if self._config.evaluator is None:
            raise ValueError("evaluator is required - use with_evaluator()")
        if self._config.fitness_calc is None:
            raise ValueError("fitness_calc is required - use with_fitness_calc()")

        return PopulationEvaluator(
            retriever=self._config.retriever,
            evaluator=self._config.evaluator,
            fitness_calc=self._config.fitness_calc,
            concurrent_evaluations=self._config.concurrent_evaluations,
            max_workers_per_retrieval=self._config.max_workers_per_retrieval,
            queries=self._config.queries,
            ground_truth=self._config.ground_truth,
            track_decisions=self._config.track_decisions,
            decision_sample_rate=self._config.decision_sample_rate,
            early_exit_threshold=self._config.early_exit_threshold,
            enable_adaptive=self._config.enable_adaptive,
            device=self._config.device,
            enable_shared_precompute=self._config.enable_shared_precompute,
            enable_cross_genome_metric_batch=self._config.enable_cross_genome_metric_batch,
            heuristic_features=self._config.heuristic_features,
            run_mode=self._config.run_mode,
            run_batch_size=self._config.run_batch_size,
        )


class PopulationEvaluator:
    """
    Evaluator with single-checkpoint early exit.

    Implements simplified evaluation with a single quarter checkpoint.
    Poor-performing genomes are filtered at the 25% point based on
    quality score threshold.

    This reduces evaluation time by:
    1. Evaluating first quarter of queries
    2. Computing quality score at quarter point
    3. Early exit if quality < threshold
    4. Full evaluation only for promising genomes

    Args:
        retriever: RetrievalBackend implementation
        evaluator: Metrics evaluator
        fitness_calc: Fitness calculator
        concurrent_evaluations: Max parallel genome evaluations
        max_workers_per_retrieval: Workers per retrieval batch
        queries: All available queries
        ground_truth: Ground truth for queries
        track_decisions: Enable decision tracking for LLM context
        decision_sample_rate: Fraction of queries to track
        early_exit_threshold: Quality threshold at quarter point (default: 0.30)
        enable_adaptive: Whether to use early exit (True) or full evaluation only
    """

    def __init__(
        self,
        retriever: RetrievalBackend,
        evaluator: Evaluator,
        fitness_calc: FitnessCalculator,
        concurrent_evaluations: int = 4,
        max_workers_per_retrieval: int = 1,
        queries: List[str] = None,
        ground_truth: List[List[Any]] = None,
        track_decisions: bool = False,
        decision_sample_rate: float = 1.0,
        early_exit_threshold: float = DEFAULT_EARLY_EXIT_THRESHOLD,
        enable_adaptive: bool = True,
        device: Any = None,  # torch.device or str
        enable_shared_precompute: bool = True,
        enable_cross_genome_metric_batch: bool = True,
        heuristic_features: HeuristicFeatureConfig = None,
        run_mode: str = "batched",  # "batched" or "sequential"
        run_batch_size: int = 100,  # GPU batch size for multi-query traversal
    ):
        self.retriever = retriever
        self.evaluator = evaluator
        self.fitness_calc = fitness_calc
        self.queries = queries or []
        self.ground_truth = ground_truth or []
        self.concurrent_evaluations = concurrent_evaluations
        self.max_workers_per_retrieval = max_workers_per_retrieval
        self.track_decisions = track_decisions
        self.decision_sample_rate = decision_sample_rate
        self.early_exit_threshold = early_exit_threshold
        self.enable_adaptive = enable_adaptive
        self.run_mode = run_mode
        self.run_batch_size = run_batch_size

        # Compilers for both modes
        self._expression_compiler = GenomeCompiler()
        self._weighted_sum_compiler = None
        self._heuristic_features = heuristic_features

        # Lazy-load weighted sum compiler when needed
        if heuristic_features is not None:
            self._init_weighted_sum_compiler(heuristic_features)

        # Legacy compatibility
        self.compiler = self._expression_compiler

        # Optimization flags
        self.enable_shared_precompute = enable_shared_precompute
        self.enable_cross_genome_metric_batch = enable_cross_genome_metric_batch

        # Store device, auto-detect if not provided
        if device is None:
            from ...utils.device import get_device
            self.device = get_device()
        else:
            # Handle torch.device objects
            self.device = str(device) if hasattr(device, 'type') else device

        # Track evaluation statistics
        self.stats = EvaluationStats()
        self._reset_stats()

        # Cached shared context (reused within a generation)
        self._shared_context: Optional[SharedPrecomputeContext] = None

        # Fitness cache for skipping duplicate genome evaluations
        self._fitness_cache = FitnessCache()

    def _init_weighted_sum_compiler(self, heuristic_features: HeuristicFeatureConfig):
        """Initialize the weighted sum compiler with feature configuration."""
        try:
            from .weighted_sum import WeightedSumCompiler
            self._weighted_sum_compiler = WeightedSumCompiler(heuristic_features)
        except ImportError:
            logger.warning("WeightedSumCompiler not available, weighted_sum genomes will fail")

    def _get_compiler_for_genome(self, genome: Genome):
        """Get the appropriate compiler for a genome based on its mode."""
        if genome.mode == GenomeMode.WEIGHTED_SUM:
            if self._weighted_sum_compiler is None:
                # Try to initialize with default features
                if self._heuristic_features is None:
                    self._heuristic_features = HeuristicFeatureConfig()
                self._init_weighted_sum_compiler(self._heuristic_features)

            if self._weighted_sum_compiler is None:
                raise RuntimeError(
                    f"Cannot compile weighted_sum genome {genome.id}: "
                    "WeightedSumCompiler not available"
                )
            return self._weighted_sum_compiler
        else:
            return self._expression_compiler

    def _reset_stats(self):
        """Reset evaluation statistics for a new run."""
        self.stats = EvaluationStats(
            tier_exits={"early_exit": 0, "full": 0}
        )

    def evaluate(
        self,
        population: List[Genome],
        queries: List[str] = None,
        ground_truth: List[List[Any]] = None,
        generation: int = 0
    ) -> EvaluationStats:
        """
        Evaluates the population in-place using adaptive progressive sampling.

        When shared_precompute is enabled, this method pre-computes query embeddings
        and initial pools once, then reuses them across all genome evaluations.

        Returns:
            EvaluationStats with information about evaluation efficiency
        """
        queries = queries or self.queries
        ground_truth = ground_truth or self.ground_truth

        self._reset_stats()

        unevaluated = [g for g in population if not g.evaluated]
        if not unevaluated:
            return self.stats

        # Check cache for each unevaluated genome
        # Duplicates and elites from previous gens may have cached fitness
        from ..types.fitness_results import FitnessResult
        cache_restored = []
        still_unevaluated = []
        for genome in unevaluated:
            cached_fitness = self._fitness_cache.get(genome)
            if cached_fitness is not None:
                # Restore from cache without re-evaluation
                genome.fitness = FitnessResult(quality_score=cached_fitness)
                genome.evaluated = True
                cache_restored.append(genome)
            else:
                still_unevaluated.append(genome)
        unevaluated = still_unevaluated

        if not unevaluated:
            # All genomes hit cache - finalize and return
            cache_stats = self._fitness_cache.finalize_generation(generation)
            logger.info(f"All {len(cache_restored)} genomes restored from cache (100% hit rate)")
            return self.stats

        self.stats.total_genomes = len(unevaluated)
        batch_size = self.concurrent_evaluations

        logger.info(f"Evaluating {len(unevaluated)} genomes ({len(cache_restored)} restored from cache)...")
        logger.info(f"  > Concurrency: {batch_size} | Adaptive: {self.enable_adaptive}")
        logger.info(f"  > Shared precompute: {self.enable_shared_precompute} | Cross-genome batch: {self.enable_cross_genome_metric_batch}")
        if self.enable_adaptive:
            quarter = len(queries) // 4
            logger.info(f"  > Early exit: quarter={quarter}, threshold={self.early_exit_threshold}")

        # Prepare shared context if enabled
        shared_context = None
        if self.enable_shared_precompute and hasattr(self.retriever, 'retrieve_batch_with_precomputed'):
            shared_context = self._prepare_shared_context(unevaluated, queries, ground_truth)
            self._shared_context = shared_context

        # Initialize buffer pool if retriever supports it
        if hasattr(self.retriever, 'init_buffer_pool'):
            # Determine max sizes from genomes
            max_pool_size = max(
                (self._get_compiler_for_genome(g).compile(g).get('initial_pool_size', 30)
                 for g in unevaluated),
                default=100
            )
            max_agents = max(
                (self._get_compiler_for_genome(g).compile(g).get('n_agents', 10)
                 for g in unevaluated),
                default=50
            )
            # Use graph's average degree * 2 as estimate for max_degree
            max_degree = int(getattr(self.retriever, 'avg_degree', 50) * 2)

            self.retriever.init_buffer_pool(
                max_pool_size=max_pool_size * 2,  # 2x headroom
                max_agents=max_agents * 2,
                max_degree=max_degree
            )
            logger.info(f"  > Buffer pool initialized: pool={max_pool_size*2}, agents={max_agents*2}")

        total_queries_used = 0

        if shared_context is not None and not self.enable_adaptive:
            # Use optimized path with shared context (non-adaptive mode)
            total_queries_used = self._evaluate_all_with_shared(
                unevaluated, queries, ground_truth, shared_context
            )
        else:
            # Standard batch evaluation (with or without shared context for adaptive)
            for i in range(0, len(unevaluated), batch_size):
                batch = unevaluated[i:i + batch_size]
                if shared_context is not None:
                    queries_used = self._evaluate_batch_with_shared(
                        batch, queries, ground_truth, shared_context
                    )
                else:
                    queries_used = self._evaluate_batch(batch, queries, ground_truth)
                total_queries_used += queries_used

        # Clear shared context after evaluation
        self._shared_context = None

        # Release buffer pool to free memory
        if hasattr(self.retriever, '_buffer_pool') and self.retriever._buffer_pool is not None:
            self.retriever._buffer_pool.release()
            self.retriever._buffer_pool = None

        # Compute stats
        max_queries = len(queries) * len(unevaluated)
        self.stats.avg_queries_per_genome = total_queries_used / max(1, len(unevaluated))
        self.stats.time_saved_estimate = 1.0 - (total_queries_used / max(1, max_queries))

        # GPU memory report (reading counters is ~0 overhead)
        if torch.cuda.is_available():
            gpu_mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            logger.info(f"Evaluation complete: GPU={gpu_mem_mb:.0f}MB (peak={gpu_peak_mb:.0f}MB)")
        else:
            logger.info(f"Evaluation complete:")
        logger.info(f"  > Avg queries/genome: {self.stats.avg_queries_per_genome:.1f} / {len(queries)}")
        logger.info(f"  > Time saved estimate: {self.stats.time_saved_estimate:.1%}")
        if self.enable_adaptive:
            logger.info(f"  > Tier exits: {self.stats.tier_exits}")

        # Finalize cache stats for this generation
        cache_stats = self._fitness_cache.finalize_generation(generation)
        logger.info(f"  > Cache: {cache_stats.hits}/{cache_stats.total} hits ({cache_stats.hit_rate:.1%})")

        return self.stats

    def _prepare_shared_context(
        self,
        genomes: List[Genome],
        queries: List[str],
        ground_truth: List[List[Any]]
    ) -> SharedPrecomputeContext:
        """
        Prepare shared pre-computed context for all genomes.

        This method computes query embeddings and initial pools ONCE,
        eliminating redundant computation across genome evaluations.
        """
        # Extract unique initial_pool_size values from genomes
        unique_pool_sizes = get_unique_pool_sizes(genomes, self.compiler)
        logger.info(f"  > Pre-computing shared context for {len(unique_pool_sizes)} unique pool sizes...")

        return prepare_shared_context(
            retriever=self.retriever,
            queries=queries,
            ground_truth=ground_truth,
            unique_pool_sizes=unique_pool_sizes,
            device=self.device
        )

    def _evaluate_all_with_shared(
        self,
        genomes: List[Genome],
        queries: List[str],
        ground_truth: List[List[Any]],
        shared_context: SharedPrecomputeContext
    ) -> int:
        """
        Evaluate all genomes using shared context and cross-genome metric batching.

        This is the most optimized path when:
        1. Shared precompute is enabled
        2. Adaptive evaluation is disabled (full evaluation for all)
        3. Cross-genome metric batching is enabled

        Returns total queries used.
        """
        logger.info("  > Using optimized evaluation with shared context...")

        n_genomes = len(genomes)
        n_queries = len(queries)
        total_queries_used = 0

        # Collect all retrieval results
        batched_results = BatchedRetrievalResults()

        for genome_idx, genome in enumerate(genomes):
            retriever_kwargs = self._get_compiler_for_genome(genome).compile(genome)
            pool_size = retriever_kwargs.get('initial_pool_size', 30)

            # Get pre-computed initial pools for this pool size
            initial_pools = shared_context.initial_pools.get(pool_size, [])
            if not initial_pools:
                # Fallback if pool size not pre-computed
                logger.warning(f"Pool size {pool_size} not pre-computed, using standard path")
                queries_used, _ = self._evaluate_single_full(genome, queries, ground_truth)
                total_queries_used += queries_used
                continue

            start_time = time.time()

            # Use pre-computed embeddings and pools
            results = self.retriever.retrieve_batch_with_precomputed(
                query_embeddings=shared_context.query_embeddings,
                initial_pools=initial_pools,
                max_workers=1 if self.run_mode == "sequential" else self.max_workers_per_retrieval,
                gpu_batch_size=self.run_batch_size,
                genome_id=genome.id,
                **retriever_kwargs
            )

            batched_results.add_genome_results(genome.id, results)
            total_queries_used += n_queries

            # Store retrieval time for later
            genome._retrieval_time = time.time() - start_time

            logger.debug(f"  > Genome '{genome.id}' ({genome_idx + 1}/{n_genomes}) retrieval complete")

            # Clear CUDA cache to prevent memory fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Batch compute metrics across all genomes
        if self.enable_cross_genome_metric_batch:
            self._batch_compute_metrics_all_genomes(
                genomes, batched_results, shared_context.ground_truth_sets,
                shared_context=shared_context
            )
        else:
            # Compute metrics individually
            for genome in genomes:
                results = batched_results.results_by_genome.get(genome.id, [])
                if results:
                    metrics = self._compute_metrics_cumulative(results, ground_truth)
                    retrieval_time = getattr(genome, '_retrieval_time', 0.0)
                    metrics['latency'] = retrieval_time / max(1, len(results))
                    metrics['complexity'] = float(genome.complexity())

                    genome.metrics = metrics
                    genome.fitness = self.fitness_calc.calculate(metrics, genome)
                    genome.evaluated = True

                    self.stats.tier_exits["full"] = self.stats.tier_exits.get("full", 0) + 1

        # Cleanup: release memory from batched results
        batched_results.clear()

        # Cleanup: release shared context tensors
        if shared_context.ground_truth_tensor is not None:
            del shared_context.ground_truth_tensor
            shared_context.ground_truth_tensor = None
        if shared_context.gt_sizes is not None:
            del shared_context.gt_sizes
            shared_context.gt_sizes = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return total_queries_used

    def _evaluate_batch_with_shared(
        self,
        batch: List[Genome],
        queries: List[str],
        ground_truth: List[List[Any]],
        shared_context: SharedPrecomputeContext
    ) -> int:
        """
        Evaluate a batch using shared pre-computed context.

        This method uses pre-computed query embeddings and initial pools
        for faster evaluation, with optional single-checkpoint early exit.
        """
        logger.debug(f"  > Starting batch of {len(batch)} genomes with shared context...")

        total_queries_used = 0
        completed_count = 0

        # Check if we can use the optimized GPU path with precomputed GT
        use_gpu_optimized = (
            self.device != "cpu" and
            self.enable_adaptive and
            shared_context.ground_truth_tensor is not None and
            shared_context.gt_sizes is not None
        )

        # GPU mode: sequential evaluation
        if self.device != "cpu":
            for genome in batch:
                if use_gpu_optimized:
                    # Use optimized path with single checkpoint
                    queries_used, exit_tier = self._evaluate_single_with_early_exit(
                        genome, queries, ground_truth, shared_context
                    )
                else:
                    queries_used, exit_tier = self._evaluate_single_with_shared(
                        genome, queries, ground_truth, shared_context
                    )
                total_queries_used += queries_used
                completed_count += 1

                if exit_tier in self.stats.tier_exits:
                    self.stats.tier_exits[exit_tier] += 1

                self._log_genome_result(genome, exit_tier, completed_count, len(batch))

                # Clear CUDA cache to prevent memory fragmentation and latency creep
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return total_queries_used

        # CPU mode: parallel evaluation
        with ThreadPoolExecutor(max_workers=min(len(batch), self.concurrent_evaluations)) as executor:
            future_to_genome = {
                executor.submit(
                    self._evaluate_single_with_shared, g, queries, ground_truth, shared_context
                ): g
                for g in batch
            }

            for future in as_completed(future_to_genome):
                genome = future_to_genome[future]
                completed_count += 1

                queries_used, exit_tier = future.result()
                total_queries_used += queries_used

                if exit_tier in self.stats.tier_exits:
                    self.stats.tier_exits[exit_tier] += 1

                self._log_genome_result(genome, exit_tier, completed_count, len(batch))

        return total_queries_used

    def _evaluate_single_with_shared(
        self,
        genome: Genome,
        queries: List[str],
        ground_truth: List[List[Any]],
        shared_context: SharedPrecomputeContext
    ) -> Tuple[int, str]:
        """
        Evaluate a single genome using shared pre-computed context.

        Uses pre-computed query embeddings and initial pools when available.
        Supports single-checkpoint early exit at quarter point (25%).
        """
        if not self.enable_adaptive:
            return self._evaluate_single_full_with_shared(
                genome, queries, ground_truth, shared_context
            )

        retriever_kwargs = self._get_compiler_for_genome(genome).compile(genome)
        pool_size = retriever_kwargs.get('initial_pool_size', 30)
        decision_tracker = self._create_decision_tracker()

        start_time = time.time()
        n_queries = len(queries)
        quarter = max(1, n_queries // 4)  # Ensure at least 1 query for early exit

        # Get pre-computed initial pools
        initial_pools = shared_context.initial_pools.get(pool_size, [])

        # Phase 1: Evaluate first quarter for early exit check
        if initial_pools and hasattr(self.retriever, 'retrieve_batch_with_precomputed'):
            first_quarter_results = self.retriever.retrieve_batch_with_precomputed(
                query_embeddings=shared_context.query_embeddings[:quarter],
                initial_pools=initial_pools[:quarter],
                max_workers=1 if self.run_mode == "sequential" else self.max_workers_per_retrieval,
                gpu_batch_size=self.run_batch_size,
                genome_id=f"{genome.id}_quarter",
                **retriever_kwargs
            )
        else:
            first_quarter_results = self.retriever.retrieve_batch(
                queries=queries[:quarter],
                max_workers=self.max_workers_per_retrieval,
                genome_id=f"{genome.id}_quarter",
                **retriever_kwargs
            )

        # Compute metrics at quarter and check early exit
        quarter_metrics = self._compute_metrics_cumulative(
            first_quarter_results, ground_truth[:quarter]
        )
        elapsed = time.time() - start_time
        quarter_metrics['latency'] = elapsed / max(1, quarter)
        quarter_metrics['complexity'] = float(genome.complexity())

        quarter_fitness = self.fitness_calc.calculate(quarter_metrics, genome)

        if quarter_fitness.quality_score < self.early_exit_threshold:
            # Early exit - this genome isn't promising
            genome.metrics = quarter_metrics
            genome.fitness = quarter_fitness
            genome.evaluated = True

            # Cache the fitness for future lookups
            self._fitness_cache.put(genome, genome.fitness.quality_score)

            if decision_tracker is not None:
                genome.decision_context = decision_tracker.to_summary_dict()

            logger.debug(
                f"  > [Early Exit] {genome.id} at quarter "
                f"(qual={quarter_fitness.quality_score:.4f} < {self.early_exit_threshold})"
            )
            return quarter, "early_exit"

        # Phase 2: Evaluate remaining 3/4 for full evaluation
        if initial_pools and hasattr(self.retriever, 'retrieve_batch_with_precomputed'):
            remaining_results = self.retriever.retrieve_batch_with_precomputed(
                query_embeddings=shared_context.query_embeddings[quarter:],
                initial_pools=initial_pools[quarter:],
                max_workers=1 if self.run_mode == "sequential" else self.max_workers_per_retrieval,
                gpu_batch_size=self.run_batch_size,
                genome_id=f"{genome.id}_full",
                **retriever_kwargs
            )
        else:
            remaining_results = self.retriever.retrieve_batch(
                queries=queries[quarter:],
                max_workers=self.max_workers_per_retrieval,
                genome_id=f"{genome.id}_full",
                **retriever_kwargs
            )

        all_results = first_quarter_results + remaining_results

        # Full evaluation completed
        total_latency = time.time() - start_time
        final_metrics = self._compute_metrics_cumulative(all_results, ground_truth)
        final_metrics['latency'] = total_latency / max(1, n_queries)
        final_metrics['complexity'] = float(genome.complexity())

        genome.metrics = final_metrics
        genome.fitness = self.fitness_calc.calculate(final_metrics, genome)
        genome.evaluated = True

        # Cache the fitness for future lookups
        self._fitness_cache.put(genome, genome.fitness.quality_score)

        if decision_tracker is not None:
            genome.decision_context = decision_tracker.to_summary_dict()

        return n_queries, "full"

    def _evaluate_single_full_with_shared(
        self,
        genome: Genome,
        queries: List[str],
        ground_truth: List[List[Any]],
        shared_context: SharedPrecomputeContext
    ) -> Tuple[int, str]:
        """Full evaluation using shared pre-computed context."""
        retriever_kwargs = self._get_compiler_for_genome(genome).compile(genome)
        pool_size = retriever_kwargs.get('initial_pool_size', 30)

        # Wrap evaluation with memory guard
        with MemoryGuard(label=f"eval_full_{genome.id}", cleanup_on_exit=True):
            with torch.no_grad():  # Belt-and-suspenders gradient prevention
                decision_tracker = self._create_decision_tracker()

                start_time = time.time()

                initial_pools = shared_context.initial_pools.get(pool_size, [])

                if initial_pools and hasattr(self.retriever, 'retrieve_batch_with_precomputed'):
                    results = self.retriever.retrieve_batch_with_precomputed(
                        query_embeddings=shared_context.query_embeddings,
                        initial_pools=initial_pools,
                        max_workers=1 if self.run_mode == "sequential" else self.max_workers_per_retrieval,
                        gpu_batch_size=self.run_batch_size,
                        genome_id=genome.id,
                        **retriever_kwargs
                    )
                else:
                    # Fallback
                    results = self.retriever.retrieve_batch(
                        queries=queries,
                        max_workers=1 if self.run_mode == "sequential" else self.max_workers_per_retrieval,
                        genome_id=genome.id,
                        **retriever_kwargs
                    )

                total_latency = time.time() - start_time

                metrics = self._compute_metrics_cumulative(results, ground_truth)
                metrics['latency'] = total_latency / max(1, len(queries))
                metrics['complexity'] = float(genome.complexity())

                genome.metrics = metrics
                genome.fitness = self.fitness_calc.calculate(metrics, genome)
                genome.evaluated = True

                # Cache the fitness for future lookups
                self._fitness_cache.put(genome, genome.fitness.quality_score)

                if decision_tracker is not None:
                    genome.decision_context = decision_tracker.to_summary_dict()

                return len(queries), "full"

    def _batch_compute_metrics_all_genomes(
        self,
        genomes: List[Genome],
        batched_results: BatchedRetrievalResults,
        ground_truth_sets: List[Set[Any]],
        shared_context: Optional[SharedPrecomputeContext] = None
    ):
        """
        Compute metrics for all genomes in a single batched GPU call.

        This provides significant speedup by:
        1. Stacking all retrieval results into a single tensor
        2. Using precomputed ground truth GPU tensors when available
        3. Single call to compute_all_metrics_batch_gpu_precomputed
        4. Reshaping and assigning back to individual genomes
        """
        from ...eval.metric_functions import MetricFunctions

        logger.info("  > Computing metrics across all genomes in batch...")

        # Prepare flattened tensor for batch computation directly on target device
        max_k = 20  # Standard max k for metrics
        target_device = self.device
        retrieved_ids, genome_query_indices = batched_results.prepare_for_batch_metrics(
            max_k, device=target_device
        )

        if retrieved_ids.numel() == 0:
            logger.warning("No retrieval results to compute metrics for")
            return

        n_total = len(genome_query_indices)
        n_queries_per_genome = len(ground_truth_sets)

        # Expand ground truth for all genomes
        n_genomes = len(genomes)

        # Check if we have precomputed GPU ground truth tensors
        has_precomputed_gt = (
            shared_context is not None and
            shared_context.ground_truth_tensor is not None and
            shared_context.gt_sizes is not None
        )

        # Compute metrics in batch (retrieved_ids already on target device)
        if self.device != "cpu":
            try:
                if has_precomputed_gt:
                    # Use fastest path with precomputed GPU tensors
                    # Expand GT tensors for all genomes
                    gt_tensor_expanded = shared_context.ground_truth_tensor.repeat(n_genomes, 1)
                    gt_sizes_expanded = shared_context.gt_sizes.repeat(n_genomes)

                    # Truncate to n_total (in case of partial batches)
                    gt_tensor_expanded = gt_tensor_expanded[:n_total]
                    gt_sizes_expanded = gt_sizes_expanded[:n_total]

                    logger.debug(f"    > Using precomputed GT tensor: {gt_tensor_expanded.shape}")

                    all_metrics_per_query = MetricFunctions.compute_all_metrics_batch_gpu_precomputed(
                        retrieved_ids,
                        gt_tensor_expanded,
                        gt_sizes_expanded,
                        k_values=self.evaluator.k_values,
                        device="cuda"
                    )
                else:
                    # Fallback to sets-based GPU computation
                    expanded_gt_sets = ground_truth_sets * n_genomes
                    if hasattr(MetricFunctions, 'compute_all_metrics_batch_gpu_vectorized'):
                        all_metrics_per_query = MetricFunctions.compute_all_metrics_batch_gpu_vectorized(
                            retrieved_ids,
                            expanded_gt_sets[:n_total],
                            k_values=self.evaluator.k_values,
                            device="cuda"
                        )
                    else:
                        all_metrics_per_query = MetricFunctions.compute_all_metrics_batch_gpu(
                            retrieved_ids,
                            expanded_gt_sets[:n_total],
                            k_values=self.evaluator.k_values,
                            device="cuda"
                        )
            except Exception as e:
                # Fail explicitly - GPU/CPU overhead makes silent fallback inefficient
                logger.error(f"GPU batch metrics failed: {e}")
                raise RuntimeError(f"GPU metric computation failed: {e}") from e
        else:
            expanded_gt_sets = ground_truth_sets * n_genomes
            all_metrics_per_query = MetricFunctions.compute_all_metrics_batch(
                retrieved_ids,
                expanded_gt_sets[:n_total],
                k_values=self.evaluator.k_values
            )

        # Cleanup: release expanded GT tensors to free GPU memory
        if has_precomputed_gt and self.device != "cpu":
            try:
                del gt_tensor_expanded
                del gt_sizes_expanded
            except NameError:
                pass  # Variables not created if exception occurred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Assign metrics back to individual genomes
        genome_id_to_genome = {g.id: g for g in genomes}

        # Group indices by genome
        genome_indices: Dict[str, List[int]] = {}
        for idx, (genome_id, query_idx) in enumerate(genome_query_indices):
            if genome_id not in genome_indices:
                genome_indices[genome_id] = []
            genome_indices[genome_id].append(idx)

        # Aggregate metrics per genome
        for genome_id, indices in genome_indices.items():
            genome = genome_id_to_genome.get(genome_id)
            if genome is None:
                continue

            # Average metrics across queries for this genome
            aggregated = {}
            for metric_name in all_metrics_per_query.keys():
                if metric_name.startswith('per_query_'):
                    # Per-query scores - average them
                    base_name = metric_name.replace('per_query_', '')
                    scores = all_metrics_per_query[metric_name]
                    if isinstance(scores, torch.Tensor):
                        genome_scores = scores[indices]
                        aggregated[base_name] = float(genome_scores.mean().item())
                        aggregated[f"var_{base_name}"] = float(genome_scores.var().item()) if len(indices) > 1 else 0.0
                else:
                    # Already aggregated
                    aggregated[metric_name] = all_metrics_per_query[metric_name]

            # If metrics are already averaged (not per-query), use as-is
            if not aggregated:
                aggregated = dict(all_metrics_per_query)

            # Add latency and complexity
            retrieval_time = getattr(genome, '_retrieval_time', 0.0)
            aggregated['latency'] = retrieval_time / max(1, len(indices))
            aggregated['complexity'] = float(genome.complexity())

            # Add variance for main metric
            priority_keys = ["Recall@10", "Hit@10", "MRR", "Recall@5", "Hit@5"]
            main_key = next((k for k in priority_keys if f"var_{k}" in aggregated), None)
            if main_key:
                aggregated["variance"] = aggregated[f"var_{main_key}"]

            genome.metrics = aggregated
            genome.fitness = self.fitness_calc.calculate(aggregated, genome)
            genome.evaluated = True

            # Cache the fitness for future lookups
            self._fitness_cache.put(genome, genome.fitness.quality_score)

            self.stats.tier_exits["full"] = self.stats.tier_exits.get("full", 0) + 1

        logger.info(f"  > Batch metrics computed for {len(genomes)} genomes")

    def _log_genome_result(
        self,
        genome: Genome,
        exit_tier: str,
        completed_count: int,
        batch_size: int
    ):
        """Log evaluation result for a genome."""
        qual = genome.fitness.quality_score
        stab = genome.fitness.stability_score
        h1 = genome.metrics.get("Hit@1", 0.0)
        h5 = genome.metrics.get("Hit@5", 0.0)
        mrr = genome.metrics.get("MRR", 0.0)
        r20 = genome.metrics.get("Recall@20", 0.0)
        latency_ms = genome.metrics.get("latency", 0.0) * 1000  # Convert to ms

        logger.info(
            f"  > Finished '{genome.id}' ({completed_count}/{batch_size}) | "
            f"Tier: {exit_tier} | Lat: {latency_ms:.1f}ms | Q: {qual:.4f} | "
            f"H@1: {h1:.2f} | H@5: {h5:.2f} | MRR: {mrr:.2f} | R@20: {r20:.2f}"
        )

    def _evaluate_batch(
        self,
        batch: List[Genome],
        queries: List[str],
        ground_truth: List[List[Any]],
    ) -> int:
        """
        Runs a batch of evaluations.

        GPU mode: Sequential evaluation (CUDA context is thread-local)
        CPU mode: Parallel evaluation with ThreadPoolExecutor

        Returns:
            Total number of queries used across all genomes in batch
        """
        logger.debug(f"  > Starting batch of {len(batch)} genomes (device={self.device})...")

        total_queries_used = 0
        completed_count = 0

        # GPU mode: sequential evaluation (GPU context is thread-local)
        if self.device != "cpu":
            for genome in batch:
                queries_used, exit_tier = self._evaluate_single(genome, queries, ground_truth)
                total_queries_used += queries_used
                completed_count += 1

                # Update tier exit stats
                if exit_tier in self.stats.tier_exits:
                    self.stats.tier_exits[exit_tier] += 1

                self._log_genome_result(genome, exit_tier, completed_count, len(batch))

            return total_queries_used

        # CPU mode: parallel evaluation with ThreadPoolExecutor
        # Cap workers at concurrent_evaluations to prevent over-subscription
        with ThreadPoolExecutor(max_workers=min(len(batch), self.concurrent_evaluations)) as executor:
            future_to_genome = {
                executor.submit(self._evaluate_single, g, queries, ground_truth): g
                for g in batch
            }

            for future in as_completed(future_to_genome):
                genome = future_to_genome[future]
                completed_count += 1

                queries_used, exit_tier = future.result()
                total_queries_used += queries_used

                # Update tier exit stats
                if exit_tier in self.stats.tier_exits:
                    self.stats.tier_exits[exit_tier] += 1

                self._log_genome_result(genome, exit_tier, completed_count, len(batch))

        return total_queries_used

    def _create_decision_tracker(self) -> Optional[Any]:
        """Create a DecisionTracker if decision tracking is enabled."""
        if not self.track_decisions:
            return None
        try:
            from ..llm.decision_tracker import DecisionTracker
            return DecisionTracker(
                enabled=True,
                sample_rate=self.decision_sample_rate
            )
        except ImportError:
            logger.warning("DecisionTracker not available, disabling decision tracking")
            return None

    def _evaluate_single(
        self,
        genome: Genome,
        queries: List[str],
        ground_truth: List[List[Any]]
    ) -> Tuple[int, str]:
        """
        Evaluates a single genome with single quarter checkpoint.

        Returns:
            Tuple of (queries_used, exit_tier_name)
        """
        if not self.enable_adaptive:
            return self._evaluate_single_full(genome, queries, ground_truth)

        retriever_kwargs = self._get_compiler_for_genome(genome).compile(genome)

        # Wrap evaluation with memory guard
        with MemoryGuard(label=f"eval_{genome.id}", cleanup_on_exit=True):
            with torch.no_grad():  # Belt-and-suspenders gradient prevention
                decision_tracker = self._create_decision_tracker()

                start_time = time.time()
                n_queries = len(queries)
                quarter = max(1, n_queries // 4)  # Ensure at least 1 query for early exit

                # Phase 1: Evaluate first quarter for early exit check
                if decision_tracker is not None:
                    # Use single-query mode for decision tracking on first batch
                    first_quarter_results = []
                    for q in queries[:quarter]:
                        res = self.retriever.retrieve(
                            query=q,
                            decision_tracker=decision_tracker,
                            **retriever_kwargs
                        )
                        first_quarter_results.append(res)
                else:
                    # Use batch mode for speed
                    first_quarter_results = self.retriever.retrieve_batch(
                        queries=queries[:quarter],
                        max_workers=self.max_workers_per_retrieval,
                        genome_id=f"{genome.id}_quarter",
                        **retriever_kwargs
                    )

                # Compute metrics at quarter and check early exit
                quarter_metrics = self._compute_metrics_cumulative(
                    first_quarter_results, ground_truth[:quarter]
                )
                elapsed = time.time() - start_time
                quarter_metrics['latency'] = elapsed / max(1, quarter)
                quarter_metrics['complexity'] = float(genome.complexity())

                quarter_fitness = self.fitness_calc.calculate(quarter_metrics, genome)

                if quarter_fitness.quality_score < self.early_exit_threshold:
                    # Early exit - this genome isn't promising
                    genome.metrics = quarter_metrics
                    genome.fitness = quarter_fitness
                    genome.evaluated = True

                    # Cache the fitness for future lookups
                    self._fitness_cache.put(genome, genome.fitness.quality_score)

                    if decision_tracker is not None:
                        genome.decision_context = decision_tracker.to_summary_dict()

                    logger.debug(
                        f"  > [Early Exit] {genome.id} at quarter "
                        f"(qual={quarter_fitness.quality_score:.4f} < {self.early_exit_threshold})"
                    )
                    return quarter, "early_exit"

                # Phase 2: Evaluate remaining 3/4 for full evaluation
                remaining_results = self.retriever.retrieve_batch(
                    queries=queries[quarter:],
                    max_workers=self.max_workers_per_retrieval,
                    genome_id=f"{genome.id}_full",
                    **retriever_kwargs
                )

                all_results = first_quarter_results + remaining_results

                # Full evaluation completed
                total_latency = time.time() - start_time
                final_metrics = self._compute_metrics_cumulative(all_results, ground_truth)
                final_metrics['latency'] = total_latency / max(1, n_queries)
                final_metrics['complexity'] = float(genome.complexity())

                genome.metrics = final_metrics
                genome.fitness = self.fitness_calc.calculate(final_metrics, genome)
                genome.evaluated = True

                # Cache the fitness for future lookups
                self._fitness_cache.put(genome, genome.fitness.quality_score)

                if decision_tracker is not None:
                    genome.decision_context = decision_tracker.to_summary_dict()

                return n_queries, "full"

    def _evaluate_single_full(
        self,
        genome: Genome,
        queries: List[str],
        ground_truth: List[List[Any]]
    ) -> Tuple[int, str]:
        """Full evaluation without adaptive sampling (fallback)."""
        retriever_kwargs = self._get_compiler_for_genome(genome).compile(genome)

        # Wrap evaluation with memory guard
        with MemoryGuard(label=f"eval_full_{genome.id}", cleanup_on_exit=True):
            with torch.no_grad():  # Belt-and-suspenders gradient prevention
                decision_tracker = self._create_decision_tracker()

                start_time = time.time()

                if decision_tracker is not None:
                    # Use single-query mode for decision tracking on subset
                    probe_size = min(20, len(queries))
                    results = []
                    for q in queries[:probe_size]:
                        res = self.retriever.retrieve(
                            query=q,
                            decision_tracker=decision_tracker,
                            **retriever_kwargs
                        )
                        results.append(res)

                    # Batch the rest
                    if len(queries) > probe_size:
                        batch_results = self.retriever.retrieve_batch(
                            queries=queries[probe_size:],
                            max_workers=self.max_workers_per_retrieval,
                            genome_id=genome.id,
                            **retriever_kwargs
                        )
                        results.extend(batch_results)
                else:
                    results = self.retriever.retrieve_batch(
                        queries=queries,
                        max_workers=self.max_workers_per_retrieval,
                        genome_id=genome.id,
                        **retriever_kwargs
                    )

                total_latency = time.time() - start_time

                metrics = self._compute_metrics_cumulative(results, ground_truth)
                metrics['latency'] = total_latency / max(1, len(queries))
                metrics['complexity'] = float(genome.complexity())

                genome.metrics = metrics
                genome.fitness = self.fitness_calc.calculate(metrics, genome)
                genome.evaluated = True

                # Cache the fitness for future lookups
                self._fitness_cache.put(genome, genome.fitness.quality_score)

                if decision_tracker is not None:
                    genome.decision_context = decision_tracker.to_summary_dict()

                return len(queries), "full"

    def _compute_metrics_cumulative(
        self,
        results: List[List[Any]],
        ground_truth: List[List[Any]]
    ) -> Dict[str, float]:
        """Compute aggregated metrics for results so far using batch operations."""
        n_queries = min(len(results), len(ground_truth))
        if n_queries == 0:
            return {}

        # Use sequential for small batches (respects evaluator interface)
        # This ensures mock evaluators work correctly in tests
        BATCH_THRESHOLD = 50  # Switch to batch mode above this

        if n_queries <= BATCH_THRESHOLD:
            return self._compute_metrics_sequential(results, ground_truth)

        # Try batch computation for large batches (much faster)
        try:
            return self._compute_metrics_batch(results[:n_queries], ground_truth[:n_queries])
        except Exception as e:
            logger.debug(f"Batch metrics failed, falling back to sequential: {e}")
            return self._compute_metrics_sequential(results, ground_truth)

    def _compute_metrics_batch(
        self,
        results: List[List[Any]],
        ground_truth: List[List[Any]]
    ) -> Dict[str, float]:
        """Compute metrics using batch operations for better performance."""
        from ...eval.metric_functions import MetricFunctions

        n_queries = len(results)
        if n_queries == 0:
            return {}

        # Determine max retrieved items
        max_retrieved = max((len(r) for r in results), default=0)
        if max_retrieved == 0:
            return {}

        # Extract IDs as tensor on target device, padding with -1 for missing values
        target_device = self.device
        retrieved_ids = torch.full(
            (n_queries, max_retrieved), -1, dtype=torch.long, device=target_device
        )

        for i, items in enumerate(results):
            for j, item in enumerate(items):
                if isinstance(item, dict):
                    retrieved_ids[i, j] = int(item.get('id', -1))
                else:
                    try:
                        retrieved_ids[i, j] = int(item)
                    except (ValueError, TypeError):
                        pass

        # Convert ground truth to sets of integers
        gt_sets = []
        for gt in ground_truth:
            try:
                gt_sets.append(set(int(g) for g in gt))
            except (ValueError, TypeError):
                gt_sets.append(set(str(g) for g in gt))

        # Use GPU-accelerated batch metric computation if on accelerated device
        if self.device != "cpu":
            try:
                metrics = MetricFunctions.compute_all_metrics_batch_gpu(
                    retrieved_ids,
                    gt_sets,
                    k_values=self.evaluator.k_values,
                    device="cuda",
                    return_per_query=True
                )
            except Exception as e:
                # Fail explicitly - GPU/CPU overhead makes silent fallback inefficient
                logger.error(f"GPU metrics failed: {e}")
                raise RuntimeError(f"GPU metric computation failed: {e}") from e
        else:
            # CPU batch computation
            metrics = MetricFunctions.compute_all_metrics_batch(
                retrieved_ids,
                gt_sets,
                k_values=self.evaluator.k_values
            )
            # Fall back to zero variance for CPU path
            for key in list(metrics.keys()):
                if not key.startswith("per_query_") and not key.startswith("var_"):
                    metrics[f"var_{key}"] = 0.0

        # Compute variance from per-query scores if available
        per_query_keys = [k for k in metrics.keys() if k.startswith("per_query_")]
        for pq_key in per_query_keys:
            base_name = pq_key.replace("per_query_", "")
            scores_tensor = metrics[pq_key]
            if isinstance(scores_tensor, torch.Tensor) and len(scores_tensor) > 1:
                metrics[f"var_{base_name}"] = float(scores_tensor.var().item())
            else:
                metrics[f"var_{base_name}"] = 0.0
            # Remove per-query tensors from final metrics
            del metrics[pq_key]

        # Set main variance from priority metric
        priority_keys = ["Recall@10", "Hit@10", "MRR", "Recall@5", "Hit@5"]
        main_key = next((k for k in priority_keys if f"var_{k}" in metrics), None)
        if main_key:
            metrics["variance"] = metrics[f"var_{main_key}"]
        else:
            metrics["variance"] = 0.0

        return metrics

    def _compute_metrics_sequential(
        self,
        results: List[List[Any]],
        ground_truth: List[List[Any]]
    ) -> Dict[str, float]:
        """Fallback sequential metric computation."""
        all_metrics = []
        for i, retrieved_items in enumerate(results):
            if i >= len(ground_truth):
                break
            m = self.evaluator.calculate_metrics(
                retrieved_nodes=retrieved_items,
                ground_truth_ids=ground_truth[i],
                latency_sec=0
            )
            all_metrics.append(m)

        return self._mean_metrics(all_metrics)

    def _mean_metrics(self, all_metrics: List[Dict]) -> Dict[str, float]:
        """Compute mean and variance of metrics."""
        if not all_metrics:
            return {}

        keys = all_metrics[0].keys()
        aggregated = {}

        for k in keys:
            values = [m[k] for m in all_metrics]
            # Skip non-numeric values
            if not values or not isinstance(values[0], (int, float)):
                continue
            t = torch.as_tensor(values, dtype=torch.float32)
            aggregated[k] = float(torch.mean(t).item())
            # Use population variance (correction=0) for consistency
            aggregated[f"var_{k}"] = float(torch.var(t, correction=0).item()) if len(t) > 1 else 0.0

        # Select variance for main metric
        priority_keys = [
            "Recall@10", "Hit@10", "MRR",
            "Recall@5", "Hit@5",
            "Recall@1", "Hit@1",
            "Recall@20", "Hit@20"
        ]

        main_key = next((k for k in priority_keys if k in keys), None)
        if main_key:
            aggregated["variance"] = aggregated[f"var_{main_key}"]
        else:
            fallback = next((k for k in keys if "Recall" in k or "Hit" in k), None)
            aggregated["variance"] = aggregated.get(f"var_{fallback}", 0.0) if fallback else 0.0

        return aggregated


    def _evaluate_single_with_early_exit(
        self,
        genome: Genome,
        queries: List[str],
        ground_truth: List[List[Any]],
        shared_context: SharedPrecomputeContext
    ) -> Tuple[int, str]:
        """
        Evaluate a genome with single quarter checkpoint using GPU-optimized metrics.

        This optimized path:
        1. Retrieves all queries up front
        2. Computes metrics at quarter and full in GPU calls
        3. Makes early-exit decision based on single threshold

        Args:
            genome: Genome to evaluate
            queries: List of query strings
            ground_truth: Ground truth lists
            shared_context: Precomputed shared context

        Returns:
            Tuple of (queries_used, exit_tier_name)
        """
        retriever_kwargs = self._get_compiler_for_genome(genome).compile(genome)
        pool_size = retriever_kwargs.get('initial_pool_size', 30)

        # Wrap evaluation with memory guard
        with MemoryGuard(label=f"eval_{genome.id}", cleanup_on_exit=True):
            with torch.no_grad():  # Belt-and-suspenders gradient prevention
                decision_tracker = self._create_decision_tracker()

                start_time = time.time()
                n_queries = len(queries)
                quarter = max(1, n_queries // 4)  # Ensure at least 1 query for early exit

                # Get pre-computed initial pools
                initial_pools = shared_context.initial_pools.get(pool_size, [])
                if not initial_pools or not hasattr(self.retriever, 'retrieve_batch_with_precomputed'):
                    # Fallback to non-optimized path
                    return self._evaluate_single_with_shared(
                        genome, queries, ground_truth, shared_context
                    )

                # Fetch all results in one batch (retrieval is the expensive part)
                all_results = self.retriever.retrieve_batch_with_precomputed(
                    query_embeddings=shared_context.query_embeddings,
                    initial_pools=initial_pools,
                    max_workers=1 if self.run_mode == "sequential" else self.max_workers_per_retrieval,
                    gpu_batch_size=self.run_batch_size,
                    genome_id=genome.id,
                    **retriever_kwargs
                )

                # Build retrieved IDs tensor on GPU
                max_k = 20
                retrieved_ids = torch.full(
                    (n_queries, max_k), -1, dtype=torch.long, device=self.device
                )
                for i, results in enumerate(all_results):
                    for j, item in enumerate(results[:max_k]):
                        if isinstance(item, dict):
                            try:
                                retrieved_ids[i, j] = int(item.get('id', -1))
                            except (ValueError, TypeError):
                                pass
                        else:
                            try:
                                retrieved_ids[i, j] = int(item)
                            except (ValueError, TypeError):
                                pass

                # Compute metrics at quarter for early exit check
                quarter_metrics = self._compute_metrics_for_slice(
                    retrieved_ids[:quarter], shared_context, quarter
                )

                elapsed = time.time() - start_time
                quarter_metrics['latency'] = elapsed / max(1, quarter)
                quarter_metrics['complexity'] = float(genome.complexity())

                quarter_fitness = self.fitness_calc.calculate(quarter_metrics, genome)

                if quarter_fitness.quality_score < self.early_exit_threshold:
                    # Early exit - this genome isn't promising
                    genome.metrics = quarter_metrics
                    genome.fitness = quarter_fitness
                    genome.evaluated = True

                    # Cache the fitness for future lookups
                    self._fitness_cache.put(genome, genome.fitness.quality_score)

                    if decision_tracker is not None:
                        genome.decision_context = decision_tracker.to_summary_dict()

                    logger.debug(
                        f"  > [Early Exit] {genome.id} at quarter "
                        f"(qual={quarter_fitness.quality_score:.4f} < {self.early_exit_threshold})"
                    )
                    return quarter, "early_exit"

                # Full evaluation - compute metrics for all queries
                total_latency = time.time() - start_time
                final_metrics = self._compute_metrics_for_slice(
                    retrieved_ids, shared_context, n_queries
                )
                final_metrics['latency'] = total_latency / max(1, n_queries)
                final_metrics['complexity'] = float(genome.complexity())

                genome.metrics = final_metrics
                genome.fitness = self.fitness_calc.calculate(final_metrics, genome)
                genome.evaluated = True

                # Cache the fitness for future lookups
                self._fitness_cache.put(genome, genome.fitness.quality_score)

                if decision_tracker is not None:
                    genome.decision_context = decision_tracker.to_summary_dict()

                return n_queries, "full"

    def _compute_metrics_for_slice(
        self,
        retrieved_ids: torch.Tensor,
        shared_context: SharedPrecomputeContext,
        n_queries: int
    ) -> Dict[str, float]:
        """
        Compute metrics for a slice of retrieved IDs using GPU-accelerated computation.

        Args:
            retrieved_ids: (n_queries, max_k) tensor of retrieved IDs on GPU
            shared_context: Shared context with precomputed ground truth tensors
            n_queries: Number of queries in this slice

        Returns:
            Dictionary of metric name -> value
        """
        from ...eval.metric_functions import MetricFunctions

        # Use precomputed GT tensors
        gt_tensor = shared_context.ground_truth_tensor[:n_queries]
        gt_sizes = shared_context.gt_sizes[:n_queries]

        try:
            metrics = MetricFunctions.compute_all_metrics_batch_gpu_precomputed(
                retrieved_ids,
                gt_tensor,
                gt_sizes,
                k_values=self.evaluator.k_values,
                device=self.device
            )
            return metrics
        except Exception as e:
            # Fail explicitly - GPU/CPU overhead makes silent fallback inefficient
            logger.error(f"GPU metrics failed: {e}")
            raise RuntimeError(f"GPU metric computation failed: {e}") from e

    def cleanup(self):
        """
        Release resources at evolution end.

        Clears embedding cache to release GPU memory.
        Should be called by EvolutionEngine or Orchestrator when evolution completes.
        """
        # Clear embedding cache to release GPU memory
        EmbeddingCacheProvider.clear()
        logger.info("PopulationEvaluator cleanup: embedding cache cleared")
