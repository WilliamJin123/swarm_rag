"""
Population Evaluator with Adaptive Multi-Tier Evaluation.

Implements progressive evaluation to reduce wasted computation on poor-performing
genomes. Most bad genomes are filtered at early tiers with small sample sizes,
while promising genomes get full evaluation.

Optimization features:
- Shared pre-computation: Query embeddings and initial pools computed once per generation
- Cross-genome metric batching: Single GPU call for metrics across all genomes
- CPU thread pool capping: Prevents worker over-subscription
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
import torch
from typing import List, Any, Optional, Dict, Tuple, Set
from dataclasses import dataclass, field

from swarm_rag.interfaces.protocols import RetrievalBackend
from ...eval.metrics import Evaluator
from .fitness import FitnessCalculator
from ..types.genome import GenomeCompiler, Genome
from .shared_precompute import (
    SharedPrecomputeContext,
    prepare_shared_context,
    get_unique_pool_sizes,
    BatchedRetrievalResults
)


logger = logging.getLogger(__name__)


@dataclass
class EvaluationTier:
    """Configuration for a single evaluation tier."""
    queries: int
    threshold: Optional[float]  # None = full evaluation (no early exit)
    name: str = ""


# Default evaluation tiers - progressively filter out bad genomes
# Final tier uses a large number to ensure all available queries are used
DEFAULT_TIERS: List[EvaluationTier] = [
    EvaluationTier(queries=5, threshold=0.10, name="quick_filter"),
    EvaluationTier(queries=15, threshold=0.25, name="promising"),
    EvaluationTier(queries=40, threshold=0.50, name="competitive"),
    EvaluationTier(queries=100_000, threshold=None, name="full"),  # Use all available
]


@dataclass
class EvaluationStats:
    """Statistics about evaluation efficiency."""
    total_genomes: int = 0
    tier_exits: Dict[str, int] = field(default_factory=dict)
    avg_queries_per_genome: float = 0.0
    time_saved_estimate: float = 0.0


class PopulationEvaluator:
    """
    Evaluator with multi-tier progressive sampling.

    Implements adaptive evaluation that progressively tests genomes with
    increasing sample sizes, exiting early for poor performers.

    This dramatically reduces evaluation time by:
    1. Quick filter (10 queries): Eliminates clearly bad genomes
    2. Promising tier (30 queries): Confirms potential
    3. Competitive tier (60 queries): Validates competitive performance
    4. Full evaluation (100 queries): Only for top candidates

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
        tiers: Custom evaluation tiers (default: DEFAULT_TIERS)
        enable_adaptive: Whether to use adaptive evaluation (True) or full evaluation only
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
        tiers: List[EvaluationTier] = None,
        enable_adaptive: bool = True,
        device: Any = None,  # torch.device or str
        enable_shared_precompute: bool = True,
        enable_cross_genome_metric_batch: bool = True,
    ):
        self.retriever = retriever
        self.evaluator = evaluator
        self.fitness_calc = fitness_calc
        self.queries = queries or []
        self.ground_truth = ground_truth or []
        self.compiler = GenomeCompiler()
        self.concurrent_evaluations = concurrent_evaluations
        self.max_workers_per_retrieval = max_workers_per_retrieval
        self.track_decisions = track_decisions
        self.decision_sample_rate = decision_sample_rate
        self.tiers = tiers or DEFAULT_TIERS
        self.enable_adaptive = enable_adaptive

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

    def _reset_stats(self):
        """Reset evaluation statistics for a new run."""
        self.stats = EvaluationStats(
            tier_exits={tier.name: 0 for tier in self.tiers}
        )

    def evaluate(
        self,
        population: List[Genome],
        queries: List[str] = None,
        ground_truth: List[List[Any]] = None
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

        self.stats.total_genomes = len(unevaluated)
        batch_size = self.concurrent_evaluations

        logger.info(f"Evaluating {len(unevaluated)} genomes...")
        logger.info(f"  > Concurrency: {batch_size} | Adaptive: {self.enable_adaptive}")
        logger.info(f"  > Shared precompute: {self.enable_shared_precompute} | Cross-genome batch: {self.enable_cross_genome_metric_batch}")
        if self.enable_adaptive:
            logger.info(f"  > Tiers: {[(t.queries, t.threshold) for t in self.tiers]}")

        # Prepare shared context if enabled
        shared_context = None
        if self.enable_shared_precompute and hasattr(self.retriever, 'retrieve_batch_with_precomputed'):
            shared_context = self._prepare_shared_context(unevaluated, queries, ground_truth)
            self._shared_context = shared_context

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

        # Compute stats
        max_queries = len(queries) * len(unevaluated)
        self.stats.avg_queries_per_genome = total_queries_used / max(1, len(unevaluated))
        self.stats.time_saved_estimate = 1.0 - (total_queries_used / max(1, max_queries))

        logger.info(f"Evaluation complete:")
        logger.info(f"  > Avg queries/genome: {self.stats.avg_queries_per_genome:.1f} / {len(queries)}")
        logger.info(f"  > Time saved estimate: {self.stats.time_saved_estimate:.1%}")
        if self.enable_adaptive:
            logger.info(f"  > Tier exits: {self.stats.tier_exits}")

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
            retriever_kwargs = self.compiler.compile(genome)
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
                max_workers=self.max_workers_per_retrieval,
                genome_id=genome.id,
                **retriever_kwargs
            )

            batched_results.add_genome_results(genome.id, results)
            total_queries_used += n_queries

            # Store retrieval time for later
            genome._retrieval_time = time.time() - start_time

            logger.debug(f"  > Genome '{genome.id}' ({genome_idx + 1}/{n_genomes}) retrieval complete")

        # Batch compute metrics across all genomes
        if self.enable_cross_genome_metric_batch:
            self._batch_compute_metrics_all_genomes(
                genomes, batched_results, shared_context.ground_truth_sets
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
        for faster evaluation, while still supporting adaptive tiers.
        """
        logger.debug(f"  > Starting batch of {len(batch)} genomes with shared context...")

        total_queries_used = 0
        completed_count = 0

        # GPU mode: sequential evaluation
        if self.device == "cuda":
            for genome in batch:
                queries_used, exit_tier = self._evaluate_single_with_shared(
                    genome, queries, ground_truth, shared_context
                )
                total_queries_used += queries_used
                completed_count += 1

                if exit_tier in self.stats.tier_exits:
                    self.stats.tier_exits[exit_tier] += 1

                self._log_genome_result(genome, exit_tier, completed_count, len(batch))

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
        """
        if not self.enable_adaptive:
            return self._evaluate_single_full_with_shared(
                genome, queries, ground_truth, shared_context
            )

        retriever_kwargs = self.compiler.compile(genome)
        pool_size = retriever_kwargs.get('initial_pool_size', 30)
        decision_tracker = self._create_decision_tracker()

        start_time = time.time()
        all_results = []
        queries_evaluated = 0
        exit_tier = "full"

        for tier in self.tiers:
            tier_end = min(tier.queries, len(queries))
            tier_start = queries_evaluated

            if tier_start >= tier_end:
                continue

            # Get pre-computed data for this tier
            tier_embeddings = shared_context.query_embeddings[tier_start:tier_end]
            initial_pools = shared_context.initial_pools.get(pool_size, [])
            tier_pools = initial_pools[tier_start:tier_end] if initial_pools else []

            if tier_pools and hasattr(self.retriever, 'retrieve_batch_with_precomputed'):
                # Use pre-computed path
                tier_results = self.retriever.retrieve_batch_with_precomputed(
                    query_embeddings=tier_embeddings,
                    initial_pools=tier_pools,
                    max_workers=self.max_workers_per_retrieval,
                    genome_id=f"{genome.id}_tier_{tier.name}",
                    **retriever_kwargs
                )
            else:
                # Fallback to standard batch
                tier_queries = queries[tier_start:tier_end]
                tier_results = self.retriever.retrieve_batch(
                    queries=tier_queries,
                    max_workers=self.max_workers_per_retrieval,
                    genome_id=f"{genome.id}_tier_{tier.name}",
                    **retriever_kwargs
                )

            all_results.extend(tier_results)
            queries_evaluated = tier_end

            # Compute metrics and check early exit
            current_metrics = self._compute_metrics_cumulative(
                all_results, ground_truth[:queries_evaluated]
            )

            if tier.threshold is not None:
                elapsed = time.time() - start_time
                current_metrics['latency'] = elapsed / max(1, queries_evaluated)
                current_metrics['complexity'] = float(genome.complexity())

                current_fitness = self.fitness_calc.calculate(current_metrics, genome)

                if current_fitness.quality_score < tier.threshold:
                    exit_tier = tier.name
                    genome.metrics = current_metrics
                    genome.fitness = current_fitness
                    genome.evaluated = True

                    if decision_tracker is not None:
                        genome.decision_context = decision_tracker.to_summary_dict()

                    return queries_evaluated, exit_tier

        # Full evaluation completed
        total_latency = time.time() - start_time
        final_metrics = self._compute_metrics_cumulative(all_results, ground_truth[:queries_evaluated])
        final_metrics['latency'] = total_latency / max(1, queries_evaluated)
        final_metrics['complexity'] = float(genome.complexity())

        genome.metrics = final_metrics
        genome.fitness = self.fitness_calc.calculate(final_metrics, genome)
        genome.evaluated = True

        if decision_tracker is not None:
            genome.decision_context = decision_tracker.to_summary_dict()

        return queries_evaluated, exit_tier

    def _evaluate_single_full_with_shared(
        self,
        genome: Genome,
        queries: List[str],
        ground_truth: List[List[Any]],
        shared_context: SharedPrecomputeContext
    ) -> Tuple[int, str]:
        """Full evaluation using shared pre-computed context."""
        retriever_kwargs = self.compiler.compile(genome)
        pool_size = retriever_kwargs.get('initial_pool_size', 30)
        decision_tracker = self._create_decision_tracker()

        start_time = time.time()

        initial_pools = shared_context.initial_pools.get(pool_size, [])

        if initial_pools and hasattr(self.retriever, 'retrieve_batch_with_precomputed'):
            results = self.retriever.retrieve_batch_with_precomputed(
                query_embeddings=shared_context.query_embeddings,
                initial_pools=initial_pools,
                max_workers=self.max_workers_per_retrieval,
                genome_id=genome.id,
                **retriever_kwargs
            )
        else:
            # Fallback
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

        if decision_tracker is not None:
            genome.decision_context = decision_tracker.to_summary_dict()

        return len(queries), "full"

    def _batch_compute_metrics_all_genomes(
        self,
        genomes: List[Genome],
        batched_results: BatchedRetrievalResults,
        ground_truth_sets: List[Set[Any]]
    ):
        """
        Compute metrics for all genomes in a single batched GPU call.

        This provides significant speedup by:
        1. Stacking all retrieval results into a single tensor
        2. Single call to compute_all_metrics_batch_gpu_vectorized
        3. Reshaping and assigning back to individual genomes
        """
        from ...eval.metric_functions import MetricFunctions

        logger.info("  > Computing metrics across all genomes in batch...")

        # Prepare flattened tensor for batch computation
        max_k = 20  # Standard max k for metrics
        retrieved_ids, genome_query_indices = batched_results.prepare_for_batch_metrics(max_k)

        if retrieved_ids.numel() == 0:
            logger.warning("No retrieval results to compute metrics for")
            return

        n_total = len(genome_query_indices)
        n_queries_per_genome = len(ground_truth_sets)

        # Expand ground truth sets for all genomes
        # Each genome evaluates against the same ground truth
        n_genomes = len(genomes)
        expanded_gt_sets = ground_truth_sets * n_genomes

        # Compute metrics in batch
        if self.device == "cuda":
            try:
                retrieved_ids_gpu = retrieved_ids.to(device="cuda")
                # Use vectorized GPU computation
                if hasattr(MetricFunctions, 'compute_all_metrics_batch_gpu_vectorized'):
                    all_metrics_per_query = MetricFunctions.compute_all_metrics_batch_gpu_vectorized(
                        retrieved_ids_gpu,
                        expanded_gt_sets[:n_total],
                        k_values=self.evaluator.k_values,
                        device="cuda"
                    )
                else:
                    all_metrics_per_query = MetricFunctions.compute_all_metrics_batch_gpu(
                        retrieved_ids_gpu,
                        expanded_gt_sets[:n_total],
                        k_values=self.evaluator.k_values,
                        device="cuda"
                    )
            except Exception as e:
                logger.debug(f"GPU batch metrics failed, falling back to CPU: {e}")
                all_metrics_per_query = MetricFunctions.compute_all_metrics_batch(
                    retrieved_ids,
                    expanded_gt_sets[:n_total],
                    k_values=self.evaluator.k_values
                )
        else:
            all_metrics_per_query = MetricFunctions.compute_all_metrics_batch(
                retrieved_ids,
                expanded_gt_sets[:n_total],
                k_values=self.evaluator.k_values
            )

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
        cost = genome.fitness.cost_score
        h1 = genome.metrics.get("Hit@1", 0.0)
        h5 = genome.metrics.get("Hit@5", 0.0)
        mrr = genome.metrics.get("MRR", 0.0)
        r20 = genome.metrics.get("Recall@20", 0.0)

        logger.info(
            f"  > Finished '{genome.id}' ({completed_count}/{batch_size}) | "
            f"Tier: {exit_tier} | Fitness(Q/S/C): {qual:.4f}/{stab:.4f}/{cost:.1f} | "
            f"H@1: {h1:.4f} | H@5: {h5:.4f} | MRR: {mrr:.4f} | R@20: {r20:.4f}"
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

        # GPU mode: sequential evaluation (CUDA context is thread-local)
        if self.device == "cuda":
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
        Evaluates a single genome with progressive tier-based sampling.

        Returns:
            Tuple of (queries_used, exit_tier_name)
        """
        if not self.enable_adaptive:
            return self._evaluate_single_full(genome, queries, ground_truth)

        retriever_kwargs = self.compiler.compile(genome)
        decision_tracker = self._create_decision_tracker()

        start_time = time.time()
        all_results = []
        queries_evaluated = 0
        exit_tier = "full"

        for tier in self.tiers:
            tier_end = min(tier.queries, len(queries))
            tier_start = queries_evaluated

            if tier_start >= tier_end:
                continue

            tier_queries = queries[tier_start:tier_end]
            tier_gt = ground_truth[tier_start:tier_end]

            # Evaluate this tier's queries
            if decision_tracker is not None and tier_start == 0:
                # Use single-query mode for first tier to capture decisions
                tier_results = []
                for q in tier_queries:
                    res = self.retriever.retrieve(
                        query=q,
                        decision_tracker=decision_tracker,
                        **retriever_kwargs
                    )
                    tier_results.append(res)
            else:
                # Use batch mode for speed
                tier_results = self.retriever.retrieve_batch(
                    queries=tier_queries,
                    max_workers=self.max_workers_per_retrieval,
                    genome_id=f"{genome.id}_tier_{tier.name}",
                    **retriever_kwargs
                )

            all_results.extend(tier_results)
            queries_evaluated = tier_end

            # Compute metrics so far
            current_metrics = self._compute_metrics_cumulative(
                all_results, ground_truth[:queries_evaluated]
            )

            # Check early exit threshold
            if tier.threshold is not None:
                # Add latency/complexity BEFORE fitness calculation so cost_score is correct
                elapsed = time.time() - start_time
                current_metrics['latency'] = elapsed / max(1, queries_evaluated)
                current_metrics['complexity'] = float(genome.complexity())

                current_fitness = self.fitness_calc.calculate(current_metrics, genome)

                if current_fitness.quality_score < tier.threshold:
                    # Early exit - this genome isn't promising
                    exit_tier = tier.name

                    genome.metrics = current_metrics
                    genome.fitness = current_fitness
                    genome.evaluated = True

                    if decision_tracker is not None:
                        genome.decision_context = decision_tracker.to_summary_dict()

                    logger.debug(
                        f"  > [Early Exit] {genome.id} at tier '{tier.name}' "
                        f"(qual={current_fitness.quality_score:.4f} < {tier.threshold})"
                    )

                    return queries_evaluated, exit_tier

        # Full evaluation completed
        total_latency = time.time() - start_time
        final_metrics = self._compute_metrics_cumulative(all_results, ground_truth[:queries_evaluated])
        final_metrics['latency'] = total_latency / max(1, queries_evaluated)
        final_metrics['complexity'] = float(genome.complexity())

        genome.metrics = final_metrics
        genome.fitness = self.fitness_calc.calculate(final_metrics, genome)
        genome.evaluated = True

        if decision_tracker is not None:
            genome.decision_context = decision_tracker.to_summary_dict()

        return queries_evaluated, exit_tier

    def _evaluate_single_full(
        self,
        genome: Genome,
        queries: List[str],
        ground_truth: List[List[Any]]
    ) -> Tuple[int, str]:
        """Full evaluation without adaptive sampling (fallback)."""
        retriever_kwargs = self.compiler.compile(genome)
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

        # Extract IDs as tensor, padding with -1 for missing values
        retrieved_ids = torch.full((n_queries, max_retrieved), -1, dtype=torch.long)

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

        # Use GPU-accelerated batch metric computation if on CUDA
        if self.device == "cuda":
            try:
                retrieved_ids_tensor = retrieved_ids.to(device="cuda")
                metrics = MetricFunctions.compute_all_metrics_batch_gpu(
                    retrieved_ids_tensor,
                    gt_sets,
                    k_values=self.evaluator.k_values,
                    device="cuda"
                )
            except Exception as e:
                logger.debug(f"GPU metrics failed, falling back to CPU: {e}")
                metrics = MetricFunctions.compute_all_metrics_batch(
                    retrieved_ids,
                    gt_sets,
                    k_values=self.evaluator.k_values
                )
        else:
            # CPU batch computation
            metrics = MetricFunctions.compute_all_metrics_batch(
                retrieved_ids,
                gt_sets,
                k_values=self.evaluator.k_values
            )

        # Add variance estimates (using simplified calculation)
        for key in list(metrics.keys()):
            metrics[f"var_{key}"] = 0.0  # Simplified - full variance would need per-query scores

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
            t = torch.tensor(values, dtype=torch.float32)
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
