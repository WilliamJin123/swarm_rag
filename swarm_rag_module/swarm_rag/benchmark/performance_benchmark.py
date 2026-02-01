"""
Performance Benchmark Harness for Evolution Optimization Stack.

Validates that the full optimization stack (memory guards, fitness cache,
embedding cache, convergence detection, async checkpointing) achieves
500 generations in 3 hours with population 50-100 and peak VRAM under 4GB.

Design decisions:
- Single run (no statistical replication needed for validation)
- No warm-up period (include cold-start in timing for realistic measurement)
- Population size: 75 (midpoint of 50-100 target range)
- Allow convergence early-stop with time extrapolation to 500 generations
"""
import sys
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch

from ..utils.device import get_device, get_gpu_memory_info
from ..evolution.execution.fitness_cache import FitnessCache
from ..evolution.execution.embedding_cache import EmbeddingCacheProvider

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmark."""
    population_size: int = 75
    target_generations: int = 500
    time_limit_hours: float = 3.0
    memory_limit_gb: float = 4.0


@dataclass
class BenchmarkResult:
    """
    Results from a performance benchmark run.

    Contains all metrics needed for pass/fail determination and
    detailed analysis of performance bottlenecks.
    """
    # Metadata
    timestamp: str
    overall_pass: bool

    # Configuration used
    config: Dict[str, Any]

    # Results
    actual_generations: int
    termination_reason: str
    was_extrapolated: bool

    # Timing metrics
    total_seconds: float
    projected_seconds: float
    generation_times_ms: List[float]
    avg_generation_ms: float
    min_generation_ms: float
    max_generation_ms: float
    p99_generation_ms: float

    # Memory metrics
    peak_allocated_mb: float
    peak_reserved_mb: float
    final_allocated_mb: float

    # Cache statistics
    fitness_cache_hit_rate: float
    fitness_cache_total_lookups: int
    embedding_cache_hit_rate: float
    embedding_cache_time_saved_sec: float

    # System info
    system_info: Dict[str, Any]

    # Pass criteria details
    time_pass: bool
    time_reason: str
    memory_pass: bool
    memory_reason: str

    # Report output path
    report_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "$schema": "benchmark-results-v1",
            "timestamp": self.timestamp,
            "pass": self.overall_pass,
            "config": self.config,
            "results": {
                "actual_generations": self.actual_generations,
                "termination_reason": self.termination_reason,
                "was_extrapolated": self.was_extrapolated,
                "timing": {
                    "total_seconds": self.total_seconds,
                    "projected_seconds": self.projected_seconds,
                    "avg_generation_ms": self.avg_generation_ms,
                    "min_generation_ms": self.min_generation_ms,
                    "max_generation_ms": self.max_generation_ms,
                    "p99_generation_ms": self.p99_generation_ms,
                },
                "memory": {
                    "peak_allocated_mb": self.peak_allocated_mb,
                    "peak_reserved_mb": self.peak_reserved_mb,
                    "final_allocated_mb": self.final_allocated_mb,
                },
                "cache_stats": {
                    "fitness_cache_hit_rate": self.fitness_cache_hit_rate,
                    "fitness_cache_total_lookups": self.fitness_cache_total_lookups,
                    "embedding_cache_hit_rate": self.embedding_cache_hit_rate,
                    "embedding_cache_time_saved_sec": self.embedding_cache_time_saved_sec,
                },
            },
            "system_info": self.system_info,
            "pass_criteria": {
                "time_pass": self.time_pass,
                "time_reason": self.time_reason,
                "memory_pass": self.memory_pass,
                "memory_reason": self.memory_reason,
            },
        }

    def to_console_summary(self) -> str:
        """Generate formatted console summary matching RESEARCH.md format."""
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("PERFORMANCE VALIDATION BENCHMARK - RESULTS")
        lines.append("=" * 80)
        lines.append("")

        # Configuration
        lines.append("Configuration:")
        lines.append(f"  Population size:     {self.config['population_size']}")
        lines.append(f"  Target generations:  {self.config['target_generations']}")
        lines.append(f"  Time limit:          {self.config['time_limit_hours']:.1f} hours")
        lines.append(f"  Memory limit:        {self.config['memory_limit_gb']:.1f} GB")
        lines.append("")

        # Results
        lines.append("Results:")
        converged_indicator = " (converged)" if self.termination_reason == "convergence" else ""
        lines.append(f"  Actual generations:  {self.actual_generations}{converged_indicator}")
        lines.append(f"  Total time:          {self._format_time(self.total_seconds)}")
        if self.was_extrapolated:
            lines.append(f"  Projected time:      {self._format_time(self.projected_seconds)} (extrapolated to {self.config['target_generations']} gens)")
        else:
            lines.append(f"  Projected time:      {self._format_time(self.projected_seconds)}")
        lines.append("")

        # Generation timing
        lines.append("  Generation timing:")
        lines.append(f"    Average:  {self.avg_generation_ms:.0f} ms")
        lines.append(f"    Min:      {self.min_generation_ms:.0f} ms")
        lines.append(f"    Max:      {self.max_generation_ms:.0f} ms")
        lines.append(f"    P99:      {self.p99_generation_ms:.0f} ms")
        lines.append("")

        # Memory
        lines.append("  Memory:")
        lines.append(f"    Peak allocated:  {self.peak_allocated_mb:.1f} MB")
        if self.peak_reserved_mb > 0:
            lines.append(f"    Peak reserved:   {self.peak_reserved_mb:.1f} MB")
        lines.append("")

        # Cache performance
        lines.append("  Cache performance:")
        lines.append(f"    Fitness cache:   {self.fitness_cache_hit_rate * 100:.1f}% hit rate ({self.fitness_cache_total_lookups} lookups)")
        lines.append(f"    Embedding cache: {self.embedding_cache_hit_rate * 100:.1f}% hit rate (saved {self.embedding_cache_time_saved_sec:.1f}s)")
        lines.append("")

        # Pass criteria
        lines.append("Pass Criteria:")
        time_status = "[PASS]" if self.time_pass else "[FAIL]"
        memory_status = "[PASS]" if self.memory_pass else "[FAIL]"
        lines.append(f"  {time_status} Time:   {self._format_time(self.projected_seconds)} < {self._format_time(self.config['time_limit_hours'] * 3600)}")
        lines.append(f"  {memory_status} Memory: {self.peak_allocated_mb:.0f} MB < {self.config['memory_limit_gb'] * 1024:.0f} MB")
        lines.append("")

        # Overall result
        lines.append("=" * 80)
        overall_status = "PASS" if self.overall_pass else "FAIL"
        lines.append(f"OVERALL: {overall_status}")
        lines.append("=" * 80)
        lines.append("")

        if self.report_path:
            lines.append(f"Report saved to: {self.report_path}")

        return "\n".join(lines)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as human-readable time string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60

        if hours > 0:
            return f"{hours}h {minutes:02d}m {secs:.1f}s"
        elif minutes > 0:
            return f"{minutes}m {secs:.1f}s"
        else:
            return f"{secs:.1f}s"


class PerformanceBenchmark:
    """
    Performance benchmark harness that wraps EvolutionEngine.

    Validates the full optimization stack (memory guards, fitness cache,
    embedding cache, convergence detection, async checkpointing) achieves
    target performance metrics.

    Usage:
        config = BenchmarkConfig(population_size=75, target_generations=500)
        benchmark = PerformanceBenchmark(config)
        result = benchmark.run()
        print(result.to_console_summary())
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark with configuration.

        Args:
            config: BenchmarkConfig with population, generation, and limit settings
        """
        self.config = config
        self._generation_times_ms: List[float] = []
        self._peak_allocated_bytes: int = 0
        self._peak_reserved_bytes: int = 0
        self._has_cuda = torch.cuda.is_available()

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for reproducibility."""
        info = {
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "cuda_available": self._has_cuda,
        }

        if self._has_cuda:
            info["cuda_version"] = torch.version.cuda or "N/A"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_total_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
            cudnn_version = torch.backends.cudnn.version()
            info["cudnn_version"] = cudnn_version if cudnn_version else "N/A"

        return info

    def _extrapolate_time(
        self,
        actual_gens: int,
        actual_time_sec: float,
        target_gens: int = 500
    ) -> Tuple[float, bool]:
        """
        Extrapolate total time if convergence triggered early.

        Uses simple linear extrapolation based on average generation time.
        This is valid because:
        - Memory usage is bounded (cache eviction, memory guards)
        - Generation work is roughly constant (fixed population size)
        - No progressive slowdown expected (addressed in earlier phases)

        Args:
            actual_gens: Generations actually completed
            actual_time_sec: Wall clock time for actual generations
            target_gens: Target generation count (default 500)

        Returns:
            Tuple of (projected_time_seconds, was_extrapolated)
        """
        if actual_gens >= target_gens:
            return actual_time_sec, False

        if actual_gens == 0:
            return 0.0, False

        avg_gen_time = actual_time_sec / actual_gens
        projected = avg_gen_time * target_gens

        return projected, True

    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value from a list."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    def run(self) -> BenchmarkResult:
        """
        Run the performance benchmark.

        This method:
        1. Loads STARK Prime data (full dataset for realistic performance)
        2. Configures EvolutionEngine with benchmark settings
        3. Runs evolution with timing instrumentation
        4. Collects all metrics (timing, memory, cache)
        5. Determines pass/fail based on criteria
        6. Returns BenchmarkResult

        Returns:
            BenchmarkResult with all metrics and pass/fail determination
        """
        logger.info("Starting performance benchmark...")
        logger.info(f"Config: population={self.config.population_size}, "
                    f"generations={self.config.target_generations}")

        # Collect system info
        system_info = self._collect_system_info()
        logger.info(f"System: {system_info.get('gpu_name', 'CPU mode')}")

        # Reset peak memory stats if CUDA available
        if self._has_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Record start time (after CUDA sync for accurate timing)
        start_time = time.perf_counter()

        # Run evolution and collect metrics
        actual_generations, termination_reason = self._run_evolution()

        # Synchronize before final timing
        if self._has_cuda:
            torch.cuda.synchronize()

        total_time = time.perf_counter() - start_time

        # Collect final memory stats
        if self._has_cuda:
            self._peak_allocated_bytes = torch.cuda.max_memory_allocated()
            self._peak_reserved_bytes = torch.cuda.memory_reserved()
            final_allocated_bytes = torch.cuda.memory_allocated()
        else:
            final_allocated_bytes = 0

        # Collect cache statistics
        fitness_stats = self._get_fitness_cache_stats()
        embedding_stats = self._get_embedding_cache_stats()

        # Calculate timing statistics
        if self._generation_times_ms:
            avg_gen_ms = sum(self._generation_times_ms) / len(self._generation_times_ms)
            min_gen_ms = min(self._generation_times_ms)
            max_gen_ms = max(self._generation_times_ms)
            p99_gen_ms = self._calculate_percentile(self._generation_times_ms, 99)
        else:
            avg_gen_ms = min_gen_ms = max_gen_ms = p99_gen_ms = 0.0

        # Extrapolate time if converged early
        projected_time, was_extrapolated = self._extrapolate_time(
            actual_generations, total_time, self.config.target_generations
        )

        # Convert memory to MB
        peak_allocated_mb = self._peak_allocated_bytes / (1024 * 1024)
        peak_reserved_mb = self._peak_reserved_bytes / (1024 * 1024)
        final_allocated_mb = final_allocated_bytes / (1024 * 1024)

        # Determine pass/fail
        time_limit_sec = self.config.time_limit_hours * 3600
        memory_limit_mb = self.config.memory_limit_gb * 1024

        time_pass = projected_time < time_limit_sec
        time_reason = (
            f"Projected {projected_time:.1f}s < {time_limit_sec:.1f}s ({self.config.time_limit_hours}h limit)"
            if time_pass else
            f"Projected {projected_time:.1f}s >= {time_limit_sec:.1f}s ({self.config.time_limit_hours}h limit)"
        )

        memory_pass = peak_allocated_mb < memory_limit_mb
        memory_reason = (
            f"Peak {peak_allocated_mb:.1f}MB < {memory_limit_mb:.1f}MB ({self.config.memory_limit_gb}GB limit)"
            if memory_pass else
            f"Peak {peak_allocated_mb:.1f}MB >= {memory_limit_mb:.1f}MB ({self.config.memory_limit_gb}GB limit)"
        )

        overall_pass = time_pass and memory_pass

        # Build result
        result = BenchmarkResult(
            timestamp=datetime.utcnow().isoformat() + "Z",
            overall_pass=overall_pass,
            config={
                "population_size": self.config.population_size,
                "target_generations": self.config.target_generations,
                "time_limit_hours": self.config.time_limit_hours,
                "memory_limit_gb": self.config.memory_limit_gb,
            },
            actual_generations=actual_generations,
            termination_reason=termination_reason,
            was_extrapolated=was_extrapolated,
            total_seconds=total_time,
            projected_seconds=projected_time,
            generation_times_ms=self._generation_times_ms,
            avg_generation_ms=avg_gen_ms,
            min_generation_ms=min_gen_ms,
            max_generation_ms=max_gen_ms,
            p99_generation_ms=p99_gen_ms,
            peak_allocated_mb=peak_allocated_mb,
            peak_reserved_mb=peak_reserved_mb,
            final_allocated_mb=final_allocated_mb,
            fitness_cache_hit_rate=fitness_stats.get("hit_rate", 0.0),
            fitness_cache_total_lookups=fitness_stats.get("total_lookups", 0),
            embedding_cache_hit_rate=embedding_stats.get("hit_rate", 0.0),
            embedding_cache_time_saved_sec=embedding_stats.get("time_saved_sec", 0.0),
            system_info=system_info,
            time_pass=time_pass,
            time_reason=time_reason,
            memory_pass=memory_pass,
            memory_reason=memory_reason,
        )

        logger.info(f"Benchmark complete: {'PASS' if overall_pass else 'FAIL'}")
        return result

    def _run_evolution(self) -> Tuple[int, str]:
        """
        Run evolution and collect per-generation timing.

        Returns:
            Tuple of (actual_generations_completed, termination_reason)
        """
        import os
        import random

        # Import here to avoid circular imports
        from ..evolution.engine import EvolutionEngine
        from ..evolution.types.config import (
            EvolutionConfig,
            MapElitesConfig,
            StorageConfig,
            ResourceConfig,
            ConvergenceConfig,
        )
        from ..evolution.execution.fitness import create_fitness_calculator
        from ..eval import Evaluator
        from ..core import SwarmRetriever
        from ..integrations.stark import (
            StarkGraphAdapter,
            StarkPreComputedEmbeddingHandler,
            StarkVectorStore,
        )
        from ..utils.device import resolve_device

        # Import STARK data loading functions
        # These are in the stark/ directory at project root
        import sys
        # __file__ is: swarm_rag_module/swarm_rag/benchmark/performance_benchmark.py
        # parents[3] is: swarm_rag_experiment (project root)
        project_root = Path(__file__).resolve().parents[3]
        stark_dir = project_root / "stark"
        if str(stark_dir) not in sys.path:
            sys.path.insert(0, str(stark_dir))

        from load_stark import (
            load_and_download_embeddings,
            load_and_download_skb,
            load_and_download_qa,
            precompute_stark_adjacency,
        )

        # Use prime dataset (129K docs vs amazon's 957K) for benchmark
        # Prime fits in most GPU memory configurations
        dataset_name = "prime"

        logger.info(f"Loading STARK {dataset_name} dataset...")
        skb = load_and_download_skb(dataset_name)
        adj_dict = precompute_stark_adjacency(skb, dataset_name)
        query_embs, doc_embs = load_and_download_embeddings(dataset_name)

        logger.info(f"Loaded {len(query_embs)} queries, {len(doc_embs)} documents")

        # Prepare train/val splits
        raw_data = load_and_download_qa(dataset_name)
        train_subset = list(raw_data.get_subset("train"))
        val_subset = list(raw_data.get_subset("val"))

        random.seed(42)
        random.shuffle(train_subset)
        random.shuffle(val_subset)

        # Limit data for faster benchmark (full dataset for realistic perf)
        train_q = [d[0] for d in train_subset]
        train_ids = [d[1] for d in train_subset]
        train_gt = [d[2] for d in train_subset]

        val_q = [d[0] for d in val_subset[:100]]
        val_ids = [d[1] for d in val_subset[:100]]
        val_gt = [d[2] for d in val_subset[:100]]

        logger.info(f"Using {len(train_ids)} train queries, {len(val_ids)} val queries")

        # Resolve device
        resolved_device = resolve_device("auto")
        logger.info(f"Using device: {resolved_device}")

        # Initialize components
        vector_store = StarkVectorStore(doc_embs, device=resolved_device)
        graph_store = StarkGraphAdapter(
            skb, dataset_name,
            adjacency_dict=adj_dict,
            cache_path=str(stark_dir / "adjacency_cache" / f"graph_{dataset_name}.npz"),
            device=resolved_device,
        )
        embedding_provider = StarkPreComputedEmbeddingHandler(query_embs)

        retriever = SwarmRetriever(
            vector_store=vector_store,
            graph_store=graph_store,
            embedding_provider=embedding_provider,
            cache_neighbors=False,
            cache_vectors=True,
            device=resolved_device,
        )

        evaluator = Evaluator(k_values=[1, 5, 10, 20])
        fitness_calc = create_fitness_calculator(
            mode="weighted_sum",
            weights={"Hit@1": 0.25, "Hit@5": 0.25, "MRR": 0.25, "Recall@20": 0.25},
        )

        # Import additional config types
        from ..evolution.types.config import (
            GeneticConfig,
            STARK_FEATURES,
        )

        # Configure evolution with benchmark settings
        # Use weighted_sum mode for faster evaluation (vs expression trees)
        config = EvolutionConfig(
            genome_mode="weighted_sum",
            heuristic_features=STARK_FEATURES,
            n_generations=self.config.target_generations,
            map_elites=MapElitesConfig(
                dimensions=["aggressiveness", "complexity"],
                bins=[10, 10],
                ranges=[(10.0, 150.0), (5.0, 60.0)],
                initial_fill=self.config.population_size,
                batch_size=min(15, self.config.population_size // 3),
            ),
            resources=ResourceConfig(
                concurrent_evaluations=4,
                max_workers_per_retrieval=4,
                enable_shared_precompute=True,
                enable_cross_genome_metric_batch=True,
                early_exit_threshold=0.25,
                run_mode="batch",
            ),
            genetic=GeneticConfig(
                creation_strategy="baseline_seeded_initialization",
                mutation_strategy="guided_mutation",
                crossover_strategy="uniform_parameter_mix",
                base_mutation_rate=0.20,
                crossover_rate=0.6,
                n_agent_groups=3,
            ),
            storage=StorageConfig(
                checkpoint_frequency=50,  # Less frequent for benchmark
                base_dir=str(project_root / ".planning" / "phases" / "06-performance-validation" / "benchmark_runs"),
            ),
            convergence=ConvergenceConfig(
                enabled=True,
                window_size=40,
                threshold_percentage=0.001,
                grace_period=20,
            ),
        )

        # Create engine
        logger.info("Creating EvolutionEngine...")
        engine = EvolutionEngine(
            retriever=retriever,
            fitness_calculator=fitness_calc,
            evaluator=evaluator,
            train_query_ids=train_ids,
            train_ground_truth=train_gt,
            val_query_ids=val_ids,
            val_ground_truth=val_gt,
            config=config,
        )

        # Store evaluator reference for cache stats
        self._evaluator = engine.population_evaluator

        # Hook into profiler to collect generation times
        orchestrator = engine._orchestrator
        original_profiler = orchestrator._profiler

        # Enable profiler for timing collection
        original_profiler.enabled = True

        # Run evolution
        logger.info(f"Starting evolution ({self.config.target_generations} generations max)...")
        try:
            best_genome = engine.optimize()
            logger.info(f"Best genome fitness: {best_genome.fitness.quality_score:.4f}")
        except KeyboardInterrupt:
            logger.warning("Benchmark interrupted by user")
            return len(self._generation_times_ms), "user_interrupt"
        except MemoryError as e:
            logger.error(f"Memory error during benchmark: {e}")
            return len(self._generation_times_ms), "memory_limit"

        # Extract generation times from profiler
        if original_profiler.enabled and original_profiler.generation_timings:
            for gen in sorted(original_profiler.generation_timings.keys()):
                timings = original_profiler.generation_timings[gen]
                total_ms = sum(timings.values())
                self._generation_times_ms.append(total_ms)
        else:
            # Fallback: estimate from total time
            actual_gens = orchestrator.context.generation + 1
            if actual_gens > 0:
                avg_time_ms = (time.perf_counter() * 1000) / actual_gens
                self._generation_times_ms = [avg_time_ms] * actual_gens

        # Get termination reason from orchestrator
        termination_reason = orchestrator._termination_reason.value

        return len(self._generation_times_ms), termination_reason

    def _get_fitness_cache_stats(self) -> Dict[str, Any]:
        """Get fitness cache statistics from the evaluator."""
        try:
            # Access via stored evaluator reference
            if hasattr(self, '_evaluator') and self._evaluator is not None:
                cache = self._evaluator._fitness_cache
                if cache is not None:
                    stats = cache.total_stats
                    return {
                        "hit_rate": stats.hit_rate,
                        "total_lookups": stats.total,
                    }
            return {
                "hit_rate": 0.0,
                "total_lookups": 0,
            }
        except Exception:
            return {"hit_rate": 0.0, "total_lookups": 0}

    def _get_embedding_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics from the provider."""
        try:
            embed_cache = EmbeddingCacheProvider.get()
            if embed_cache is not None:
                stats = embed_cache.stats
                return {
                    "hit_rate": stats.hit_rate,
                    "time_saved_sec": stats.compute_time_saved_sec,
                }
            return {"hit_rate": 0.0, "time_saved_sec": 0.0}
        except Exception:
            return {"hit_rate": 0.0, "time_saved_sec": 0.0}
