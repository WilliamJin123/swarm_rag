
"""
Benchmarking utilities for CPU vs GPU performance comparison.

Provides comprehensive benchmarks for:
- Vector search operations
- Batch similarity computation
- Heuristic calculations
- End-to-end retrieval performance
"""

import time
import statistics
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
import torch
import logging

from .device import get_device

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    device: str
    n_iterations: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput: float  # operations per second
    extra_metrics: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(name='{self.name}', device='{self.device}', "
            f"mean={self.mean_time_ms:.3f}ms ± {self.std_time_ms:.3f}ms, "
            f"throughput={self.throughput:.1f} ops/s)"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'device': self.device,
            'n_iterations': self.n_iterations,
            'mean_time_ms': self.mean_time_ms,
            'std_time_ms': self.std_time_ms,
            'min_time_ms': self.min_time_ms,
            'max_time_ms': self.max_time_ms,
            'throughput': self.throughput,
            'extra_metrics': self.extra_metrics
        }


@dataclass
class ComparisonResult:
    """Results comparing CPU vs GPU performance."""
    benchmark_name: str
    cpu_result: BenchmarkResult
    gpu_result: BenchmarkResult
    speedup: float  # GPU speedup factor (cpu_time / gpu_time)

    def __repr__(self) -> str:
        return (
            f"ComparisonResult(name='{self.benchmark_name}', "
            f"speedup={self.speedup:.2f}x, "
            f"CPU={self.cpu_result.mean_time_ms:.3f}ms, "
            f"GPU={self.gpu_result.mean_time_ms:.3f}ms)"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'benchmark_name': self.benchmark_name,
            'cpu': self.cpu_result.to_dict(),
            'gpu': self.gpu_result.to_dict(),
            'speedup': self.speedup
        }


class Benchmarker:
    """
    Benchmarking utility for comparing CPU vs GPU performance.
    """

    def __init__(
        self,
        warmup_iterations: int = 3,
        n_iterations: int = 10,
        sync_cuda: bool = True
    ):
        self.warmup_iterations = warmup_iterations
        self.n_iterations = n_iterations
        self.sync_cuda = sync_cuda

    def _sync_if_needed(self, device: str):
        """Synchronize CUDA if needed for accurate timing."""
        if self.sync_cuda and device == "cuda":
            torch.cuda.synchronize()

    def run(
        self,
        name: str,
        func: Callable[[], Any],
        device: str = None,
        n_ops: int = 1,
        extra_metrics_func: Callable[[Any], Dict[str, float]] = None
    ) -> BenchmarkResult:
        device = device or get_device()

        # Warmup
        for _ in range(self.warmup_iterations):
            self._sync_if_needed(device)
            _ = func()
            self._sync_if_needed(device)

        # Timed runs
        times_ms = []
        last_result = None

        for _ in range(self.n_iterations):
            self._sync_if_needed(device)
            start = time.perf_counter()
            result = func()
            self._sync_if_needed(device)
            end = time.perf_counter()

            times_ms.append((end - start) * 1000)
            last_result = result

        # Compute statistics
        mean_time = statistics.mean(times_ms)
        std_time = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        min_time = min(times_ms)
        max_time = max(times_ms)
        throughput = (n_ops * 1000) / mean_time if mean_time > 0 else 0.0

        extra_metrics = {}
        if extra_metrics_func and last_result is not None:
            extra_metrics = extra_metrics_func(last_result)

        return BenchmarkResult(
            name=name,
            device=device,
            n_iterations=self.n_iterations,
            mean_time_ms=mean_time,
            std_time_ms=std_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            throughput=throughput,
            extra_metrics=extra_metrics
        )

    def compare(
        self,
        name: str,
        cpu_func: Callable[[], Any],
        gpu_func: Callable[[], Any],
        n_ops: int = 1
    ) -> ComparisonResult:
        cpu_result = self.run(f"{name}_cpu", cpu_func, device="cpu", n_ops=n_ops)
        gpu_result = self.run(f"{name}_gpu", gpu_func, device="cuda", n_ops=n_ops)

        speedup = cpu_result.mean_time_ms / gpu_result.mean_time_ms if gpu_result.mean_time_ms > 0 else 0.0

        return ComparisonResult(
            benchmark_name=name,
            cpu_result=cpu_result,
            gpu_result=gpu_result,
            speedup=speedup
        )


def benchmark_vector_search(
    n_docs: int = 100000,
    dim: int = 768,
    n_queries: int = 100,
    top_k: int = 100,
    warmup: int = 3,
    iterations: int = 10
) -> ComparisonResult:
    """
    Benchmark vector search CPU vs GPU.
    """
    logger.info(f"Generating benchmark data: {n_docs} docs, {dim} dim, {n_queries} queries")

    torch.manual_seed(42)
    doc_embeddings = torch.randn(n_docs, dim, dtype=torch.float32)
    doc_embeddings = doc_embeddings / torch.linalg.norm(doc_embeddings, dim=1, keepdim=True)

    query_embeddings = torch.randn(n_queries, dim, dtype=torch.float32)
    query_embeddings = query_embeddings / torch.linalg.norm(query_embeddings, dim=1, keepdim=True)

    # CPU setup
    # FIX: Use batched matrix multiplication instead of a Python for-loop.
    # Comparing a Python loop on CPU vs CUDA kernel on GPU is misleading.
    # A fair comparison requires vectorized operations on both sides.
    def cpu_search():
        with torch.no_grad():
            scores = torch.mm(query_embeddings, doc_embeddings.t())
            _, top_indices = torch.topk(scores, top_k, dim=1)
        return top_indices

    # GPU setup
    doc_tensor = doc_embeddings.cuda()
    query_tensor = query_embeddings.cuda()

    def gpu_search():
        with torch.no_grad():
            scores = torch.mm(query_tensor, doc_tensor.t())
            _, top_indices = torch.topk(scores, top_k, dim=1)
            return top_indices.cpu()

    benchmarker = Benchmarker(warmup_iterations=warmup, n_iterations=iterations)

    result = benchmarker.compare(
        name=f"vector_search_{n_docs}docs_{n_queries}queries",
        cpu_func=cpu_search,
        gpu_func=gpu_search,
        n_ops=n_queries
    )

    del doc_tensor, query_tensor
    torch.cuda.empty_cache()

    return result


def benchmark_batch_similarity(
    n_candidates: int = 1000,
    dim: int = 768,
    batch_sizes: List[int] = None,
    warmup: int = 3,
    iterations: int = 10
) -> List[ComparisonResult]:
    if batch_sizes is None:
        batch_sizes = [1, 10, 50, 100, 500]

    results = []
    benchmarker = Benchmarker(warmup_iterations=warmup, n_iterations=iterations)

    for batch_size in batch_sizes:
        logger.info(f"Benchmarking batch size {batch_size}")

        torch.manual_seed(42)
        candidates = torch.randn(n_candidates, dim, dtype=torch.float32)
        candidates = candidates / torch.linalg.norm(candidates, dim=1, keepdim=True)

        queries = torch.randn(batch_size, dim, dtype=torch.float32)
        queries = queries / torch.linalg.norm(queries, dim=1, keepdim=True)

        def cpu_similarity():
            with torch.no_grad():
                return torch.mm(queries, candidates.t())

        candidates_gpu = candidates.cuda()
        queries_gpu = queries.cuda()

        def gpu_similarity():
            with torch.no_grad():
                return torch.mm(queries_gpu, candidates_gpu.t()).cpu()

        comparison = benchmarker.compare(
            name=f"batch_similarity_batch{batch_size}",
            cpu_func=cpu_similarity,
            gpu_func=gpu_similarity,
            n_ops=batch_size
        )
        results.append(comparison)

        del candidates_gpu, queries_gpu
        torch.cuda.empty_cache()

    return results


def benchmark_heuristics(
    n_candidates: int = 500,
    dim: int = 768,
    warmup: int = 3,
    iterations: int = 20
) -> List[ComparisonResult]:
    results = []
    benchmarker = Benchmarker(warmup_iterations=warmup, n_iterations=iterations)

    torch.manual_seed(42)
    query_cpu = torch.randn(dim, dtype=torch.float32)
    query_cpu = query_cpu / torch.linalg.norm(query_cpu)

    targets_cpu = torch.randn(n_candidates, dim, dtype=torch.float32)
    targets_cpu = targets_cpu / torch.linalg.norm(targets_cpu, dim=1, keepdim=True)

    query_gpu = query_cpu.cuda()
    targets_gpu = targets_cpu.cuda()

    # Semantic similarity
    def cpu_semantic():
        with torch.no_grad():
            scores = torch.matmul(targets_cpu, query_cpu)
            return (scores + 1.0) / 2.0

    def gpu_semantic():
        with torch.no_grad():
            scores = torch.matmul(targets_gpu, query_gpu)
            return ((scores + 1.0) / 2.0).cpu()

    results.append(benchmarker.compare(
        "semantic_similarity",
        cpu_semantic,
        gpu_semantic,
        n_ops=n_candidates
    ))

    # Node centrality
    degrees_cpu = torch.randint(1, 100, (n_candidates,), dtype=torch.float32)
    avg_degree = 50.0
    
    # FIX: Pre-calculate scalar values to ensure fair comparison. 
    # The previous CPU implementation created a new tensor inside the loop.
    log_avg_val = math.log(1 + avg_degree)

    def cpu_centrality():
        with torch.no_grad():
            log_degrees = torch.log(1 + degrees_cpu)
            return log_degrees / (log_degrees + log_avg_val + 1e-8)

    degrees_gpu = degrees_cpu.cuda()

    def gpu_centrality():
        with torch.no_grad():
            log_degrees = torch.log(1 + degrees_gpu)
            return (log_degrees / (log_degrees + log_avg_val + 1e-8)).cpu()

    results.append(benchmarker.compare(
        "node_centrality",
        cpu_centrality,
        gpu_centrality,
        n_ops=n_candidates
    ))

    del query_gpu, targets_gpu, degrees_gpu
    torch.cuda.empty_cache()

    return results


def run_all_benchmarks(
    n_docs: int = 50000,
    dim: int = 768,
    verbose: bool = True
) -> Dict[str, Union[ComparisonResult, List[ComparisonResult]]]:
    device = get_device()
    if device != "cuda":
        logger.warning("GPU not available, skipping GPU benchmarks")
        return {}

    results = {}

    logger.info("Running vector search benchmark...")
    results['vector_search'] = benchmark_vector_search(
        n_docs=n_docs, dim=dim, n_queries=100
    )

    logger.info("Running batch similarity benchmark...")
    results['batch_similarity'] = benchmark_batch_similarity(
        n_candidates=5000, dim=dim
    )

    logger.info("Running heuristics benchmark...")
    results['heuristics'] = benchmark_heuristics(
        n_candidates=1000, dim=dim
    )

    if verbose:
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)

        for name, result in results.items():
            print(f"\n{name.upper()}")
            print("-" * 40)

            if isinstance(result, list):
                for r in result:
                    print(f"  {r}")
            else:
                print(f"  {result}")

        print("\n" + "=" * 60)

    return results


def print_benchmark_summary(results: Dict) -> None:
    print("\n" + "=" * 70)
    print(" BENCHMARK SUMMARY - CPU vs GPU Performance")
    print("=" * 70)

    print(f"\n{'Benchmark':<35} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
    print("-" * 70)

    for name, result in results.items():
        if isinstance(result, list):
            for r in result:
                print(f"{r.benchmark_name:<35} {r.cpu_result.mean_time_ms:<12.3f} "
                      f"{r.gpu_result.mean_time_ms:<12.3f} {r.speedup:<10.2f}x")
        else:
            print(f"{result.benchmark_name:<35} {result.cpu_result.mean_time_ms:<12.3f} "
                  f"{result.gpu_result.mean_time_ms:<12.3f} {result.speedup:<10.2f}x")

    print("-" * 70)


__all__ = [
    'BenchmarkResult',
    'ComparisonResult',
    'Benchmarker',
    'benchmark_vector_search',
    'benchmark_batch_similarity',
    'benchmark_heuristics',
    'run_all_benchmarks',
    'print_benchmark_summary'
]