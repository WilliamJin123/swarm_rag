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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
import numpy as np
import logging

from .device import get_device, ensure_tensor, to_numpy, clear_device_cache

logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


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

    Usage:
        benchmarker = Benchmarker(warmup_iterations=3, n_iterations=10)

        # Benchmark a function
        result = benchmarker.run(
            "vector_search",
            func=lambda: store.search(query, 100),
            device="cuda"
        )

        # Compare CPU vs GPU
        comparison = benchmarker.compare(
            "vector_search",
            cpu_func=lambda: cpu_store.search(query, 100),
            gpu_func=lambda: gpu_store.search(query, 100)
        )
    """

    def __init__(
        self,
        warmup_iterations: int = 3,
        n_iterations: int = 10,
        sync_cuda: bool = True
    ):
        """
        Initialize benchmarker.

        Args:
            warmup_iterations: Number of warmup runs before measurement
            n_iterations: Number of timed iterations
            sync_cuda: Whether to synchronize CUDA before timing (for accurate GPU timing)
        """
        self.warmup_iterations = warmup_iterations
        self.n_iterations = n_iterations
        self.sync_cuda = sync_cuda

    def _sync_if_needed(self, device: str):
        """Synchronize CUDA if needed for accurate timing."""
        if self.sync_cuda and device == "cuda" and _TORCH_AVAILABLE:
            torch.cuda.synchronize()

    def run(
        self,
        name: str,
        func: Callable[[], Any],
        device: str = None,
        n_ops: int = 1,
        extra_metrics_func: Callable[[Any], Dict[str, float]] = None
    ) -> BenchmarkResult:
        """
        Run a benchmark.

        Args:
            name: Name of the benchmark
            func: Function to benchmark (should take no arguments)
            device: Device being used ("cuda" or "cpu")
            n_ops: Number of operations per call (for throughput calculation)
            extra_metrics_func: Optional function to compute extra metrics from result

        Returns:
            BenchmarkResult with timing statistics
        """
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

        # Extra metrics
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
        """
        Compare CPU vs GPU performance.

        Args:
            name: Name of the benchmark
            cpu_func: CPU implementation
            gpu_func: GPU implementation
            n_ops: Number of operations per call

        Returns:
            ComparisonResult with speedup factor
        """
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

    Args:
        n_docs: Number of documents in the index
        dim: Embedding dimension
        n_queries: Number of queries to run
        top_k: Number of results per query
        warmup: Warmup iterations
        iterations: Timed iterations

    Returns:
        ComparisonResult with speedup factor
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for vector search benchmark")

    logger.info(f"Generating benchmark data: {n_docs} docs, {dim} dim, {n_queries} queries")

    # Generate random data
    np.random.seed(42)
    doc_embeddings = np.random.randn(n_docs, dim).astype(np.float32)
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

    query_embeddings = np.random.randn(n_queries, dim).astype(np.float32)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    # CPU setup (NumPy)
    def cpu_search():
        results = []
        for q in query_embeddings:
            scores = np.dot(doc_embeddings, q)
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            results.append(top_indices)
        return results

    # GPU setup (PyTorch)
    doc_tensor = torch.from_numpy(doc_embeddings).cuda()
    query_tensor = torch.from_numpy(query_embeddings).cuda()

    def gpu_search():
        # Batch matrix multiplication
        scores = torch.mm(query_tensor, doc_tensor.t())
        _, top_indices = torch.topk(scores, top_k, dim=1)
        return top_indices.cpu().numpy()

    benchmarker = Benchmarker(warmup_iterations=warmup, n_iterations=iterations)

    result = benchmarker.compare(
        name=f"vector_search_{n_docs}docs_{n_queries}queries",
        cpu_func=cpu_search,
        gpu_func=gpu_search,
        n_ops=n_queries
    )

    # Clean up GPU memory after benchmark
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
    """
    Benchmark batch similarity computation at various batch sizes.

    Args:
        n_candidates: Number of candidate vectors
        dim: Embedding dimension
        batch_sizes: List of batch sizes to test
        warmup: Warmup iterations
        iterations: Timed iterations

    Returns:
        List of ComparisonResults for each batch size
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for similarity benchmark")

    if batch_sizes is None:
        batch_sizes = [1, 10, 50, 100, 500]

    results = []
    benchmarker = Benchmarker(warmup_iterations=warmup, n_iterations=iterations)

    for batch_size in batch_sizes:
        logger.info(f"Benchmarking batch size {batch_size}")

        # Generate data
        np.random.seed(42)
        candidates = np.random.randn(n_candidates, dim).astype(np.float32)
        candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)

        queries = np.random.randn(batch_size, dim).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

        # CPU
        def cpu_similarity():
            return np.dot(queries, candidates.T)

        # GPU
        candidates_gpu = torch.from_numpy(candidates).cuda()
        queries_gpu = torch.from_numpy(queries).cuda()

        def gpu_similarity():
            return torch.mm(queries_gpu, candidates_gpu.t()).cpu().numpy()

        comparison = benchmarker.compare(
            name=f"batch_similarity_batch{batch_size}",
            cpu_func=cpu_similarity,
            gpu_func=gpu_similarity,
            n_ops=batch_size
        )
        results.append(comparison)

        # Clean up GPU memory between batch sizes
        del candidates_gpu, queries_gpu
        torch.cuda.empty_cache()

    return results


def benchmark_heuristics(
    n_candidates: int = 500,
    dim: int = 768,
    warmup: int = 3,
    iterations: int = 20
) -> List[ComparisonResult]:
    """
    Benchmark heuristic computations on CPU vs GPU.

    Args:
        n_candidates: Number of candidate vectors
        dim: Embedding dimension
        warmup: Warmup iterations
        iterations: Timed iterations

    Returns:
        List of ComparisonResults for each heuristic
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for heuristics benchmark")

    results = []
    benchmarker = Benchmarker(warmup_iterations=warmup, n_iterations=iterations)

    # Generate data
    np.random.seed(42)
    query_np = np.random.randn(dim).astype(np.float32)
    query_np = query_np / np.linalg.norm(query_np)

    targets_np = np.random.randn(n_candidates, dim).astype(np.float32)
    targets_np = targets_np / np.linalg.norm(targets_np, axis=1, keepdims=True)

    query_gpu = torch.from_numpy(query_np).cuda()
    targets_gpu = torch.from_numpy(targets_np).cuda()

    # Semantic similarity
    def cpu_semantic():
        scores = np.dot(targets_np, query_np)
        return (scores + 1.0) / 2.0

    def gpu_semantic():
        scores = torch.matmul(targets_gpu, query_gpu)
        return ((scores + 1.0) / 2.0).cpu().numpy()

    results.append(benchmarker.compare(
        "semantic_similarity",
        cpu_semantic,
        gpu_semantic,
        n_ops=n_candidates
    ))

    # Node centrality (pure numpy, no GPU benefit expected)
    degrees_np = np.random.randint(1, 100, n_candidates).astype(np.float32)
    avg_degree = 50.0

    def cpu_centrality():
        log_degrees = np.log(1 + degrees_np)
        log_avg = np.log(1 + avg_degree)
        return log_degrees / (log_degrees + log_avg + 1e-8)

    degrees_gpu = torch.from_numpy(degrees_np).cuda()
    log_avg_t = np.log(1 + avg_degree)

    def gpu_centrality():
        log_degrees = torch.log(1 + degrees_gpu)
        return (log_degrees / (log_degrees + log_avg_t + 1e-8)).cpu().numpy()

    results.append(benchmarker.compare(
        "node_centrality",
        cpu_centrality,
        gpu_centrality,
        n_ops=n_candidates
    ))

    # Clean up GPU memory after heuristics benchmark
    del query_gpu, targets_gpu, degrees_gpu
    torch.cuda.empty_cache()

    return results


def run_all_benchmarks(
    n_docs: int = 50000,
    dim: int = 768,
    verbose: bool = True
) -> Dict[str, Union[ComparisonResult, List[ComparisonResult]]]:
    """
    Run all benchmarks and return comprehensive results.

    Args:
        n_docs: Number of documents for search benchmarks
        dim: Embedding dimension
        verbose: Whether to print results

    Returns:
        Dictionary of benchmark results
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for benchmarks")

    device = get_device()
    if device != "cuda":
        logger.warning("GPU not available, skipping GPU benchmarks")
        return {}

    results = {}

    # Vector search
    logger.info("Running vector search benchmark...")
    results['vector_search'] = benchmark_vector_search(
        n_docs=n_docs, dim=dim, n_queries=100
    )

    # Batch similarity
    logger.info("Running batch similarity benchmark...")
    results['batch_similarity'] = benchmark_batch_similarity(
        n_candidates=5000, dim=dim
    )

    # Heuristics
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
    """Print a formatted summary of benchmark results."""
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
