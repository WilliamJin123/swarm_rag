"""
Performance Benchmark Module.

Provides benchmarking harness to validate the full evolution optimization stack
achieves target performance (500 generations in 3 hours, peak VRAM under 4GB).
"""
from .performance_benchmark import (
    PerformanceBenchmark,
    BenchmarkResult,
    BenchmarkConfig,
)

__all__ = [
    "PerformanceBenchmark",
    "BenchmarkResult",
    "BenchmarkConfig",
]
