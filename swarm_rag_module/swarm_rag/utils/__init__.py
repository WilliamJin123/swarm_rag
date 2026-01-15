"""
swarm_rag utilities - re-exports from submodules for backward compatibility.
"""

import importlib.util
from threading import Lock
from collections import OrderedDict
import logging
from tqdm.auto import tqdm

# Re-export device utilities
from .device import (
    get_device, get_array_module, ensure_tensor, to_numpy,
    clear_device_cache, get_gpu_memory_info,
    # CuPy integration
    is_cupy_available, to_cupy, cupy_to_numpy,
    cupy_matmul, cupy_dot, cupy_norm, cupy_normalize,
    cupy_cosine_similarity, cupy_topk, sync_device
)

# Re-export benchmark utilities
from .benchmark import (
    Benchmarker,
    BenchmarkResult,
    ComparisonResult,
    benchmark_vector_search,
    benchmark_batch_similarity,
    benchmark_heuristics,
    run_all_benchmarks,
    print_benchmark_summary
)

# Re-export memory utilities
from .memory import (
    MemoryProfiler,
    MemorySnapshot,
    MemoryDelta,
    get_gpu_memory_info as memory_get_gpu_info,
    clear_gpu_cache,
    memory_guard,
    estimate_tensor_memory
)

def fail_on_missing_imports(modules: list[str], extra_name: str = None):
    """
    Checks if a list of modules can be imported.
    If not, raises an ImportError with the specific pip command to fix it.

    Args:
        modules: List of python import names (e.g. ['torch', 'stark_qa'])
        extra_name: The name of the extra in pyproject.toml (e.g. 'stark')
    """
    missing = [
        m for m in modules
        if importlib.util.find_spec(m) is None
    ]
    if not missing:
        return

    if extra_name is not None:
        msg = (
            f"Missing required dependencies: {', '.join(missing)}.\n"
            f"Please install them by running:\n\n"
            f"    pip install \"swarm_rag[{extra_name}]\""
        )
    else:
        msg = (
            f"Missing required dependencies: {', '.join(missing)}.\n"
            f"Please install them by running:\n\n"
            f"    pip install {' '.join(missing)}"
        )

    raise ImportError(msg) from None

class LRUCache:
    __slots__ = ['maxsize', 'data', 'lock']

    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.data = OrderedDict()
        self.lock = Lock()

    def get(self, key):
         with self.lock:
            if key not in self.data:
                return None
            self.data.move_to_end(key)
            return self.data[key]
    def set(self, key, value):
        with self.lock:
            self.data[key] = value
            self.data.move_to_end(key)
            if len(self.data) > self.maxsize:
                self.data.popitem(last=False)

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

__all__ = [
    # Core utilities
    'fail_on_missing_imports',
    'LRUCache',
    'TqdmLoggingHandler',
    # Device utilities
    'get_device',
    'get_array_module',
    'ensure_tensor',
    'to_numpy',
    'clear_device_cache',
    'get_gpu_memory_info',
    # CuPy integration
    'is_cupy_available',
    'to_cupy',
    'cupy_to_numpy',
    'cupy_matmul',
    'cupy_dot',
    'cupy_norm',
    'cupy_normalize',
    'cupy_cosine_similarity',
    'cupy_topk',
    'sync_device',
    # Benchmark utilities
    'Benchmarker',
    'BenchmarkResult',
    'ComparisonResult',
    'benchmark_vector_search',
    'benchmark_batch_similarity',
    'benchmark_heuristics',
    'run_all_benchmarks',
    'print_benchmark_summary',
    # Memory utilities
    'MemoryProfiler',
    'MemorySnapshot',
    'MemoryDelta',
    'clear_gpu_cache',
    'memory_guard',
    'estimate_tensor_memory',
]
