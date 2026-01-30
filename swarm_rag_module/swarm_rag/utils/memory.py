"""
Memory profiling utilities for GPU and CPU usage monitoring.

Provides tools to:
- Track GPU memory usage during operations
- Profile memory allocation patterns
- Detect memory leaks
- Generate memory usage reports
"""

import time
import gc
import math
import contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Generator, Tuple
from functools import wraps
import logging
import os
import torch

from .device import get_device

logger = logging.getLogger(__name__)

# Memory threshold constants - read from environment with defaults
# These are the standard thresholds used throughout the codebase for memory monitoring
MEMORY_WARNING_THRESHOLD: float = float(os.environ.get('MEMORY_WARNING_THRESHOLD', '0.70'))
MEMORY_HARD_STOP_THRESHOLD: float = float(os.environ.get('MEMORY_HARD_STOP_THRESHOLD', '0.85'))

# Optional imports
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


@dataclass
class MemorySnapshot:
    """A snapshot of memory usage at a point in time."""
    timestamp: float
    label: str

    # GPU memory (bytes)
    gpu_allocated: int = 0
    gpu_cached: int = 0
    gpu_total: int = 0

    # CPU memory (bytes)
    cpu_used: int = 0
    cpu_available: int = 0
    cpu_percent: float = 0.0

    # Process memory
    process_rss: int = 0  # Resident Set Size
    process_vms: int = 0  # Virtual Memory Size

    @property
    def gpu_used_mb(self) -> float:
        """GPU allocated memory in MB."""
        return self.gpu_allocated / (1024 * 1024)

    @property
    def gpu_cached_mb(self) -> float:
        """GPU cached (reserved) memory in MB."""
        return self.gpu_cached / (1024 * 1024)

    @property
    def gpu_free_mb(self) -> float:
        """GPU free memory in MB."""
        return (self.gpu_total - self.gpu_allocated) / (1024 * 1024)

    @property
    def cpu_used_mb(self) -> float:
        """CPU used memory in MB."""
        return self.cpu_used / (1024 * 1024)

    @property
    def process_rss_mb(self) -> float:
        """Process RSS in MB."""
        return self.process_rss / (1024 * 1024)

    def __repr__(self) -> str:
        parts = [f"MemorySnapshot('{self.label}'"]
        if self.gpu_total > 0:
            parts.append(f"GPU: {self.gpu_used_mb:.1f}MB/{self.gpu_total/(1024*1024):.1f}MB")
        if self.process_rss > 0:
            parts.append(f"Process: {self.process_rss_mb:.1f}MB")
        return ", ".join(parts) + ")"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'label': self.label,
            'gpu_allocated_bytes': self.gpu_allocated,
            'gpu_allocated_mb': self.gpu_used_mb,
            'gpu_cached_bytes': self.gpu_cached,
            'gpu_cached_mb': self.gpu_cached_mb,
            'gpu_total_bytes': self.gpu_total,
            'cpu_used_bytes': self.cpu_used,
            'cpu_used_mb': self.cpu_used_mb,
            'cpu_percent': self.cpu_percent,
            'process_rss_bytes': self.process_rss,
            'process_rss_mb': self.process_rss_mb,
            'process_vms_bytes': self.process_vms,
        }


@dataclass
class MemoryDelta:
    """Memory change between two snapshots."""
    before: MemorySnapshot
    after: MemorySnapshot
    label: str

    @property
    def gpu_delta_bytes(self) -> int:
        return self.after.gpu_allocated - self.before.gpu_allocated

    @property
    def gpu_delta_mb(self) -> float:
        return self.gpu_delta_bytes / (1024 * 1024)

    @property
    def cpu_delta_bytes(self) -> int:
        return self.after.process_rss - self.before.process_rss

    @property
    def cpu_delta_mb(self) -> float:
        return self.cpu_delta_bytes / (1024 * 1024)

    @property
    def duration_ms(self) -> float:
        return (self.after.timestamp - self.before.timestamp) * 1000

    def __repr__(self) -> str:
        parts = [f"MemoryDelta('{self.label}'"]
        if self.before.gpu_total > 0:
            sign = "+" if self.gpu_delta_mb >= 0 else ""
            parts.append(f"GPU: {sign}{self.gpu_delta_mb:.2f}MB")
        if self.before.process_rss > 0:
            sign = "+" if self.cpu_delta_mb >= 0 else ""
            parts.append(f"CPU: {sign}{self.cpu_delta_mb:.2f}MB")
        parts.append(f"{self.duration_ms:.1f}ms")
        return ", ".join(parts) + ")"


class MemoryProfiler:
    """
    Memory profiler for tracking GPU and CPU memory usage.

    Usage:
        profiler = MemoryProfiler()

        # Take snapshots
        profiler.snapshot("start")
        do_something()
        profiler.snapshot("after_operation")

        # Context manager
        with profiler.track("operation"):
            do_something()

        # Decorator
        @profiler.profile
        def my_function():
            pass

        # Report
        profiler.print_report()
    """

    def __init__(self, track_cpu: bool = True, track_gpu: bool = True, maxsize: int = 1000):
        """
        Initialize memory profiler.

        Args:
            track_cpu: Whether to track CPU/process memory
            track_gpu: Whether to track GPU memory
            maxsize: Maximum number of snapshots/deltas to keep (oldest are removed)
        """
        self.track_cpu = track_cpu and _PSUTIL_AVAILABLE
        self.track_gpu = track_gpu and get_device() != "cpu"
        self.maxsize = maxsize

        self.snapshots: List[MemorySnapshot] = []
        self.deltas: List[MemoryDelta] = []

        if self.track_cpu:
            import psutil
            self._process = psutil.Process()

    def _get_gpu_memory(self) -> tuple:
        """Get GPU memory stats."""
        if not self.track_gpu:
            return 0, 0, 0

        try:
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            return allocated, cached, total
        except Exception:
            return 0, 0, 0

    def _get_cpu_memory(self) -> tuple:
        """Get CPU memory stats."""
        if not self.track_cpu:
            return 0, 0, 0.0, 0, 0

        try:
            import psutil
            mem = psutil.virtual_memory()
            proc = self._process.memory_info()
            return mem.used, mem.available, mem.percent, proc.rss, proc.vms
        except Exception:
            return 0, 0, 0.0, 0, 0

    def snapshot(self, label: str = "") -> MemorySnapshot:
        """
        Take a memory snapshot.

        Args:
            label: Label for this snapshot

        Returns:
            MemorySnapshot with current memory state
        """
        gpu_alloc, gpu_cached, gpu_total = self._get_gpu_memory()
        cpu_used, cpu_avail, cpu_pct, proc_rss, proc_vms = self._get_cpu_memory()

        snap = MemorySnapshot(
            timestamp=time.perf_counter(),
            label=label or f"snapshot_{len(self.snapshots)}",
            gpu_allocated=gpu_alloc,
            gpu_cached=gpu_cached,
            gpu_total=gpu_total,
            cpu_used=cpu_used,
            cpu_available=cpu_avail,
            cpu_percent=cpu_pct,
            process_rss=proc_rss,
            process_vms=proc_vms
        )

        self.snapshots.append(snap)
        # Trim to maxsize to prevent unbounded growth
        if len(self.snapshots) > self.maxsize:
            self.snapshots.pop(0)
        return snap

    @contextlib.contextmanager
    def track(self, label: str) -> Generator[None, None, None]:
        """
        Context manager to track memory delta.

        Args:
            label: Label for this tracking region

        Yields:
            None

        Example:
            with profiler.track("my_operation"):
                do_something()
        """
        before = self.snapshot(f"{label}_before")
        try:
            yield
        finally:
            after = self.snapshot(f"{label}_after")
            delta = MemoryDelta(before=before, after=after, label=label)
            self.deltas.append(delta)
            # Trim to maxsize to prevent unbounded growth
            if len(self.deltas) > self.maxsize:
                self.deltas.pop(0)
            logger.debug(f"Memory: {delta}")

    def profile(self, func: Callable) -> Callable:
        """
        Decorator to profile a function's memory usage.

        Args:
            func: Function to profile

        Returns:
            Wrapped function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.track(func.__name__):
                return func(*args, **kwargs)
        return wrapper

    def clear(self):
        """Clear all snapshots and deltas."""
        self.snapshots.clear()
        self.deltas.clear()

    def get_peak_gpu_memory(self) -> float:
        """Get peak GPU memory usage in MB."""
        if not self.snapshots:
            return 0.0
        return max(s.gpu_used_mb for s in self.snapshots)

    def get_peak_cpu_memory(self) -> float:
        """Get peak process memory usage in MB."""
        if not self.snapshots:
            return 0.0
        return max(s.process_rss_mb for s in self.snapshots)

    def get_summary(self) -> Dict:
        """Get a summary of memory profiling results."""
        if not self.snapshots:
            return {}

        return {
            'n_snapshots': len(self.snapshots),
            'n_deltas': len(self.deltas),
            'peak_gpu_mb': self.get_peak_gpu_memory(),
            'peak_cpu_mb': self.get_peak_cpu_memory(),
            'final_gpu_mb': self.snapshots[-1].gpu_used_mb if self.snapshots else 0,
            'final_cpu_mb': self.snapshots[-1].process_rss_mb if self.snapshots else 0,
            'deltas': [d.gpu_delta_mb for d in self.deltas]
        }

    def print_report(self):
        """Print a formatted memory profiling report."""
        print("\n" + "=" * 60)
        print(" MEMORY PROFILING REPORT")
        print("=" * 60)

        if self.snapshots:
            print(f"\nSnapshots: {len(self.snapshots)}")
            print(f"Peak GPU Memory: {self.get_peak_gpu_memory():.2f} MB")
            print(f"Peak CPU Memory: {self.get_peak_cpu_memory():.2f} MB")

        if self.deltas:
            print(f"\nTracked Operations ({len(self.deltas)}):")
            print("-" * 60)
            print(f"{'Operation':<30} {'GPU Delta':<15} {'CPU Delta':<15} {'Time':<10}")
            print("-" * 60)

            for delta in self.deltas:
                gpu_str = f"{delta.gpu_delta_mb:+.2f} MB" if delta.before.gpu_total > 0 else "N/A"
                cpu_str = f"{delta.cpu_delta_mb:+.2f} MB" if delta.before.process_rss > 0 else "N/A"
                print(f"{delta.label:<30} {gpu_str:<15} {cpu_str:<15} {delta.duration_ms:.1f}ms")

        print("=" * 60 + "\n")


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get current GPU memory information.

    Returns:
        Dictionary with memory stats in MB
    """
    if get_device() == "cpu":
        return {}

    try:
        return {
            'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'cached_mb': torch.cuda.memory_reserved() / (1024 * 1024),
            'total_mb': torch.cuda.get_device_properties(0).total_memory / (1024 * 1024),
            'free_mb': (torch.cuda.get_device_properties(0).total_memory -
                       torch.cuda.memory_allocated()) / (1024 * 1024)
        }
    except Exception:
        return {}


def clear_gpu_cache():
    """Clear GPU memory cache."""
    if get_device() != "cpu":
        torch.cuda.empty_cache()
        gc.collect()


@contextlib.contextmanager
def memory_guard(max_gpu_mb: float = None, cleanup: bool = True) -> Generator[MemoryProfiler, None, None]:
    """
    Context manager that monitors memory and optionally enforces limits.

    Args:
        max_gpu_mb: Maximum GPU memory allowed (raises if exceeded)
        cleanup: Whether to clear GPU cache after

    Yields:
        MemoryProfiler instance

    Example:
        with memory_guard(max_gpu_mb=1000) as profiler:
            do_something()
        print(f"Peak: {profiler.get_peak_gpu_memory()}")
    """
    profiler = MemoryProfiler()
    profiler.snapshot("guard_start")

    try:
        yield profiler
    finally:
        profiler.snapshot("guard_end")

        if max_gpu_mb is not None:
            peak = profiler.get_peak_gpu_memory()
            if peak > max_gpu_mb:
                logger.warning(f"GPU memory exceeded limit: {peak:.2f}MB > {max_gpu_mb}MB")

        if cleanup:
            clear_gpu_cache()


def estimate_tensor_memory(shape: tuple, dtype: torch.dtype = None) -> float:
    """
    Estimate memory required for a tensor.

    Args:
        shape: Tensor shape
        dtype: Data type (defaults to float32)

    Returns:
        Memory in MB
    """
    if dtype is None:
        dtype = torch.float32

    size = math.prod(shape)

    # Map torch dtype to bytes per element
    dtype_to_bytes = {
        torch.float32: 4,
        torch.float64: 8,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1,
    }
    bytes_per_element = dtype_to_bytes.get(dtype, 4)
    total_bytes = size * bytes_per_element

    return total_bytes / (1024 * 1024)


def check_memory_thresholds(
    warning_pct: float = None,
    hard_stop_pct: float = None
) -> Tuple[bool, bool, float]:
    """
    Check current GPU memory against thresholds.

    Uses torch.cuda.memory_allocated() for accurate tensor usage tracking,
    not memory_reserved() which includes caching allocator overhead.

    Args:
        warning_pct: Warning threshold as ratio (0.0-1.0).
            Defaults to MEMORY_WARNING_THRESHOLD module constant.
        hard_stop_pct: Hard stop threshold as ratio (0.0-1.0).
            Defaults to MEMORY_HARD_STOP_THRESHOLD module constant.

    Returns:
        Tuple of (is_warning, is_critical, usage_pct) where:
        - is_warning: True if usage >= warning_pct
        - is_critical: True if usage >= hard_stop_pct
        - usage_pct: Current usage as ratio of total VRAM (0.0-1.0)

    Example:
        is_warn, is_crit, usage = check_memory_thresholds()
        if is_crit:
            raise MemoryError("GPU memory critical")
        elif is_warn:
            logger.warning(f"GPU memory at {usage:.1%}")
    """
    if not torch.cuda.is_available():
        return False, False, 0.0

    # Use defaults from module constants if not specified
    warning = warning_pct if warning_pct is not None else MEMORY_WARNING_THRESHOLD
    hard_stop = hard_stop_pct if hard_stop_pct is not None else MEMORY_HARD_STOP_THRESHOLD

    # Use memory_allocated for accurate tensor usage (not memory_reserved which includes cache)
    allocated = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory

    # Handle edge case of zero total memory
    if total == 0:
        return False, False, 0.0

    usage_pct = allocated / total

    return (
        usage_pct >= warning,
        usage_pct >= hard_stop,
        usage_pct
    )


__all__ = [
    'MemorySnapshot',
    'MemoryDelta',
    'MemoryProfiler',
    'get_gpu_memory_info',
    'clear_gpu_cache',
    'memory_guard',
    'estimate_tensor_memory',
    'check_memory_thresholds',
    'MEMORY_WARNING_THRESHOLD',
    'MEMORY_HARD_STOP_THRESHOLD',
]
