"""
Memory guard context manager for GPU memory threshold enforcement.

Provides tools to:
- Wrap genome evaluation code with memory tracking
- Enforce configurable memory thresholds (warning/hard stop)
- Automatically clean up CUDA cache after evaluation
- Combine torch.no_grad() with memory tracking via decorator
"""

import gc
import os
import functools
import logging
from typing import Optional, Callable, TypeVar

import torch

from ...utils.memory import MEMORY_WARNING_THRESHOLD, MEMORY_HARD_STOP_THRESHOLD

logger = logging.getLogger(__name__)

# Type variable for decorated function return type
F = TypeVar('F', bound=Callable)

# Read thresholds from environment variables with defaults
_DEFAULT_WARNING_THRESHOLD = 0.70
_DEFAULT_HARD_STOP_THRESHOLD = 0.85


class MemoryThresholdExceeded(Exception):
    """Exception raised when GPU memory exceeds the hard stop threshold.

    This exception signals that memory usage has reached a critical level
    and the current operation should be aborted to prevent OOM crashes.
    """

    def __init__(
        self,
        message: str,
        usage_ratio: float = 0.0,
        threshold: float = 0.0,
        delta_bytes: int = 0,
        label: str = ""
    ):
        """
        Initialize MemoryThresholdExceeded exception.

        Args:
            message: Human-readable error message
            usage_ratio: Current memory usage as ratio of total VRAM
            threshold: The threshold that was exceeded
            delta_bytes: Memory change during the guarded operation
            label: Label identifying the operation that exceeded the threshold
        """
        super().__init__(message)
        self.usage_ratio = usage_ratio
        self.threshold = threshold
        self.delta_bytes = delta_bytes
        self.label = label


class MemoryGuard:
    """
    Context manager that enforces GPU memory thresholds during evaluation.

    Wraps code sections to track memory usage and enforce configurable
    warning/hard-stop thresholds. On exit, optionally clears CUDA cache.

    The guard tracks:
    - Memory allocated before and after the guarded section
    - Delta (change) in memory allocation
    - Current usage ratio relative to total VRAM

    Thresholds:
    - Warning threshold (default 70%): Logs a warning when exceeded
    - Hard stop threshold (default 85%): Raises MemoryThresholdExceeded

    Example:
        with MemoryGuard(warning_threshold=0.70, hard_stop_threshold=0.85) as guard:
            # Evaluation code here
            result = evaluate_genome(genome)

        print(f"Memory delta: {guard.delta_mb:.2f} MB")

    Environment Variables:
        MEMORY_WARNING_THRESHOLD: Override default warning threshold
        MEMORY_HARD_STOP_THRESHOLD: Override default hard stop threshold
    """

    def __init__(
        self,
        warning_threshold: Optional[float] = None,
        hard_stop_threshold: Optional[float] = None,
        cleanup_on_exit: bool = True,
        label: str = ""
    ):
        """
        Initialize MemoryGuard.

        Args:
            warning_threshold: VRAM ratio (0.0-1.0) to trigger warning log.
                Defaults to MEMORY_WARNING_THRESHOLD env var or 0.70.
            hard_stop_threshold: VRAM ratio (0.0-1.0) to raise exception.
                Defaults to MEMORY_HARD_STOP_THRESHOLD env var or 0.85.
            cleanup_on_exit: Whether to call gc.collect() and torch.cuda.empty_cache()
                on context exit. Default True.
            label: Identifier for this guard instance, used in log messages
                and exception details.
        """
        # Read defaults from environment
        env_warning = os.environ.get('MEMORY_WARNING_THRESHOLD')
        env_hard_stop = os.environ.get('MEMORY_HARD_STOP_THRESHOLD')

        default_warning = float(env_warning) if env_warning else _DEFAULT_WARNING_THRESHOLD
        default_hard_stop = float(env_hard_stop) if env_hard_stop else _DEFAULT_HARD_STOP_THRESHOLD

        self.warning_threshold = warning_threshold if warning_threshold is not None else default_warning
        self.hard_stop_threshold = hard_stop_threshold if hard_stop_threshold is not None else default_hard_stop
        self.cleanup_on_exit = cleanup_on_exit
        self.label = label

        # Internal state
        self._before_allocated: int = 0
        self._after_allocated: int = 0
        self._total_vram: int = 0
        self._cuda_available: bool = False

    def __enter__(self) -> "MemoryGuard":
        """Enter the memory guard context.

        Records current CUDA memory allocation and total VRAM.

        Returns:
            self for use in 'with ... as guard:' pattern
        """
        self._cuda_available = torch.cuda.is_available()

        if self._cuda_available:
            self._before_allocated = torch.cuda.memory_allocated()
            self._total_vram = torch.cuda.get_device_properties(0).total_memory
        else:
            self._before_allocated = 0
            self._total_vram = 0

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit the memory guard context.

        Checks memory usage against thresholds and performs cleanup.

        Args:
            exc_type: Exception type if raised inside context
            exc_val: Exception value if raised inside context
            exc_tb: Exception traceback if raised inside context

        Returns:
            False (do not suppress exceptions)

        Raises:
            MemoryThresholdExceeded: If hard stop threshold is exceeded
        """
        if not self._cuda_available:
            return False

        self._after_allocated = torch.cuda.memory_allocated()
        delta = self._after_allocated - self._before_allocated

        # Calculate usage ratio (handle zero total_vram edge case)
        if self._total_vram > 0:
            usage_ratio = self._after_allocated / self._total_vram
        else:
            usage_ratio = 0.0

        # Check hard stop threshold
        if usage_ratio >= self.hard_stop_threshold:
            # Perform cleanup before raising
            if self.cleanup_on_exit:
                gc.collect()
                torch.cuda.empty_cache()

            label_str = f" [{self.label}]" if self.label else ""
            raise MemoryThresholdExceeded(
                f"Hard stop{label_str}: GPU memory usage {usage_ratio:.1%} "
                f">= {self.hard_stop_threshold:.1%} threshold. "
                f"Delta: {delta / (1024 * 1024):.2f} MB",
                usage_ratio=usage_ratio,
                threshold=self.hard_stop_threshold,
                delta_bytes=delta,
                label=self.label
            )

        # Check warning threshold
        if usage_ratio >= self.warning_threshold:
            label_str = f" [{self.label}]" if self.label else ""
            logger.warning(
                f"Memory warning{label_str}: GPU memory usage {usage_ratio:.1%} "
                f">= {self.warning_threshold:.1%} threshold. "
                f"Delta: {delta / (1024 * 1024):.2f} MB"
            )

        # Cleanup logic:
        # - empty_cache() is fast and always safe to call
        # - gc.collect() is slow (can take 30+ seconds under memory pressure)
        #   so only call it when memory usage is concerning (above 50% of warning threshold)
        if self.cleanup_on_exit:
            # Always clear CUDA cache (fast operation)
            torch.cuda.empty_cache()

            # Only run expensive gc.collect when memory is elevated
            # This prevents long pauses during normal operation
            gc_threshold = self.warning_threshold * 0.7  # e.g., 49% if warning is 70%
            if usage_ratio >= gc_threshold:
                gc.collect()

        return False

    @property
    def delta_bytes(self) -> int:
        """Memory change in bytes (after - before).

        Returns 0 if accessed before context exit.
        """
        return self._after_allocated - self._before_allocated

    @property
    def delta_mb(self) -> float:
        """Memory change in megabytes.

        Returns 0.0 if accessed before context exit.
        """
        return self.delta_bytes / (1024 * 1024)

    @property
    def usage_ratio(self) -> float:
        """Current memory usage as ratio of total VRAM.

        Returns the ratio after context exit, or current ratio if still inside.
        Returns 0.0 if CUDA is not available.
        """
        if not self._cuda_available or self._total_vram == 0:
            return 0.0

        if self._after_allocated > 0:
            return self._after_allocated / self._total_vram
        else:
            # Still inside context, return current
            return torch.cuda.memory_allocated() / self._total_vram


def evaluation_no_grad(
    track_memory: bool = True,
    warning_threshold: Optional[float] = None,
    hard_stop_threshold: Optional[float] = None
) -> Callable[[F], F]:
    """
    Decorator for genome evaluation functions.

    Combines torch.no_grad() context with optional MemoryGuard tracking.
    All operations inside the decorated function run without gradient computation,
    and memory usage is tracked and cleaned up after execution.

    Args:
        track_memory: Whether to use MemoryGuard for memory tracking.
            Default True. Set False to use only torch.no_grad().
        warning_threshold: VRAM ratio (0.0-1.0) to trigger warning log.
            Defaults to MEMORY_WARNING_THRESHOLD from utils.memory.
        hard_stop_threshold: VRAM ratio (0.0-1.0) to raise MemoryThresholdExceeded.
            Defaults to MEMORY_HARD_STOP_THRESHOLD from utils.memory.

    Returns:
        Decorator function that wraps the target function.

    Example:
        @evaluation_no_grad(track_memory=True)
        def evaluate_genome(genome, retriever, ...):
            # All operations are gradient-free
            # Memory is tracked and cleaned up
            results = retriever.retrieve_batch(...)
            return compute_fitness(results)

        @evaluation_no_grad(track_memory=False)
        def quick_eval(genome):
            # Just no_grad, no memory tracking
            return genome.score()

    Raises:
        MemoryThresholdExceeded: If hard stop threshold is exceeded during execution.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with torch.no_grad():
                if track_memory and torch.cuda.is_available():
                    warn_thresh = warning_threshold if warning_threshold is not None else MEMORY_WARNING_THRESHOLD
                    stop_thresh = hard_stop_threshold if hard_stop_threshold is not None else MEMORY_HARD_STOP_THRESHOLD
                    with MemoryGuard(
                        warning_threshold=warn_thresh,
                        hard_stop_threshold=stop_thresh,
                        cleanup_on_exit=True,
                        label=func.__name__
                    ):
                        return func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


__all__ = [
    'MemoryGuard',
    'MemoryThresholdExceeded',
    'evaluation_no_grad',
]
