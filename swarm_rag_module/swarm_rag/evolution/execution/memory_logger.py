"""
Per-generation memory logger for evolution runs.

Creates dedicated memory.log file and tracks stats across generations.
Enables visibility into memory behavior so developers can detect
accumulation patterns before they cause OOM crashes.
"""
import json
import os
import time
import logging
import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .fitness_cache import CacheStats

logger = logging.getLogger(__name__)

__all__ = ['MemoryLogger', 'GenerationMemoryStats']


@dataclass
class GenerationMemoryStats:
    """Memory statistics for a single generation."""
    generation: int
    timestamp: float
    allocated_mb: float      # torch.cuda.memory_allocated() / MB
    cached_mb: float         # torch.cuda.memory_reserved() / MB
    peak_mb: float           # torch.cuda.max_memory_allocated() / MB
    delta_mb: float          # Change from previous generation
    total_vram_mb: float     # Total GPU memory
    # Fitness cache stats (optional)
    cache_hits: int = 0
    cache_total: int = 0

    @property
    def usage_ratio(self) -> float:
        """Current allocation as ratio of total VRAM."""
        return self.allocated_mb / self.total_vram_mb if self.total_vram_mb > 0 else 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as ratio (0.0 to 1.0)."""
        return self.cache_hits / self.cache_total if self.cache_total > 0 else 0.0

    def to_log_line(self) -> str:
        """Format as single log line."""
        line = (
            f"gen={self.generation:04d} "
            f"alloc={self.allocated_mb:7.1f}MB "
            f"cached={self.cached_mb:7.1f}MB "
            f"peak={self.peak_mb:7.1f}MB "
            f"delta={self.delta_mb:+7.1f}MB "
            f"usage={self.usage_ratio:.1%}"
        )
        # Append cache stats if present
        if self.cache_total > 0:
            line += f" cache={self.cache_hits}/{self.cache_total}({self.cache_hit_rate:.0%})"
        return line

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        d = {
            'generation': self.generation,
            'timestamp': self.timestamp,
            'allocated_mb': self.allocated_mb,
            'cached_mb': self.cached_mb,
            'peak_mb': self.peak_mb,
            'delta_mb': self.delta_mb,
            'total_vram_mb': self.total_vram_mb,
            'usage_ratio': self.usage_ratio,
        }
        # Include cache stats if present
        if self.cache_total > 0:
            d['cache_hits'] = self.cache_hits
            d['cache_total'] = self.cache_total
            d['cache_hit_rate'] = self.cache_hit_rate
        return d


class MemoryLogger:
    """
    Per-generation memory logger for evolution runs.

    Creates dedicated memory.log file and tracks stats across generations.
    """

    def __init__(self, log_dir: str, warning_threshold: float = 0.70):
        """
        Initialize memory logger.

        Args:
            log_dir: Directory to write memory.log
            warning_threshold: Ratio at which to log warnings (default 0.70)
        """
        self.log_dir = log_dir
        self.warning_threshold = warning_threshold
        self._log_path = os.path.join(log_dir, "memory.log")
        self._prev_allocated_mb = 0.0
        self._stats_history: List[GenerationMemoryStats] = []
        self._total_vram_mb = self._get_total_vram_mb()

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Initialize log file with header
        with open(self._log_path, 'w') as f:
            f.write(f"# Memory Log - Started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total VRAM: {self._total_vram_mb:.1f} MB\n")
            f.write(f"# Warning threshold: {self.warning_threshold:.0%}\n")
            f.write("#\n")

    def _get_total_vram_mb(self) -> float:
        """Get total GPU VRAM in MB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

    def log_generation(
        self,
        generation: int,
        cache_stats: Optional['CacheStats'] = None
    ) -> GenerationMemoryStats:
        """
        Log memory stats for a generation.

        Call at the START of each generation to track memory before/after.

        Args:
            generation: Generation number
            cache_stats: Optional fitness cache stats for this generation

        Returns:
            GenerationMemoryStats for this generation
        """
        # Extract cache stats if provided
        cache_hits = cache_stats.hits if cache_stats else 0
        cache_total = cache_stats.total if cache_stats else 0

        if not torch.cuda.is_available():
            # Return placeholder stats for CPU mode
            stats = GenerationMemoryStats(
                generation=generation,
                timestamp=time.time(),
                allocated_mb=0.0,
                cached_mb=0.0,
                peak_mb=0.0,
                delta_mb=0.0,
                total_vram_mb=0.0,
                cache_hits=cache_hits,
                cache_total=cache_total,
            )
            self._stats_history.append(stats)
            return stats

        allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        cached_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        delta_mb = allocated_mb - self._prev_allocated_mb

        stats = GenerationMemoryStats(
            generation=generation,
            timestamp=time.time(),
            allocated_mb=allocated_mb,
            cached_mb=cached_mb,
            peak_mb=peak_mb,
            delta_mb=delta_mb,
            total_vram_mb=self._total_vram_mb,
            cache_hits=cache_hits,
            cache_total=cache_total,
        )

        # Write to log file
        with open(self._log_path, 'a') as f:
            f.write(stats.to_log_line() + "\n")

        # Log warning if approaching threshold
        if stats.usage_ratio >= self.warning_threshold:
            logger.warning(
                f"Memory warning at gen {generation}: "
                f"{stats.usage_ratio:.1%} >= {self.warning_threshold:.0%}"
            )

        # Update state
        self._prev_allocated_mb = allocated_mb
        self._stats_history.append(stats)

        # Reset peak for next generation
        torch.cuda.reset_peak_memory_stats()

        return stats

    def get_trend(self, window: int = 10) -> float:
        """
        Get memory growth trend over recent generations.

        Returns average delta_mb over last `window` generations.
        Positive = growing, negative = shrinking.
        """
        if len(self._stats_history) < 2:
            return 0.0
        recent = self._stats_history[-window:]
        deltas = [s.delta_mb for s in recent]
        return sum(deltas) / len(deltas)

    def export_stats(self, path: Optional[str] = None) -> str:
        """Export stats history to JSON file."""
        path = path or os.path.join(self.log_dir, "memory_stats.json")
        with open(path, 'w') as f:
            json.dump([s.to_dict() for s in self._stats_history], f, indent=2)
        return path

    @property
    def stats_history(self) -> List[GenerationMemoryStats]:
        """Access to recorded stats history."""
        return self._stats_history
