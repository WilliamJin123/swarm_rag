"""
Unified cache statistics protocol.

Defines a common interface that both FitnessCache stats (CacheStats)
and EmbeddingCache stats (EmbeddingCacheStats) conform to, enabling
generic code to work with either cache type.
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class CacheStatsProtocol(Protocol):
    """
    Protocol defining the minimal cache statistics interface.

    Both CacheStats (fitness cache) and EmbeddingCacheStats (embedding cache)
    conform to this protocol, allowing generic cache monitoring code.

    Required attributes/properties:
        hits: Total cache hits (int)
        misses: Total cache misses (int)
        hit_rate: Cache hit rate as float 0.0-1.0 (property)
    """

    @property
    def hits(self) -> int:
        """Total cache hits."""
        ...

    @property
    def misses(self) -> int:
        """Total cache misses."""
        ...

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as ratio (0.0 to 1.0)."""
        ...
