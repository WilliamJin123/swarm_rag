"""Cache coordination for genome evaluation.

Manages the lifecycle of fitness and embedding caches during evolution.
Centralizes cache lookup, storage, generation finalization, and cleanup
that was previously scattered across PopulationEvaluator methods.
"""
import logging
from typing import Optional
from dataclasses import dataclass

from .fitness_cache import FitnessCache, CacheStats
from .embedding_cache import EmbeddingCacheProvider

logger = logging.getLogger(__name__)


@dataclass
class GenerationCacheStats:
    """Combined cache statistics for a generation."""
    fitness_stats: CacheStats
    embedding_stats: Optional[object] = None  # EmbeddingCacheStats when available


class CacheCoordinator:
    """Coordinates fitness and embedding caches during evaluation.

    Provides a single interface for:
    - Looking up cached fitness scores before evaluation
    - Storing fitness scores after evaluation
    - Finalizing generation statistics
    - Cleanup at evolution end
    """

    def __init__(self):
        self._fitness_cache = FitnessCache()

    @property
    def fitness_cache(self) -> FitnessCache:
        """Access the underlying fitness cache (for backward compatibility)."""
        return self._fitness_cache

    def lookup_fitness(self, genome) -> Optional[float]:
        """Look up cached fitness for a genome.

        Returns:
            Cached fitness score, or None if not cached.
        """
        return self._fitness_cache.get(genome)

    def store_fitness(self, genome, quality_score: float):
        """Store fitness score in cache after evaluation."""
        self._fitness_cache.put(genome, quality_score)

    def finalize_generation(self, generation: int) -> GenerationCacheStats:
        """Finalize caches for a generation and return combined stats.

        Logs cache performance for both fitness and embedding caches.
        """
        fitness_stats = self._fitness_cache.finalize_generation(generation)
        logger.info(f"  > Cache: {fitness_stats.hits}/{fitness_stats.total} hits ({fitness_stats.hit_rate:.1%})")

        embed_cache = EmbeddingCacheProvider.get()
        embed_stats = None
        if embed_cache is not None:
            embed_stats = embed_cache.finalize_generation(generation)
            logger.info(
                f"  > Embedding cache: {embed_stats.generation_hits + embed_stats.generation_misses} lookups, "
                f"{embed_stats.compute_time_saved_sec:.1f}s saved"
            )

        return GenerationCacheStats(
            fitness_stats=fitness_stats,
            embedding_stats=embed_stats,
        )

    def cleanup(self):
        """Release all cache resources at evolution end."""
        EmbeddingCacheProvider.clear()
        logger.info("CacheCoordinator cleanup: embedding cache cleared")
