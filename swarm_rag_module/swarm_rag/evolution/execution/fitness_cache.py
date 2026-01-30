"""
Fitness caching for genome evaluation.

Content-hash based caching to skip redundant genome evaluations.
Duplicate genomes and elite genomes carried forward hit cache
instead of re-evaluating.

Design decisions (from CONTEXT.md):
- In-memory only (no disk persistence)
- Store just fitness score (single float per entry)
- No thread locks (evolution is synchronous)
- Unlimited cache size (fitness floats are tiny, ~25K max entries)
"""
import json
import xxhash
from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..types.genome import Genome

__all__ = ['FitnessCache', 'CacheStats', 'hash_genome']


def hash_genome(genome: 'Genome') -> str:
    """
    Compute a deterministic hash for a genome based on config fields.

    Hashes ALL fields that affect retrieval behavior:
    - mode (GenomeMode enum -> .value string)
    - params (all hyperparameters)
    - group_ratios (agent distribution)
    - strategies (expression trees, via to_dict())
    - weight_tensors (weighted sum mode, via to_dict())

    Does NOT hash (metadata, not config):
    - id, fitness, metrics, evaluated, latency

    Floats are rounded to 6 decimal places to handle FP noise.

    Args:
        genome: Genome instance to hash

    Returns:
        xxhash64 hex digest string
    """
    # Build canonical dictionary of config fields
    config = {}

    # Mode (enum -> string)
    config['mode'] = genome.mode.value if hasattr(genome.mode, 'value') else str(genome.mode)

    # Params - round floats for FP stability
    config['params'] = _round_dict(genome.params)

    # Group ratios - round floats
    config['group_ratios'] = _round_dict(genome.group_ratios)

    # Strategies (expression trees)
    if genome.strategies:
        config['strategies'] = {
            k: v.to_dict() for k, v in sorted(genome.strategies.items())
        }
    else:
        config['strategies'] = {}

    # Weight tensors (weighted sum mode)
    if genome.weight_tensors is not None:
        config['weight_tensors'] = genome.weight_tensors.to_dict()
    else:
        config['weight_tensors'] = None

    # Canonical JSON serialization (sorted keys, compact)
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':'))

    # xxhash64 for speed
    return xxhash.xxh64_hexdigest(canonical.encode('utf-8'))


def _round_dict(d: Dict) -> Dict:
    """
    Recursively round floats in a dictionary to 6 decimal places.

    Handles nested dicts and lists.
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, float):
            result[k] = round(v, 6)
        elif isinstance(v, dict):
            result[k] = _round_dict(v)
        elif isinstance(v, list):
            result[k] = _round_list(v)
        else:
            result[k] = v
    return result


def _round_list(lst: list) -> list:
    """Recursively round floats in a list."""
    result = []
    for item in lst:
        if isinstance(item, float):
            result.append(round(item, 6))
        elif isinstance(item, dict):
            result.append(_round_dict(item))
        elif isinstance(item, list):
            result.append(_round_list(item))
        else:
            result.append(item)
    return result


@dataclass
class CacheStats:
    """Statistics for cache performance tracking."""
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        """Total lookups (hits + misses)."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as ratio (0.0 to 1.0)."""
        if self.total == 0:
            return 0.0
        return self.hits / self.total

    def reset(self) -> None:
        """Reset statistics to zero."""
        self.hits = 0
        self.misses = 0

    def copy(self) -> 'CacheStats':
        """Create a copy of the stats."""
        return CacheStats(hits=self.hits, misses=self.misses)


class FitnessCache:
    """
    In-memory fitness cache using content-based genome hashing.

    Stores fitness scores (single floats) keyed by genome config hash.
    Duplicate genomes (same config, different id) share cached fitness.

    Thread safety: Not thread-safe (evolution is synchronous).
    Memory: Unlimited size (floats are tiny, ~25K max entries typical).

    Usage:
        cache = FitnessCache()

        # Before evaluation
        cached = cache.get(genome)
        if cached is not None:
            genome.fitness = cached
        else:
            # ... evaluate genome ...
            cache.put(genome, fitness_score)

        # End of generation
        stats = cache.finalize_generation(gen_num)
        print(f"Cache: {stats.hits}/{stats.total} ({stats.hit_rate:.1%})")
    """

    def __init__(self):
        """Initialize empty cache."""
        self._cache: Dict[str, float] = {}
        self._generation_stats = CacheStats()
        self._total_stats = CacheStats()

    def get(self, genome: 'Genome') -> Optional[float]:
        """
        Look up cached fitness for a genome.

        Args:
            genome: Genome to look up

        Returns:
            Cached fitness score if found, None otherwise
        """
        key = hash_genome(genome)
        value = self._cache.get(key)

        if value is not None:
            self._generation_stats.hits += 1
            self._total_stats.hits += 1
        else:
            self._generation_stats.misses += 1
            self._total_stats.misses += 1

        return value

    def put(self, genome: 'Genome', fitness_score: float) -> None:
        """
        Store fitness for a genome.

        Args:
            genome: Genome that was evaluated
            fitness_score: Quality score to cache (float, detached from any gradient graph)
        """
        key = hash_genome(genome)
        # Store as plain float (no tensor, no gradient)
        self._cache[key] = float(fitness_score)

    def finalize_generation(self, generation: int) -> CacheStats:
        """
        Finalize stats for current generation and reset per-generation counters.

        Args:
            generation: Generation number (for logging)

        Returns:
            Copy of per-generation stats before reset
        """
        stats = self._generation_stats.copy()
        self._generation_stats.reset()
        return stats

    @property
    def size(self) -> int:
        """Number of entries in cache."""
        return len(self._cache)

    @property
    def total_stats(self) -> CacheStats:
        """Cumulative stats across all generations."""
        return self._total_stats

    def clear(self) -> None:
        """Clear all cached entries and reset stats."""
        self._cache.clear()
        self._generation_stats.reset()
        self._total_stats.reset()
