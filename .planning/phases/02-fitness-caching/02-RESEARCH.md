# Phase 2: Fitness Caching - Research

**Researched:** 2026-01-29
**Domain:** Content-hash based fitness memoization for evolutionary algorithms
**Confidence:** HIGH

## Summary

Fitness caching for evolutionary genomes involves computing a content-hash of genome configuration and using it as a dictionary key to store/retrieve fitness scores. This is a well-established pattern in evolutionary computing to avoid redundant evaluation of duplicate or elite genomes.

The user decisions constrain the implementation to:
- **xxhash** for fast, non-cryptographic hashing (user chose "xxhash or cityhash")
- **In-memory only** cache that lives within a single evolution run
- **Per-generation logging** integrated with existing memory logger
- **Unlimited cache size** (fitness floats are negligible)

The primary challenge is ensuring the hash correctly captures ALL genome fields that affect retrieval behavior (params, strategies, group_ratios, weight_tensors) while handling floating-point precision issues.

**Primary recommendation:** Use xxhash (xxh64) with a canonicalized JSON serialization of genome config, rounding floats to 6 decimal places. Integrate cache as a thin layer around PopulationEvaluator.evaluate(), with per-generation stats logged alongside memory stats.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| xxhash | 3.6.0 | Fast non-cryptographic hashing | 5x faster than CityHash, supports incremental hashing, actively maintained |
| json | stdlib | Canonical serialization | Built-in, deterministic with sort_keys=True |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| logging | stdlib | Per-generation cache stats | Already used by MemoryLogger |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| xxhash | cityhash | CityHash lacks incremental hashing, inactive maintenance |
| json serialization | pickle + hash | Pickle not guaranteed deterministic across runs |
| dict cache | functools.lru_cache | lru_cache requires hashable args, genome objects not hashable |

**Installation:**
```bash
pip install xxhash
```

## Architecture Patterns

### Recommended Integration Point

```
PopulationEvaluator.evaluate(population)
    |
    +-> FitnessCache.get_or_evaluate(genome)
            |
            +-> cache hit: return cached fitness
            +-> cache miss: evaluate, store, return
```

The cache should wrap the evaluation flow INSIDE `PopulationEvaluator.evaluate()`, not at a higher level. This ensures:
1. All evaluation paths (adaptive, non-adaptive, shared context) benefit
2. Cache stats can be collected per-generation
3. Elite genomes carried forward naturally hit cache

### Recommended Module Structure
```
swarm_rag/evolution/execution/
    fitness_cache.py        # FitnessCache class + hash_genome function
    evaluator.py            # Modified to use FitnessCache
    memory_logger.py        # Extended to log cache stats
```

### Pattern 1: Content-Hash Function
**What:** Deterministic hash of all genome fields that affect fitness
**When to use:** Every cache lookup/store operation
**Example:**
```python
# Source: xxhash PyPI documentation
import xxhash
import json

def hash_genome(genome: Genome) -> str:
    """
    Compute content hash for genome fitness caching.

    Hashes ALL fields that affect retrieval behavior:
    - params (hyperparameters)
    - mode (expression_tree vs weighted_sum)
    - group_ratios (agent distribution)
    - strategies (expression trees, serialized)
    - weight_tensors (weighted sum mode)

    Floats rounded to 6 decimals to handle FP noise.
    """
    def round_floats(obj, decimals=6):
        """Recursively round floats in nested structures."""
        if isinstance(obj, float):
            return round(obj, decimals)
        elif isinstance(obj, dict):
            return {k: round_floats(v, decimals) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [round_floats(v, decimals) for v in obj]
        return obj

    # Build canonical representation
    config = {
        "mode": genome.mode.value,
        "params": round_floats(genome.params),
        "group_ratios": round_floats(genome.group_ratios),
    }

    # Add mode-specific fields
    if genome.mode == GenomeMode.EXPRESSION_TREE and genome.strategies:
        config["strategies"] = {
            k: v.to_dict() for k, v in sorted(genome.strategies.items())
        }
    if genome.mode == GenomeMode.WEIGHTED_SUM and genome.weight_tensors:
        config["weight_tensors"] = genome.weight_tensors.to_dict()

    # Canonical JSON (sorted keys, no whitespace)
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':'))

    # xxh64 returns 64-bit hash as hex string
    return xxhash.xxh64_hexdigest(canonical.encode('utf-8'))
```

### Pattern 2: Cache Class with Stats
**What:** Simple dict-based cache with per-generation statistics
**When to use:** Wrapped around evaluation
**Example:**
```python
from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Per-generation cache statistics."""
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0

    def reset(self):
        self.hits = 0
        self.misses = 0


class FitnessCache:
    """
    In-memory fitness cache for evolution runs.

    - Lives within single run (no persistence)
    - Stores only fitness score (single float per entry)
    - Thread-safe not required (evolution is synchronous)
    """

    def __init__(self):
        self._cache: Dict[str, float] = {}
        self._generation_stats = CacheStats()
        self._total_stats = CacheStats()

    def get(self, genome: Genome) -> Optional[float]:
        """Look up cached fitness for genome."""
        key = hash_genome(genome)
        if key in self._cache:
            self._generation_stats.hits += 1
            self._total_stats.hits += 1
            return self._cache[key]
        self._generation_stats.misses += 1
        self._total_stats.misses += 1
        return None

    def put(self, genome: Genome, fitness_score: float):
        """Store fitness for genome."""
        key = hash_genome(genome)
        self._cache[key] = fitness_score

    def finalize_generation(self, generation: int):
        """Log stats and reset per-generation counters."""
        stats = self._generation_stats
        logger.info(
            f"Gen {generation}: {stats.hits}/{stats.total} cache hits "
            f"({stats.hit_rate:.0%})"
        )
        self._generation_stats.reset()

    @property
    def size(self) -> int:
        return len(self._cache)
```

### Anti-Patterns to Avoid
- **Hashing genome.id:** IDs change on copy(), would miss duplicates
- **Using pickle for serialization:** Non-deterministic across Python versions
- **Caching full FitnessResult:** Only quality_score needed, stability_score is deterministic from metrics
- **Adding thread locks:** Evolution loop is synchronous, locks add overhead
- **Disk persistence:** Violates user decision "in-memory only"

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Fast hashing | Custom hash function | xxhash.xxh64_hexdigest | Battle-tested, C implementation, 10GB/s throughput |
| Canonical serialization | Custom serializer | json.dumps(sort_keys=True) | Deterministic, handles all Python types |
| Float precision handling | Manual epsilon comparison | round(x, 6) | Simple, deterministic, covers FP noise |

**Key insight:** The complexity is in correctly identifying WHAT to hash, not HOW to hash it. Genome's to_dict() method already exists but includes non-config fields (metrics, fitness, evaluated). A targeted hash function is needed.

## Common Pitfalls

### Pitfall 1: Incomplete Hash Coverage
**What goes wrong:** Cache returns stale fitness for modified genome
**Why it happens:** Forgot to include a field that affects behavior (e.g., weight_tensors)
**How to avoid:** Hash ALL fields from genome that affect retrieval:
  - params (all hyperparameters)
  - mode (expression_tree vs weighted_sum)
  - group_ratios (agent distribution)
  - strategies (expression trees)
  - weight_tensors (weighted sum weights)
**Warning signs:** Fitness values don't change when they should

### Pitfall 2: Float Precision Mismatch
**What goes wrong:** Semantically identical genomes hash differently
**Why it happens:** Floating-point arithmetic produces slightly different values
**How to avoid:** Round all floats to fixed precision (6 decimals) before hashing
**Warning signs:** Cache hit rate is 0% despite duplicates existing

### Pitfall 3: Hashing Non-Deterministic Fields
**What goes wrong:** Same genome hashes differently on each call
**Why it happens:** Including timestamp, random seed, or id in hash
**How to avoid:** Only hash configuration fields, not metadata
**Warning signs:** Genome.copy() produces different hash than original

### Pitfall 4: Caching Too Much Data
**What goes wrong:** Memory grows unexpectedly
**Why it happens:** Storing entire FitnessResult or genome reference
**How to avoid:** Store only the float fitness score (8 bytes per entry)
**Warning signs:** Memory stats show growth despite user decision "unlimited cache"

### Pitfall 5: Missing Detached Tensor Handling
**What goes wrong:** Gradient graph retained, GPU memory grows
**Why it happens:** Caching tensor with gradient history
**How to avoid:** Cache only float values, not tensors. Fitness score is already a float.
**Warning signs:** CUDA out of memory despite fitness values being cached

## Code Examples

### Evaluator Integration Pattern
```python
# Source: Codebase analysis of evaluator.py

class PopulationEvaluator:
    def __init__(self, ...):
        # ... existing init ...
        self._fitness_cache = FitnessCache()

    def evaluate(self, population: List[Genome], ...) -> EvaluationStats:
        """Modified to use fitness cache."""
        # ... existing setup ...

        unevaluated = []
        for genome in population:
            if genome.evaluated:
                continue

            # Check cache first
            cached_fitness = self._fitness_cache.get(genome)
            if cached_fitness is not None:
                # Cache hit - restore fitness without evaluation
                genome.fitness = FitnessResult(quality_score=cached_fitness)
                genome.evaluated = True
            else:
                unevaluated.append(genome)

        # Evaluate only cache misses
        if unevaluated:
            # ... existing evaluation logic ...

            # Store results in cache
            for genome in unevaluated:
                self._fitness_cache.put(genome, genome.fitness.quality_score)

        return self.stats

    def finalize_generation(self, generation: int):
        """Call at end of generation to log cache stats."""
        self._fitness_cache.finalize_generation(generation)
```

### Memory Logger Extension
```python
# Source: Codebase analysis of memory_logger.py

class MemoryLogger:
    def log_generation(self, generation: int, cache_stats: CacheStats = None):
        """Extended to include cache stats."""
        # ... existing memory logging ...

        if cache_stats is not None:
            line = (
                f"gen={generation:04d} "
                f"cache_hits={cache_stats.hits} "
                f"cache_total={cache_stats.total} "
                f"cache_rate={cache_stats.hit_rate:.0%}"
            )
            with open(self._cache_log_path, 'a') as f:
                f.write(line + "\n")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No caching | Content-hash caching | Standard practice | 30-70% eval savings for typical runs |
| MD5/SHA hashing | xxHash | ~2015 | 10x speed improvement |
| Full object serialization | Targeted field selection | Best practice | Smaller hash input, faster hashing |

**Deprecated/outdated:**
- CityHash Python bindings: Inactive maintenance, last release 12+ months ago
- pickle-based hashing: Non-deterministic, security issues

## Open Questions

1. **Metric Caching**
   - What we know: User decision says "store just the fitness score"
   - What's unclear: Should we also cache full metrics dict for LLM context?
   - Recommendation: Start with fitness-only per user decision; can extend later if needed

2. **Hash Collision Risk**
   - What we know: xxh64 has ~1e-19 collision probability for typical workloads
   - What's unclear: Is this acceptable for scientific reproducibility?
   - Recommendation: 64-bit hash is sufficient; upgrade to xxh128 only if paranoid

## Sources

### Primary (HIGH confidence)
- [xxhash PyPI](https://pypi.org/project/xxhash/) - v3.6.0 API, installation
- [python-xxhash GitHub](https://github.com/ifduyue/python-xxhash) - Usage patterns
- Codebase: `genome.py`, `evaluator.py`, `memory_logger.py`, `config.py` - Integration points

### Secondary (MEDIUM confidence)
- [CityHash vs xxHash comparison](https://ssojet.com/compare-hashing-algorithms/cityhash-vs-xxhash/) - Performance data
- [LRU Cache Python](https://realpython.com/lru-cache-python/) - Memoization patterns

### Tertiary (LOW confidence)
- [Fitness caching ACM paper](https://dl.acm.org/doi/10.1145/3205651.3205788) - Academic validation of approach

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - xxhash is widely used, well-documented
- Architecture: HIGH - Based on codebase analysis, integration points clear
- Pitfalls: HIGH - Standard evolutionary computing concerns, verified against codebase

**Research date:** 2026-01-29
**Valid until:** 60 days (stable domain, no rapid changes expected)
