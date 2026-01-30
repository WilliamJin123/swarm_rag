# Phase 3: Embedding Cache - Research

**Researched:** 2026-01-30
**Domain:** Cross-generation query embedding persistence for evolutionary optimization
**Confidence:** HIGH

## Summary

This phase extends the existing `QueryEmbeddingCache` to persist query embeddings across multiple generations within an evolution run. Currently, `SharedPrecomputeContext` computes query embeddings once per generation, but these are discarded and recomputed each generation. By persisting embeddings for the entire evolution run, we eliminate redundant embedding computation and achieve the target 50-80% retrieval savings.

The codebase already has most of the infrastructure in place:
- `QueryEmbeddingCache` class with LRU eviction, device management, and batch operations
- `EmbeddingCacheProvider` singleton for global cache access
- `SharedPrecomputeContext` that prepares query embeddings per generation
- Established device mode pattern (`get_device()`) used throughout

The primary work is:
1. Extend cache lifecycle to span entire evolution run (not just single generation)
2. Add per-generation and end-of-run statistics tracking (mirroring FitnessCache pattern)
3. Integrate stats with MemoryLogger (same pattern as Phase 2)
4. Add time-saved calculation and debug dump capability

**Primary recommendation:** Modify `QueryEmbeddingCache` to add generation-aware stats tracking and evolution-lifecycle management. Integrate with `SharedPrecomputeContext.prepare_shared_context()` to check cache before recomputing. Use the `EmbeddingCacheProvider` singleton to maintain cache across generations.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.x | Tensor storage, device management | Already used throughout, GPU-native |
| dataclasses | stdlib | Stats structures | Matches FitnessCache pattern |
| time | stdlib | Compute time tracking | For time-saved calculation |
| logging | stdlib | Per-generation stats logging | Matches existing patterns |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json | stdlib | Debug dump export | Claude's discretion: dump format |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Dict-based cache | functools.lru_cache | lru_cache requires hashable args, query strings are hashable but we need tensor values |
| In-memory only | Disk persistence | User decided in-memory only (matches Phase 2 pattern) |
| LRU eviction | LFU eviction | LRU is simpler, queries accessed uniformly during evolution |

**Installation:**
No new dependencies required. All libraries already present.

## Architecture Patterns

### Recommended Integration Points

```
EvolutionEngine.optimize()
    |
    +-> Orchestrator runs generations
            |
            +-> PopulationEvaluator.evaluate()
                    |
                    +-> SharedPrecomputeContext.prepare_shared_context()
                            |
                            +-> EmbeddingCacheProvider.get_or_create()
                                    |
                                    +-> QueryEmbeddingCache.get_batch() / precompute()
                                            |
                                            +-> Cache HIT: Return embeddings (no compute)
                                            +-> Cache MISS: Embed, store, return
```

The key insight is that `prepare_shared_context()` already batch-embeds queries. We need to:
1. Check the global embedding cache FIRST before computing
2. Store newly computed embeddings in the global cache
3. Track stats per-generation (like FitnessCache does)

### Recommended Module Structure
```
swarm_rag/evolution/execution/
    embedding_cache.py        # MODIFY: Add EmbeddingCacheStats, generation tracking
    shared_precompute.py      # MODIFY: Use EmbeddingCacheProvider for embeddings
    memory_logger.py          # MODIFY: Add embedding cache stats (same as fitness cache)
    evaluator.py              # MINOR: Ensure cache cleanup at evolution end
```

### Pattern 1: Stats Tracking with Time-Saved Calculation
**What:** Track hits, misses, compute time saved, and memory usage per generation
**When to use:** Every cache operation and generation finalization
**Example:**
```python
# Source: Codebase analysis - mirroring FitnessCache pattern
from dataclasses import dataclass
import time

@dataclass
class EmbeddingCacheStats:
    """Statistics about embedding cache usage per generation."""
    hits: int = 0
    misses: int = 0
    compute_time_saved_sec: float = 0.0  # Estimated time not spent embedding
    memory_bytes: int = 0  # Approximate cache memory usage
    entry_count: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0

    def reset(self):
        """Reset per-generation counters (keep cumulative separately)."""
        self.hits = 0
        self.misses = 0
        self.compute_time_saved_sec = 0.0
```

### Pattern 2: LRU Eviction with Access Order Tracking
**What:** Remove oldest entries when max size reached
**When to use:** After each cache store operation
**Example:**
```python
# Source: Existing embedding_cache.py - already implemented
def _evict_if_needed(self):
    """Evict oldest entries if cache exceeds maxsize."""
    while len(self._cache) > self.maxsize and self._access_order:
        oldest = self._access_order.pop(0)
        if oldest in self._cache:
            del self._cache[oldest]

# On cache get - update access order for LRU
if query in self._access_order:
    self._access_order.remove(query)
self._access_order.append(query)
```

### Pattern 3: Time-Saved Estimation
**What:** Estimate embedding compute time saved by cache hits
**When to use:** Track during precompute, report at generation end
**Approach (Claude's discretion):**
```python
# Track average embedding time during misses
# Multiply by hit count to estimate savings

class QueryEmbeddingCache:
    def __init__(self, ...):
        self._avg_embed_time_sec = 0.0
        self._embed_time_samples = 0

    def _update_avg_embed_time(self, elapsed_sec: float, batch_size: int):
        """Update running average of embedding time per query."""
        per_query = elapsed_sec / max(1, batch_size)
        # Exponential moving average
        alpha = 0.1
        if self._embed_time_samples == 0:
            self._avg_embed_time_sec = per_query
        else:
            self._avg_embed_time_sec = alpha * per_query + (1 - alpha) * self._avg_embed_time_sec
        self._embed_time_samples += 1

    def _estimate_time_saved(self, hit_count: int) -> float:
        """Estimate compute time saved by cache hits."""
        return hit_count * self._avg_embed_time_sec
```

### Pattern 4: Device-Aware Storage
**What:** Store embeddings on configured device (GPU or CPU)
**When to use:** Already implemented - follow existing pattern
**Example:**
```python
# Source: Existing embedding_cache.py and device.py
from ...utils.device import get_device

class QueryEmbeddingCache:
    def __init__(self, ..., device: str = None):
        # Resolve device using system pattern
        if device is not None:
            self._device = device
        else:
            self._device = get_device()  # Auto-detect: cuda > mps > cpu
```

### Anti-Patterns to Avoid
- **Recomputing embeddings every generation:** The current SharedPrecomputeContext does this. Must check cache first.
- **Caching on wrong device:** Always use `get_device()` or explicit device parameter. Embeddings must match retrieval device.
- **Storing gradient-attached tensors:** Always `.detach()` before caching to avoid GPU memory leaks.
- **Invalidating cache on query set change:** User decided "no validation" - trust caller.
- **Eager initialization:** User decided lazy init on first access.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LRU eviction | Custom data structure | OrderedDict pattern (existing) | Already implemented in embedding_cache.py |
| Device management | Manual device checking | `get_device()` from utils/device.py | Centralized, handles cuda/mps/cpu |
| Stats tracking | New stats class | Extend existing EmbeddingCacheStats | Already exists, add generation tracking |
| Singleton pattern | New singleton | EmbeddingCacheProvider (existing) | Already exists for global cache |

**Key insight:** The infrastructure already exists. Phase 3 is about LIFECYCLE EXTENSION (generation -> evolution run) and STATS INTEGRATION (like FitnessCache in Phase 2), not building new cache infrastructure.

## Common Pitfalls

### Pitfall 1: Clearing Cache Between Generations
**What goes wrong:** Embeddings recomputed every generation, no savings
**Why it happens:** SharedPrecomputeContext creates fresh context each generation
**How to avoid:** Use EmbeddingCacheProvider singleton, check before computing
**Warning signs:** Cache hit rate is 0% after first generation

### Pitfall 2: Device Mismatch
**What goes wrong:** Tensor on CPU, retriever expects GPU (or vice versa)
**Why it happens:** Cache device differs from retriever device
**How to avoid:** Pass device explicitly to cache, or use get_device() consistently
**Warning signs:** RuntimeError about tensor device mismatch

### Pitfall 3: Memory Leak from Gradient Graphs
**What goes wrong:** GPU memory grows despite embedding being "just a tensor"
**Why it happens:** Tensor retains autograd graph from embedding model
**How to avoid:** Always `.detach()` embeddings before caching
**Warning signs:** CUDA OOM after many generations despite constant query count

### Pitfall 4: Not Updating Access Order on Hits
**What goes wrong:** LRU evicts recently-used items
**Why it happens:** Forgot to move item to end of access list on cache hit
**How to avoid:** Update access order on every get() call, not just misses
**Warning signs:** Frequently-used queries keep getting evicted

### Pitfall 5: Cache Not Cleared at Evolution End
**What goes wrong:** Memory not released between evolution runs
**Why it happens:** EmbeddingCacheProvider singleton lives beyond evolution
**How to avoid:** Call EmbeddingCacheProvider.clear() when evolution completes
**Warning signs:** Memory grows across back-to-back runs

### Pitfall 6: Incorrect Time-Saved Calculation
**What goes wrong:** Reported savings don't match reality
**Why it happens:** Using fixed estimate instead of actual measured embedding time
**How to avoid:** Track actual embedding time during misses, use moving average
**Warning signs:** Reported 10x speedup but wall-clock time unchanged

## Code Examples

### SharedPrecomputeContext Integration
```python
# Source: Codebase analysis of shared_precompute.py

def prepare_shared_context(
    retriever: RetrievalBackend,
    queries: List[str],
    ground_truth: List[List[Any]],
    unique_pool_sizes: List[int],
    device: str = "cpu"
) -> SharedPrecomputeContext:
    """Modified to use embedding cache."""
    import time
    start_time = time.time()

    # Get or create global embedding cache
    from .embedding_cache import EmbeddingCacheProvider

    # Get embedding functions from retriever
    embed_fn = None
    batch_embed_fn = None
    if hasattr(retriever, 'embed_fn'):
        if hasattr(retriever.embed_fn, 'embed_query'):
            embed_fn = retriever.embed_fn.embed_query
        if hasattr(retriever.embed_fn, 'embed_query_batch'):
            batch_embed_fn = retriever.embed_fn.embed_query_batch

    embedding_cache = EmbeddingCacheProvider.get_or_create(
        embedding_fn=embed_fn,
        batch_embedding_fn=batch_embed_fn,
        device=device
    )

    # Get embeddings from cache (will compute misses automatically)
    query_embeddings_dict = embedding_cache.get_batch(queries)

    # Stack into tensor in query order
    query_embeddings = torch.stack([query_embeddings_dict[q] for q in queries])

    # ... rest of function unchanged ...
```

### MemoryLogger Extension (Matching Phase 2 Pattern)
```python
# Source: Codebase analysis of memory_logger.py (FitnessCache pattern)

@dataclass
class GenerationMemoryStats:
    """Memory statistics for a single generation."""
    generation: int
    timestamp: float
    # ... existing fields ...

    # Fitness cache stats (from Phase 2)
    cache_hits: int = 0
    cache_total: int = 0

    # Embedding cache stats (Phase 3)
    embed_cache_hits: int = 0
    embed_cache_total: int = 0
    embed_cache_time_saved_sec: float = 0.0

    def to_log_line(self) -> str:
        line = (
            f"gen={self.generation:04d} "
            f"alloc={self.allocated_mb:7.1f}MB "
            # ... existing fields ...
        )
        # Fitness cache
        if self.cache_total > 0:
            line += f" fcache={self.cache_hits}/{self.cache_total}({self.cache_hit_rate:.0%})"
        # Embedding cache
        if self.embed_cache_total > 0:
            line += f" ecache={self.embed_cache_hits}/{self.embed_cache_total}({self.embed_cache_hit_rate:.0%})"
            if self.embed_cache_time_saved_sec > 0:
                line += f" saved={self.embed_cache_time_saved_sec:.1f}s"
        return line
```

### Debug Dump Method (Claude's Discretion)
```python
# Recommended format: JSON with cache metadata and optionally embeddings

def dump_debug_info(self, path: str = None, include_embeddings: bool = False) -> dict:
    """
    Export cache contents and stats for debugging.

    Args:
        path: Optional file path to write JSON dump
        include_embeddings: If True, include embedding tensors (large!)

    Returns:
        Dictionary with cache debug information
    """
    import json

    info = {
        "cache_size": self.size,
        "embedding_dim": self._embedding_dim,
        "device": self._device,
        "maxsize": self.maxsize,
        "stats": {
            "total_hits": self.stats.cache_hits,
            "total_misses": self.stats.cache_misses,
            "hit_rate": self.stats.hit_rate,
            "total_embedding_time": self.stats.total_embedding_time,
            "precompute_time": self.stats.precompute_time,
        },
        "cached_queries": list(self._cache.keys()),
    }

    if include_embeddings:
        info["embeddings"] = {
            q: emb.cpu().tolist() for q, emb in self._cache.items()
        }

    if path:
        with open(path, 'w') as f:
            json.dump(info, f, indent=2)

    return info
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Recompute per generation | Cache across evolution run | Phase 3 | 50-80% embedding compute savings |
| No stats tracking | Per-generation stats | Phase 2 pattern | Visibility into cache efficiency |
| Fixed cache size | Configurable with LRU eviction | Existing | Memory-bounded operation |

**Deprecated/outdated:**
- Eager cache population: User decided lazy initialization
- Cache validation on query change: User decided "trust caller"
- Disk persistence: User decided in-memory only

## Open Questions

1. **Query ID vs Query String as Cache Key**
   - What we know: User decided "key by query ID (queries have specific IDs)"
   - What's unclear: In some code paths, queries are strings. Are IDs always available?
   - Recommendation: Support both - if query is string, use string as key. If query has ID attribute, use ID. The existing cache already uses query strings as keys.

2. **Cache Size Limit Default**
   - What we know: User decided "configurable max cache size limit"
   - What's unclear: What default? Current is 10,000
   - Recommendation: Keep 10,000 default. Typical evolution uses 100-500 queries, well under limit.

3. **Memory Calculation Accuracy**
   - What we know: Need to report memory used by cache
   - What's unclear: Exact tensor memory accounting (device-specific)
   - Recommendation: Estimate as `entry_count * embedding_dim * 4 bytes` (float32). Good enough for observability.

## Sources

### Primary (HIGH confidence)
- Codebase: `embedding_cache.py` - Existing cache implementation with LRU, device management
- Codebase: `shared_precompute.py` - Current per-generation embedding computation
- Codebase: `fitness_cache.py` - Pattern for stats tracking and MemoryLogger integration
- Codebase: `memory_logger.py` - Integration point for cache stats
- Codebase: `device.py` - Device resolution pattern

### Secondary (MEDIUM confidence)
- [Real Python - LRU Cache](https://realpython.com/lru-cache-python/) - LRU implementation patterns
- [PyTorch CUDA Semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) - Memory management best practices
- [GPU Memory Management Best Practices 2026](https://copyprogramming.com/howto/how-to-clear-gpu-memory-after-using-model) - PyTorch memory clearing patterns

### Tertiary (LOW confidence)
- [GeeksforGeeks - Cache Eviction Policies](https://www.geeksforgeeks.org/system-design/cache-eviction-policies-system-design/) - General eviction policy comparison

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in use, patterns established
- Architecture: HIGH - Integration points clearly identified in codebase
- Pitfalls: HIGH - Based on codebase analysis and established patterns

**Research date:** 2026-01-30
**Valid until:** 60 days (stable domain, no rapid changes expected)
