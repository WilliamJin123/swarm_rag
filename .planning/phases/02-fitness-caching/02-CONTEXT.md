# Phase 2: Fitness Caching - Context

**Gathered:** 2026-01-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Cache evaluated genomes to skip redundant fitness computation. Duplicate genomes and elite genomes carried forward should hit cache instead of re-evaluating. Target 30-70% evaluation savings.

</domain>

<decisions>
## Implementation Decisions

### Hashing Strategy
- Hash ALL config fields (every parameter that affects retrieval behavior)
- Genome config only — no dataset identifier in hash (dataset is constant per run)
- Round floating-point parameters to reasonable precision (4-6 decimals) before hashing to avoid floating point noise
- Use fast hash algorithm (xxhash or cityhash) — speed over cryptographic strength

### Cache Invalidation
- In-memory only — cache lives within single evolution run, no disk persistence
- Never expire during a run — same genome = same fitness within a run
- Fresh cache on checkpoint resume — cache not included in checkpoints
- No special stale-data handling needed — in-memory means every fresh run starts clean

### Logging
- Per-generation summary: "Gen 45: 23/50 cache hits (46%)" — one line per generation
- Integrate with existing memory logger — cache stats appear alongside memory stats
- Track hit rate only (hits/total) — no time estimates or additional metrics
- No warnings for low hit rates — early generations naturally have low rates

### Memory Limits
- Unlimited cache within run — fitness values are tiny floats, 25K entries max is negligible
- Store just the fitness score — single float per entry
- No interaction with MemoryGuard — cache is CPU RAM, doesn't affect GPU thresholds
- Single-threaded — evolution loop is synchronous, no locking needed

### Claude's Discretion
- Exact hash key serialization format
- Cache data structure choice (dict vs specialized container)
- Integration point in evaluation flow

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-fitness-caching*
*Context gathered: 2026-01-29*
