# Phase 3: Embedding Cache - Context

**Gathered:** 2026-01-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Persist query embeddings across generations to eliminate redundant embedding computation. SharedPrecomputeContext lifetime extends beyond single generation. Target 50-80% retrieval savings.

</domain>

<decisions>
## Implementation Decisions

### Cache Lifecycle
- Lazy initialization on first access (not eager at evolution start)
- Never auto-invalidate — cache lives for entire evolution run
- Auto-cleanup on evolution completion (releases memory automatically)
- Shared across back-to-back runs if same query set (reuse if queries match)

### Memory/Persistence
- Store on current device (GPU mode → GPU memory, CPU mode → CPU memory)
- Follows system's existing device mode pattern
- Dynamic growth allocation (not pre-allocated)
- Configurable max cache size limit with eviction on overflow

### Query Handling
- No validation of query changes — trust caller
- Re-compute silently if cache is manually cleared
- Supports additive updates (new queries can be added mid-run)
- Key by query ID (queries have specific IDs, use those as cache keys)

### Observability
- Full stats per generation: hits, misses, memory, entry count, compute time saved
- Integrate with existing MemoryLogger (appears alongside memory stats)
- End-of-evolution summary: total embeddings cached, time saved, memory used
- Debug dump method to export cache contents/stats for troubleshooting

### Claude's Discretion
- Exact eviction policy when max size reached
- Implementation of time-saved calculation
- Debug dump format

</decisions>

<specifics>
## Specific Ideas

- Follow the same device mode pattern the swarm and evolution already use (CPU/GPU modes)
- Integration with MemoryLogger should mirror how FitnessCache stats were added in Phase 2

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-embedding-cache*
*Context gathered: 2026-01-30*
