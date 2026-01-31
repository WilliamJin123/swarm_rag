# Phase 5: Async Checkpointing - Context

**Gathered:** 2026-01-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Non-blocking checkpoint saves during evolution loop. Checkpoints are written to disk without blocking the main evolution thread, allowing evolution to continue while previous checkpoint writes complete.

</domain>

<decisions>
## Implementation Decisions

### Threading model
- Single background thread (not ThreadPoolExecutor)
- Queue checkpoints to the background thread for sequential writing
- If new checkpoint requested while one is writing, queue it (don't drop or skip)
- On shutdown (evolution ends or convergence detected), wait for queue to drain before exiting

### Data consistency
- Deep copy all checkpoint data before queuing to background thread
- Full evolution state: population, metrics, archive, generation number, config
- Tensor data (if any) must be detached and moved to CPU before copying
- Prevents GPU memory pinning by background thread

### Failure handling
- Write to temp file, atomic rename on success (prevents partial/corrupt files)
- On write failure: retry once, then log warning and continue evolution
- Clean up failed temp files automatically (don't leave partial files)
- Keep all checkpoint files (no automatic rotation/deletion)

### Status/feedback
- Log when checkpoint queued: "Checkpoint queued (gen N)"
- Log when checkpoint completed: "Checkpoint saved (gen N, size MB, time s)"
- Log queue depth warning if > 1 pending: "Checkpoint queue: N pending"
- Log summary on shutdown: "Checkpointing: N saves, Xs total, avg Ys"

### Claude's Discretion
- Archive handling in checkpoint (include full or rebuild on restore)
- Exact temp file naming convention
- Queue implementation details (stdlib queue vs simple list)
- Thread daemon status

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

*Phase: 05-async-checkpointing*
*Context gathered: 2026-01-30*
