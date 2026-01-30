# Phase 1: Memory Hardening - Context

**Gathered:** 2026-01-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish stable GPU memory management to prevent OOM crashes and memory accumulation during long evolution runs (500+ generations). GPU memory should remain stable within 10% of initial allocation with no upward trend.

</domain>

<decisions>
## Implementation Decisions

### Cleanup behavior
- Clean up tensors immediately after each genome evaluation (most aggressive)
- Wrap all evaluation code in `torch.no_grad()` — no gradients ever computed
- Claude's discretion: synchronous vs async cleanup, CUDA cache clearing strategy

### Memory thresholds
- Warning threshold: 70% of VRAM
- Hard stop threshold: 85% of VRAM
- Thresholds configurable via environment variables
- Detect memory growth via per-generation delta (compare before/after each generation)

### Monitoring output
- Log full stats every generation: current usage, delta, peak, allocated, cached
- Dedicated `memory.log` file separate from main evolution log
- Claude's discretion: export format for visualization (JSON/CSV/etc.)

### Failure handling
- On hard stop threshold: checkpoint current state and exit cleanly
- No auto-resume — manual intervention required to resume from checkpoint
- Genome evaluation failures are bugs — crash the entire evolution loop immediately
- On crash: dump full memory state (allocation details, tensor shapes, stack trace)

### Claude's Discretion
- Synchronous vs async cleanup timing
- CUDA cache clearing frequency and triggers
- Memory stats export format
- Specific cleanup implementation patterns

</decisions>

<specifics>
## Specific Ideas

- Genome evaluation failures indicate design flaws and should crash immediately rather than be silently handled
- The 70%/85% warning/stop thresholds are conservative to ensure headroom
- Per-generation delta detection for immediate visibility into memory accumulation

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-memory-hardening*
*Context gathered: 2026-01-29*
