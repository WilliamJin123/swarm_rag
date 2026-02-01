# Phase 6: Performance Validation - Context

**Gathered:** 2026-01-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Validate that the full optimization stack (memory guards, fitness cache, embedding cache, convergence detection, async checkpointing) achieves 500 generations in 3 hours with population 50-100. This is a benchmarking/validation phase — no new features, just proving the system meets performance targets.

</domain>

<decisions>
## Implementation Decisions

### Benchmark Protocol
- Single run (no statistical replication needed for validation)
- No warm-up period — include cold-start in timing (reflects real-world usage)
- Population size: 75 (midpoint of 50-100 target range)
- Allow convergence early-stop — extrapolate to 500 gens if stopped early

### Metrics & Reporting
- Per-generation timing: track each generation's duration, report avg/min/max/p99
- Cache stats: report fitness cache hit % and embedding cache hit %
- Output: console summary + JSON report file (`.planning/phases/06-performance-validation/benchmark-results.json`)
- Pass criteria: projected time (avg_gen_time * 500) < 3 hours

### Bottleneck Response
- On failure: profile, suggest tuning, and report (no auto-tune)
- Profiling: use PyTorch profiler for GPU-level insights
- Track peak GPU memory across the run
- Memory threshold: VRAM > 4GB is a failure even if timing passes

### Test Environment
- GPU: whatever is available, record model in results
- Dataset: full STARK Prime (realistic performance)
- Isolation: not required, run on normal development machine
- System info: capture GPU, CUDA, Python, PyTorch versions in JSON report

### Claude's Discretion
- Exact JSON schema for benchmark results
- Console summary formatting
- Profiler configuration details
- How to extrapolate timing if convergence triggers early

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

*Phase: 06-performance-validation*
*Context gathered: 2026-01-31*
