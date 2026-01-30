# Phase 4: Convergence Detection - Context

**Gathered:** 2026-01-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Detect when evolution has stagnated and stop early to save wasted generations. This phase adds convergence detection logic to the evolution loop — it does NOT change how evolution works, only when it terminates.

</domain>

<decisions>
## Implementation Decisions

### Detection Sensitivity
- Conservative approach: larger window (30-50 gens default), stricter threshold
- Adaptive window: expands when score is climbing, shrinks when flat
- Relative threshold accounting for diminishing returns — expect bigger jumps when QD-score is low, smaller acceptable near ceiling (percentage-based relative to current score or remaining headroom)
- Grace period: minimum N generations must complete before convergence detection activates

### Convergence Signals
- QD-score only — single metric keeps logic simple, captures both quality and diversity
- No secondary signals (archive saturation, diversity metrics, best fitness)
- End-of-generation QD snapshot for comparison (no smoothing/averaging)
- Single stagnant window sufficient to trigger convergence

### Termination Behavior
- Immediate stop at end of current generation when convergence detected
- Summary log with stats: generation number, window stats, final QD-score, reason for detection
- Same save behavior as normal end: full checkpoint, final metrics, best genome
- Mark termination reason: 'stopped: convergence' vs 'stopped: max_generations' in output/checkpoint

### Override/Recovery
- Config flag to disable convergence detection entirely (run always goes to max_generations)
- Terminated is terminated — no resume past convergence point, start new run from checkpoint if desired
- Grace period handles minimum generations (no separate min_generations config needed)
- Config read at start only — settings locked for duration of run

### Claude's Discretion
- Exact window size defaults within conservative range
- Adaptive window expansion/shrink algorithm
- Specific threshold percentage/formula for diminishing returns
- Log message formatting details

</decisions>

<specifics>
## Specific Ideas

- Threshold should be "relative" — when QD-score is 100, expect bigger jumps; when QD-score is 900 (near ceiling), smaller improvements are acceptable
- This is an evolution optimization, not a user-facing feature — logging is the primary visibility

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-convergence-detection*
*Context gathered: 2026-01-30*
