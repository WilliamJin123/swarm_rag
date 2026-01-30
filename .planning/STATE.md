# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-29)

**Core value:** Find a genome configuration that hits SOTA metrics on STARK Prime (Hit@1 > 60%, Hit@5 > 80%, Recall@20 > 85%, MRR > 80%)
**Current focus:** Phase 4 - Convergence Detection (plan 01 complete)

## Current Position

Phase: 4 of 7 (Convergence Detection)
Plan: 1 of 1 complete
Status: Phase 04 plan 01 complete
Last activity: 2026-01-30 - Completed 04-01-PLAN.md

Progress: [=======----] 70%

## Performance Metrics

**Velocity:**
- Total plans completed: 8
- Average duration: 8min
- Total execution time: 1.1 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-memory-hardening | 4 | 37min | 9min |
| 02-fitness-caching | 1 | 8min | 8min |
| 03-embedding-cache | 2 | 17min | 9min |
| 04-convergence-detection | 1 | 5min | 5min |

**Recent Trend:**
- Last 5 plans: 04-01 (5min), 03-02 (5min), 03-01 (12min), 02-01 (8min), 01-04 (8min)
- Trend: stable (5-12min average)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Memory hardening before caching (caching increases memory pressure)
- [Roadmap]: 7 phases derived from requirement clusters (not imposed template)
- [01-01]: Use memory_allocated() not memory_reserved() for thresholds
- [01-01]: Default thresholds 70% warning, 85% hard stop
- [01-01]: Environment variable override for thresholds
- [01-02]: Memory stats logged at START of generation (before operations)
- [01-02]: Peak memory reset each generation for accurate per-gen peaks
- [01-03]: Buffer pool uses 2x headroom for pool_size and agents
- [01-03]: max_degree estimated from graph's avg_degree * 2
- [01-03]: All 4 single-genome evaluation methods wrapped with MemoryGuard
- [01-04]: torch.no_grad() placed inside MemoryGuard for belt-and-suspenders gradient prevention
- [01-04]: Buffer pool uses conditional pattern for backward compatibility
- [01-04]: clear() called after each traversal/batch to prevent state leakage
- [02-01]: xxhash64 for genome hashing (speed over cryptographic strength)
- [02-01]: In-memory cache only (fresh on restart, no disk persistence)
- [02-01]: Store quality_score only (single float per entry)
- [02-01]: Round floats to 6 decimals to handle FP noise
- [02-01]: Hash config not id (same config = same behavior = same fitness)
- [03-01]: EMA (alpha=0.1) for average embedding time tracking
- [03-01]: Lazy integration: prepare_shared_context uses cache if exists
- [03-01]: Detach tensors before caching to prevent memory leaks
- [03-01]: Per-generation stats separate from cumulative stats
- [03-02]: Log format: 'Embedding cache: N lookups, Xs saved' consistent with fitness cache
- [03-02]: Null-safe: only log if EmbeddingCacheProvider.get() returns non-None
- [04-01]: Window size 40 default (conservative 30-50 range)
- [04-01]: Grace period 20 generations before detection activates
- [04-01]: Threshold 0.1% relative improvement required
- [04-01]: Adaptive window: expand when improving, shrink when flat

### Pending Todos

None yet.

### Blockers/Concerns

- Recent bug fixes for memory/latency explosion are untested (from PROJECT.md)
- Unknown if current heuristic space can express SOTA-level genomes

## Session Continuity

Last session: 2026-01-30
Stopped at: Completed 04-01-PLAN.md
Resume file: None
