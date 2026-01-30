# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-29)

**Core value:** Find a genome configuration that hits SOTA metrics on STARK Prime (Hit@1 > 60%, Hit@5 > 80%, Recall@20 > 85%, MRR > 80%)
**Current focus:** Phase 3 - Embedding Cache (in progress)

## Current Position

Phase: 3 of 7 (Embedding Cache)
Plan: 1 of 1 in current phase complete
Status: Plan 03-01 complete, awaiting phase verification
Last activity: 2026-01-30 - Completed 03-01-PLAN.md

Progress: [======-----] 55%

## Performance Metrics

**Velocity:**
- Total plans completed: 6
- Average duration: 9min
- Total execution time: 0.9 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-memory-hardening | 4 | 37min | 9min |
| 02-fitness-caching | 1 | 8min | 8min |
| 03-embedding-cache | 1 | 12min | 12min |

**Recent Trend:**
- Last 5 plans: 03-01 (12min), 02-01 (8min), 01-04 (8min), 01-03 (6min), 01-02 (8min)
- Trend: stable (8-12min average)

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

### Pending Todos

None yet.

### Blockers/Concerns

- Recent bug fixes for memory/latency explosion are untested (from PROJECT.md)
- Unknown if current heuristic space can express SOTA-level genomes

## Session Continuity

Last session: 2026-01-30
Stopped at: Completed 03-01-PLAN.md
Resume file: None
