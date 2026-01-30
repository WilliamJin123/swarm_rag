# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-29)

**Core value:** Find a genome configuration that hits SOTA metrics on STARK Prime (Hit@1 > 60%, Hit@5 > 80%, Recall@20 > 85%, MRR > 80%)
**Current focus:** Phase 3 - Embedding Cache (next)

## Current Position

Phase: 2 of 7 (Fitness Caching) - VERIFIED ✓
Plan: 1 of 1 in current phase complete
Status: Phase verified (4/4 truths), ready for Phase 3
Last activity: 2026-01-29 - Verification passed

Progress: [=====------] 50%

## Performance Metrics

**Velocity:**
- Total plans completed: 5
- Average duration: 9min
- Total execution time: 0.7 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-memory-hardening | 4 | 37min | 9min |
| 02-fitness-caching | 1 | 8min | 8min |

**Recent Trend:**
- Last 5 plans: 02-01 (8min), 01-04 (8min), 01-03 (6min), 01-02 (8min), 01-01 (15min)
- Trend: stable (8min average for recent plans)

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

### Pending Todos

None yet.

### Blockers/Concerns

- Recent bug fixes for memory/latency explosion are untested (from PROJECT.md)
- Unknown if current heuristic space can express SOTA-level genomes

## Session Continuity

Last session: 2026-01-29
Stopped at: Phase 2 verified, ready for Phase 3
Resume file: None
