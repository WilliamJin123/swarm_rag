# Roadmap: Swarm RAG Evolution

## Overview

This roadmap delivers a performant evolution system capable of running 500 generations in 3 hours to discover SOTA GraphRAG configurations on STARK Prime. The journey progresses from memory stability (preventing crashes) through caching optimizations (eliminating redundant computation) to convergence detection (stopping early when stagnating), culminating in actual evolution runs that achieve target metrics (Hit@1 > 60%, Hit@5 > 80%, Recall@20 > 85%, MRR > 80%).

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Memory Hardening** - Establish stable memory management to prevent crashes during long runs
- [x] **Phase 2: Fitness Caching** - Cache evaluated genomes to skip redundant fitness computation
- [x] **Phase 3: Embedding Cache** - Persist query embeddings across generations to eliminate recomputation
- [x] **Phase 4: Convergence Detection** - Detect stagnation and stop early when evolution plateaus
- [x] **Phase 5: Async Checkpointing** - Non-blocking checkpoint saves during evolution
- [ ] **Phase 6: Performance Validation** - Validate 500 gen / 3 hour target with full optimization stack
- [ ] **Phase 7: SOTA Evolution** - Run evolution to discover genome achieving SOTA metrics

## Phase Details

### Phase 1: Memory Hardening
**Goal**: GPU memory remains stable across 500 generations with no accumulation or OOM crashes
**Depends on**: Nothing (first phase)
**Requirements**: MEM-01, MEM-02, MEM-03, PERF-02, PERF-03
**Success Criteria** (what must be TRUE):
  1. Evolution runs 100+ generations with GPU memory staying within 10% of initial allocation
  2. Memory monitoring dashboard shows no upward trend across generations
  3. Per-genome evaluation cleans up all intermediate tensors (no gradients retained)
  4. Traversal buffers are pre-allocated and reused, not created per-step
  5. VRAM usage stays under 4GB for STARK Prime throughout run
**Plans**: 4 plans (3 original + 1 gap closure)

Plans:
- [x] 01-01-PLAN.md - MemoryGuard context manager with threshold enforcement
- [x] 01-02-PLAN.md - MemoryLogger for per-generation tracking and alerting
- [x] 01-03-PLAN.md - Buffer pre-allocation and MemoryGuard integration
- [x] 01-04-PLAN.md - Gap closure: Wire buffer pool + torch.no_grad() coverage

### Phase 2: Fitness Caching
**Goal**: Duplicate and elite genomes skip evaluation via content-hash caching (30-70% eval savings)
**Depends on**: Phase 1 (stable memory required before adding caches)
**Requirements**: CACHE-01
**Success Criteria** (what must be TRUE):
  1. Genomes with identical configurations return cached fitness without re-evaluation
  2. Cache hit rate is logged and visible per generation
  3. Elite genomes carried forward hit cache instead of re-evaluating
  4. Cache properly handles detached tensors (no gradient graph retention)
**Plans**: 1 plan

Plans:
- [x] 02-01-PLAN.md - FitnessCache module with genome hashing and evaluator integration

### Phase 3: Embedding Cache
**Goal**: Query embeddings persist across generations, eliminating redundant embedding computation (50-80% retrieval savings)
**Depends on**: Phase 2 (caching patterns established)
**Requirements**: CACHE-02
**Success Criteria** (what must be TRUE):
  1. Query embeddings computed once at start, reused across all generations
  2. Embedding cache validates query consistency (detects if queries change)
  3. SharedPrecomputeContext lifetime extends beyond single generation
  4. Retrieval time per genome drops measurably after first generation
**Plans**: 2 plans (1 original + 1 gap closure)

Plans:
- [x] 03-01-PLAN.md - Cross-generation embedding cache with stats and MemoryLogger integration
- [x] 03-02-PLAN.md - Gap closure: Wire embedding cache finalize_generation in evaluator

### Phase 4: Convergence Detection
**Goal**: Evolution stops early when QD-score stagnates, saving 20-40% of wasted generations
**Depends on**: Phase 3 (optimizations in place before adding early stopping)
**Requirements**: CONV-01, CONV-02
**Success Criteria** (what must be TRUE):
  1. Sliding window detector identifies when QD-score improvement falls below threshold
  2. Convergence threshold and window size are configurable via evolution config
  3. Evolution run terminates gracefully when convergence detected
  4. Final checkpoint and metrics saved on early termination
**Plans**: 1 plan

Plans:
- [x] 04-01-PLAN.md - ConvergenceDetector module with sliding window analysis and orchestrator integration

### Phase 5: Async Checkpointing
**Goal**: Checkpoint saves happen without blocking evolution loop
**Depends on**: Phase 4 (convergence detection can trigger checkpoints)
**Requirements**: CKPT-01
**Success Criteria** (what must be TRUE):
  1. Checkpoint saves occur in background thread without blocking generation loop
  2. Evolution continues while previous checkpoint writes to disk
  3. Checkpoint integrity maintained (no partial writes on crash)
  4. Async checkpointing adds negligible overhead (<1% of generation time)
**Plans**: 1 plan

Plans:
- [x] 05-01-PLAN.md - Refactor AsyncCheckpointWriter with queue-all semantics and atomic writes

### Phase 6: Performance Validation
**Goal**: Validate full optimization stack achieves 500 generations in 3 hours with population 50-100
**Depends on**: Phase 5 (all optimizations complete)
**Requirements**: PERF-01, PERF-04
**Success Criteria** (what must be TRUE):
  1. Evolution completes 500 generations in under 3 hours with population 50
  2. Average generation time is under 21.6 seconds (3 hours / 500 gens)
  3. Query latency remains at ~50ms throughout run (no degradation)
  4. All caches, memory guards, and convergence detection functioning together
**Plans**: 1 plan

Plans:
- [ ] 06-01-PLAN.md - Full-stack performance benchmark with JSON report and pass/fail determination

### Phase 7: SOTA Evolution
**Goal**: Run evolution search to discover genome achieving SOTA metrics on STARK Prime
**Depends on**: Phase 6 (performance validated)
**Requirements**: METRIC-01, METRIC-02, METRIC-03, METRIC-04
**Success Criteria** (what must be TRUE):
  1. Best genome achieves Hit@1 > 60% on STARK Prime validation set
  2. Best genome achieves Hit@5 > 80% on STARK Prime validation set
  3. Best genome achieves Recall@20 > 85% on STARK Prime validation set
  4. Best genome achieves MRR > 80% on STARK Prime validation set
  5. Winning genome configuration is saved and reproducible
**Plans**: TBD

Plans:
- [ ] 07-01: Evolution run with metric tracking
- [ ] 07-02: Genome analysis and reproducibility validation

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Memory Hardening | 4/4 | ✓ Complete | 2026-01-29 |
| 2. Fitness Caching | 1/1 | ✓ Complete | 2026-01-29 |
| 3. Embedding Cache | 2/2 | ✓ Complete | 2026-01-30 |
| 4. Convergence Detection | 1/1 | ✓ Complete | 2026-01-30 |
| 5. Async Checkpointing | 1/1 | ✓ Complete | 2026-01-30 |
| 6. Performance Validation | 0/1 | Planned | - |
| 7. SOTA Evolution | 0/2 | Not started | - |

---
*Roadmap created: 2026-01-29*
*Phase 1 planned: 2026-01-29*
*Phase 1 gap closure planned: 2026-01-30*
*Phase 1 complete: 2026-01-29*
*Phase 2 planned: 2026-01-30*
*Phase 2 complete: 2026-01-29*
*Phase 3 planned: 2026-01-30*
*Phase 3 gap closure planned: 2026-01-30*
*Phase 3 complete: 2026-01-30*
*Phase 4 planned: 2026-01-30*
*Phase 4 complete: 2026-01-30*
*Phase 5 planned: 2026-01-30*
*Phase 5 complete: 2026-01-30*
*Phase 6 planned: 2026-01-31*
*Depth: standard (7 phases)*
*Requirements coverage: 16/16 mapped*
