# Requirements: Swarm RAG Evolution

**Defined:** 2026-01-29
**Core Value:** Find a genome configuration that hits SOTA metrics on STARK Prime

## v1 Requirements

Requirements for achieving SOTA metrics with performant evolution loop.

### Performance Targets

- [ ] **PERF-01**: Evolution loop completes 500 generations in 3 hours with population 50-100
- [ ] **PERF-02**: Memory usage stays stable across all 500 generations (no accumulation)
- [ ] **PERF-03**: VRAM usage stays under a few GB even for larger datasets
- [ ] **PERF-04**: Query latency maintained at ~50ms for STARK Prime

### Retrieval Metrics (SOTA)

- [ ] **METRIC-01**: Genome achieves Hit@1 > 60% on STARK Prime
- [ ] **METRIC-02**: Genome achieves Hit@5 > 80% on STARK Prime
- [ ] **METRIC-03**: Genome achieves Recall@20 > 85% on STARK Prime
- [ ] **METRIC-04**: Genome achieves MRR > 80% on STARK Prime

### Caching

- [x] **CACHE-01**: Fitness caching by genome hash — skip re-evaluation of duplicate/elite genomes
- [x] **CACHE-02**: Cross-generation embedding cache — persist query embeddings across generations

### Memory Management

- [x] **MEM-01**: Tensor lifecycle guards — strict detach/no_grad/empty_cache patterns in hot paths
- [x] **MEM-02**: Memory monitoring — track GPU memory per generation, alert on growth trends
- [x] **MEM-03**: Buffer reuse — pre-allocate and reuse traversal buffers instead of per-step allocation

### Convergence

- [x] **CONV-01**: Early stopping based on QD-score stagnation (sliding window detection)
- [x] **CONV-02**: Configurable convergence threshold and window size

### Checkpointing

- [x] **CKPT-01**: Async checkpointing — non-blocking checkpoint saves during evolution

## v2 Requirements

Deferred to future if v1 doesn't achieve targets.

### Advanced Optimization

- **ADV-01**: Adaptive population sizing — start smaller, scale up during refinement
- **ADV-02**: Surrogate-assisted evaluation — use surrogate model for cheap fitness approximation
- **ADV-03**: EvoTorch migration — if existing implementation can't hit performance targets

### Extended Benchmarks

- **BENCH-01**: Achieve SOTA on Amazon dataset
- **BENCH-02**: Achieve SOTA on MAG dataset
- **BENCH-03**: Answer generation agent using retrieved nodes

## Out of Scope

| Feature | Reason |
|---------|--------|
| EvoTorch migration | Significant refactor, evaluate existing fixes first |
| Answer generation agent | Secondary priority, defer until SOTA metrics achieved |
| Amazon/MAG optimization | Focus on STARK Prime first |
| New heuristic types | Evaluate if current space can express winning genome first |
| Surrogate models | Requires domain validation, defer to v2 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PERF-01 | Phase 6 | Pending |
| PERF-02 | Phase 1 | Pending |
| PERF-03 | Phase 1 | Pending |
| PERF-04 | Phase 6 | Pending |
| METRIC-01 | Phase 7 | Pending |
| METRIC-02 | Phase 7 | Pending |
| METRIC-03 | Phase 7 | Pending |
| METRIC-04 | Phase 7 | Pending |
| CACHE-01 | Phase 2 | Complete |
| CACHE-02 | Phase 3 | Complete |
| MEM-01 | Phase 1 | Complete |
| MEM-02 | Phase 1 | Complete |
| MEM-03 | Phase 1 | Complete |
| CONV-01 | Phase 4 | Complete |
| CONV-02 | Phase 4 | Complete |
| CKPT-01 | Phase 5 | Complete |

**Coverage:**
- v1 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0

---
*Requirements defined: 2026-01-29*
*Last updated: 2026-01-30 after Phase 5 completion*
