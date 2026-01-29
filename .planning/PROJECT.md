# Swarm RAG Evolution

## What This Is

An evolutionary search system that discovers optimal GraphRAG retrieval configurations (genomes) for the STARK benchmark. Genomes encode movement heuristics, deposit strategies, and ranking algorithms that guide ant-colony-inspired agents traversing knowledge graphs. The goal is to find a genome that achieves state-of-the-art retrieval metrics on STARK Prime.

## Core Value

Find a genome configuration that hits SOTA metrics on STARK Prime (Hit@1 > 60%, Hit@5 > 80%, Recall@20 > 85%, MRR > 80%).

## Requirements

### Validated

- ✓ Swarm retriever with multi-agent graph traversal — existing
- ✓ Pluggable heuristics (movement, deposit, ranking) — existing
- ✓ MAP-Elites evolutionary optimization — existing
- ✓ Dual genome modes (weighted_sum and expression_tree) — existing
- ✓ STARK dataset integration (Prime, Amazon, MAG) — existing
- ✓ Batch query evaluation pipeline — existing
- ✓ Fitness calculation with retrieval metrics (MRR, Hit@K, Recall@K) — existing
- ✓ GPU-accelerated tensor operations — existing
- ✓ Checkpoint/resume for evolution runs — existing
- ✓ 50ms query latency on STARK Prime — existing

### Active

- [ ] Evolution loop runs 500 generations in 3 hours with population 50-100
- [ ] Memory stays stable across generations (no accumulation/explosion)
- [ ] Genome achieves Hit@1 > 60% on STARK Prime
- [ ] Genome achieves Hit@5 > 80% on STARK Prime
- [ ] Genome achieves Recall@20 > 85% on STARK Prime
- [ ] Genome achieves MRR > 80% on STARK Prime
- [ ] VRAM usage stays under a few GB even for Amazon/MAG (10x larger)

### Out of Scope

- Answer generation agent — secondary priority, defer until SOTA metrics achieved
- Amazon/MAG optimization — focus is STARK Prime first
- New heuristic types — may be needed if current space can't express winning genome, but evaluate first

## Context

The system uses ant-colony optimization where agents traverse a knowledge graph guided by heuristics. Each genome defines hyperparameters (n_agents, steps, decay) and weights for combining heuristic scores during movement, deposit, and ranking phases.

Recent bug fixes may have resolved memory/latency explosion issues where genomes got progressively slower over generations. These fixes are untested.

Current state:
- Query latency (50ms) is acceptable
- Evolution loop performance is the bottleneck
- Unknown if current heuristic space can express SOTA-level genomes
- Need profiling to identify specific bottlenecks

The path forward is: optimize performance → run evolution → either celebrate SOTA or learn heuristics need redesign.

## Constraints

- **Performance**: 500 generations in 3 hours with population 50-100
- **Memory**: Few GB VRAM max, even for larger datasets
- **Query latency**: Maintain ~50ms per query on STARK Prime
- **Tech stack**: Python/PyTorch, existing architecture

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Focus on STARK Prime first | Most tractable dataset, prove concept before scaling | — Pending |
| Defer answer generation agent | Core value is retrieval metrics, not end-to-end QA | — Pending |
| Test bug fixes before major changes | Recent fixes may have solved performance issues | — Pending |

---
*Last updated: 2026-01-29 after initialization*
