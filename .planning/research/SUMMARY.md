# Project Research Summary

**Project:** Swarm RAG Evolution - GPU-accelerated MAP-Elites search for SOTA GraphRAG configurations
**Domain:** High-Performance Evolutionary Optimization for Information Retrieval
**Researched:** 2026-01-29
**Confidence:** HIGH

## Executive Summary

This project requires optimizing a GPU-accelerated evolutionary search system to evaluate 500 generations in 3 hours with population sizes of 50-100. Research reveals the evaluation loop is the dominant bottleneck (90%+ of wall-clock time), with fitness evaluation taking approximately 94% of total runtime. The current system has strong foundations - GPU batch metrics, shared precompute contexts, early exit filtering, and generation profiling - but three critical optimizations are needed: fitness caching to eliminate redundant evaluations (30-70% savings), cross-generation embedding cache to prevent recomputation of constant query embeddings (50-80% retrieval speedup), and adaptive convergence detection to stop early when evolution stagnates (20-40% time savings).

The recommended technical approach follows established GPU-accelerated quality-diversity patterns: vectorized batch fitness evaluation, aggressive memory cleanup with explicit tensor lifecycle management, torch.compile integration for kernel fusion, and strategic use of torch.no_grad() to prevent gradient graph accumulation. The system should avoid common pitfalls including tensor reference retention in loops, per-step buffer creation without reuse, and shared context lifetime mismanagement. The existing architecture of tight generation loops with GPU-batched evaluation is correct; the focus must be on memory discipline and eliminating redundant computation.

The primary risk is GPU memory accumulation across generations leading to out-of-memory crashes or performance degradation. This is mitigated through: (1) explicit `detach().clone()` on all genome copies, (2) `torch.no_grad()` wrappers around evaluation, (3) per-genome `torch.cuda.empty_cache()` calls, and (4) memory guards that assert stability across generations. Secondary risks include thread pool over-subscription causing CPU contention and expression tree interpreter overhead as genomes evolve complexity. The roadmap should prioritize hot path optimizations first (fitness caching, cross-gen caching), followed by memory management hardening, with architectural refactoring deferred until performance targets are met.

## Key Findings

### Recommended Stack

The optimization stack centers on PyTorch 2.9+ GPU acceleration with aggressive memory management and optional integration of production-grade evolutionary frameworks. Core technologies include EvoTorch 0.6.1 for GPU-vectorized MAP-Elites (eliminates serial loops, processes entire populations in batch), torch.compile with `mode="reduce-overhead"` for kernel fusion and CUDA graph optimization (20-40% speedup), and PyTorch Memory Profiler for tracking allocations across generations. Ray 2.x provides multi-process parallelization with zero-copy tensor sharing when needed.

**Core technologies:**
- **EvoTorch 0.6.1**: GPU-vectorized MAP-Elites implementation - native batch processing of entire populations, eliminates serial mutation/evaluation loops
- **torch.compile (PyTorch 2.9+)**: JIT compilation with kernel fusion - use `mode="reduce-overhead"` for CUDA graphs when shapes are static (evolution has fixed population sizes)
- **torch.profiler + Memory Profiler**: Bottleneck identification and leak detection - critical for validating optimization impact and catching memory accumulation
- **Ray 2.x**: Multi-process parallelization - zero-copy tensor sharing via shared memory, superior to multiprocessing for tensor workloads

**Critical implementation techniques:**
- Detach tensors when copying across generations: `new_genome = old_genome.detach().clone()`
- Wrap fitness evaluation in `@torch.no_grad()` to prevent gradient graph retention
- Use scalar extraction for fitness aggregation: `total_fitness += float(fitness_tensor)` not `+= fitness_tensor`
- Explicit cleanup per generation: `del intermediate_tensors; torch.cuda.empty_cache(); gc.collect()`
- Batch fitness evaluation with static shapes to enable CUDA graph optimization

### Expected Features

Research into high-performance evolutionary optimization reveals a clear feature hierarchy focused on evaluation time per generation. The existing system implements most table stakes features (checkpointing, batch evaluation, GPU acceleration, progress tracking, early exit filtering), but critical gaps remain in fitness caching, convergence detection, and cross-generation state reuse.

**Must have (table stakes):**
- Checkpointing & resume - EXISTING: RunManager.save_checkpoint() with torch RNG state
- Batch/parallel evaluation - EXISTING: PopulationEvaluator with concurrent_evaluations
- GPU acceleration - EXISTING: compute_all_metrics_batch_gpu_precomputed
- Progress tracking - EXISTING: ProgressTracker with JSONL logging
- Early exit for poor genomes - EXISTING: quarter-checkpoint at 25% with threshold filtering

**Should have (significant speedup):**
- Fitness caching/memoization - NOT IMPLEMENTED: 30-70% eval savings by caching duplicate/near-duplicate genomes
- Adaptive convergence detection - NOT IMPLEMENTED: Stop 20-40% early when evolution stagnates (sliding window on QD-score)
- Cross-generation embedding cache - PARTIALLY IMPLEMENTED: SharedPrecomputeContext exists but doesn't persist across generations (50-80% retrieval speedup available)
- Adaptive population sizing - NOT IMPLEMENTED: 20-50% speedup by scaling population based on diversity
- Generation-level profiling - EXISTING: GenerationProfiler with section timing and GPU memory tracking

**Defer (anti-features that hurt performance):**
- Per-individual GPU parallelism - causes memory fragmentation and CUDA context overhead (batch in single GPU call instead)
- Very large populations (>200) - diminishing diversity returns with O(N) evaluation cost
- Checkpointing every generation - I/O overhead becomes 5-10% of gen time (checkpoint every 5-10 gens or on improvement)
- Full logging of all metrics - JSON serialization overhead (use sample logging and aggregate stats)
- Extremely fine-grained early exit - overhead exceeds savings past single quarter checkpoint

### Architecture Approach

High-performance evolutionary systems follow a predictable architecture: tight generation loops with aggressive memory reuse, GPU-batched fitness evaluation, minimal state transfer between generations, and strategic profiling integration. The canonical structure is: BREED (CPU, ~3%), EVALUATE (GPU, ~90%), ARCHIVE (CPU, ~2%), LOG/TRACK (CPU, ~1%), CHECKPOINT (I/O, async). Evaluation dominates wall-clock time; all optimization effort should focus here first.

**Major components:**
1. **GenomeFactory** - Create/mutate/crossover genomes (CPU, never touches GPU, ephemeral per-generation lifecycle)
2. **SharedPrecomputeContext** - Query embeddings + initial pools allocated once per generation, provides shared GPU tensors to eliminate redundant computation
3. **BatchRetriever** - Multi-query traversal with GPU tensors, per-genome batch processing, strict cleanup after metrics
4. **MetricsComputer** - Batch fitness computation from retrieval results, single GPU kernel, immediate tensor cleanup
5. **Archive** - Store elite genomes (CPU only, never stores GPU tensors, long-lived data structure)
6. **GenerationProfiler** - Section-level timing and memory tracking to identify bottlenecks and validate optimizations

**Critical architecture patterns:**
- Allocate-late, free-early: Shared tensors allocated once at generation start, freed immediately after evaluation phase
- No tensor storage in long-lived objects: Archive stores only serializable dataclasses, never GPU tensors
- Context managers for GPU state: Ensure cleanup on exit, explicit `del` and `empty_cache()` calls
- Immutable genomes during evaluation: Mutations create copies to prevent archive corruption
- Hot path focus: 90% of time in evaluation, 70% of evaluation in retrieval, 20% in metrics - optimize in that order

### Critical Pitfalls

Research identified 10 critical pitfalls from project-specific bugs, PyTorch documentation, and industry best practices. The top 5 directly threaten the 500 gen / 3 hour target.

1. **Tensor reference retention in loops** - GPU memory grows across generations as intermediate tensors accumulate; avoid with explicit `del`, per-genome `torch.cuda.empty_cache()`, and conversion to Python scalars (`float(tensor)`) for aggregation
2. **Step-level buffer creation without reuse** - 625,000 allocations/generation causes severe fragmentation; pre-allocate buffers once at traversal start, reset with `.fill_()` instead of creating new tensors
3. **Shared context lifetime mismanagement** - Expanded copies for per-genome computation accumulate despite "optimization" of base sharing; delete expanded tensors immediately after metrics, call `empty_cache()` inside per-genome loops not outside
4. **Fitness history accumulation with gradient graphs** - Memory grows linearly with population when `total_fitness += genome_fitness` retains computation graphs; always use `float(genome_fitness)` or `fitness.detach()` for accumulation
5. **`empty_cache()` timing anti-pattern** - Too infrequent causes OOM before cleanup, too frequent loses warmed cache; call after each genome's full evaluation (not per-query, not per-generation)

**Warning signs requiring immediate action:**
- GPU memory growth (nvidia-smi or torch.cuda.memory_allocated()) across generations
- Later generations slower than early ones with identical population sizes
- `torch.cuda.memory_reserved()` >> `torch.cuda.memory_allocated()` (fragmentation indicator)
- Memory stable for first few genomes then grows within generation
- Memory proportional to population size rather than constant

## Implications for Roadmap

Based on research, the roadmap should prioritize hot path optimizations that directly reduce evaluation time, followed by memory management hardening to prevent crashes, with architectural refactoring deferred until performance targets are met. The 500 gen / 3 hour target requires 21.6 seconds per generation with 50-100 offspring, translating to 0.2-0.4 seconds per genome evaluation. Current bottleneck is evaluation time at ~94% of total runtime.

### Phase 1: Hot Path Optimization - Fitness Caching
**Rationale:** Highest ROI optimization (30-70% eval savings) with lowest implementation complexity. Fitness caching is P0 because MAP-Elites re-selects elite genomes and mutations produce duplicates. This phase delivers immediate speedup without architectural changes.
**Delivers:** FitnessCache module with content-hash based caching, genome hashing support, cache hit/miss tracking
**Addresses:** Fitness caching/memoization (P0 from FEATURES.md), repeated computation elimination
**Avoids:** Fitness history accumulation pitfall by forcing explicit detach on cached results
**Research flag:** Standard pattern (hash-based memoization), no additional research needed

### Phase 2: Hot Path Optimization - Cross-Generation Caching
**Rationale:** Second-highest ROI (50-80% retrieval savings) with low complexity. Query embeddings are constant across generations but recomputed. Extends existing SharedPrecomputeContext to persist across generation boundaries.
**Delivers:** PersistentPrecomputeContext with generation-independent lifetime, query hash validation, invalidation logic
**Addresses:** Cross-generation embedding cache (P1 from FEATURES.md)
**Uses:** Existing SharedPrecomputeContext infrastructure
**Research flag:** Standard pattern (cache persistence), no additional research needed

### Phase 3: Convergence & Early Stopping
**Rationale:** Stop 20-40% early when evolution stagnates. Sliding window convergence detection on QD-score with configurable thresholds. Minimal complexity, high value for long runs.
**Delivers:** ConvergenceDetector module with sliding window analysis, configurable stop criteria, integration with ProgressTracker
**Addresses:** Adaptive convergence detection (P1 from FEATURES.md)
**Avoids:** Archive accumulation pitfall by terminating before unbounded growth
**Research flag:** Standard pattern (sliding window on metrics), no additional research needed

### Phase 4: Memory Management Hardening
**Rationale:** Phases 1-3 add caching which increases memory pressure. This phase implements defensive guards and aggressive cleanup to prevent OOM crashes during long runs.
**Delivers:** Memory guards with per-generation assertions, explicit tensor lifecycle management, profiler integration for leak detection, buffer reuse patterns
**Addresses:** Memory stability requirement for 500-generation runs
**Avoids:** All top 5 pitfalls (tensor retention, buffer creation, context lifetime, gradient accumulation, cache timing)
**Research flag:** Implementation-specific (project debugging), may need profiling during development

### Phase 5: Adaptive Population & Advanced Optimization
**Rationale:** After stability is proven, adaptive population sizing and surrogate-assisted evaluation offer further speedup. These are complex features with higher risk; defer until core performance is validated.
**Delivers:** Adaptive population module, optional surrogate pre-screening, multi-level checkpointing
**Addresses:** Adaptive population sizing (P2), surrogate evaluation (P3), multi-level checkpointing (P4)
**Uses:** torch.compile with `mode="reduce-overhead"` for static shapes, potential EvoTorch migration
**Research flag:** NEEDS RESEARCH - surrogate model selection, population scaling strategies are domain-specific

### Phase Ordering Rationale

- **Hot paths first (Phases 1-2):** Fitness evaluation is 94% of runtime; optimizing here has multiplicative impact on total time
- **Early stopping before hardening (Phase 3):** Convergence detection reduces total work, making memory management simpler
- **Memory hardening before advanced features (Phase 4):** Stability is prerequisite for multi-hour runs; don't add complexity until baseline is solid
- **Defer architectural refactoring:** SwarmRetriever split, async checkpointing, and other structural changes should wait until performance targets are met to avoid premature optimization
- **Dependencies respected:** Cross-gen caching extends SharedPrecomputeContext (already exists), convergence uses ProgressTracker (already exists), memory guards require profiler integration (already exists)

### Research Flags

Phases with standard patterns (skip research-phase):
- **Phases 1-3:** Well-documented optimization patterns (hash-based caching, persistence, sliding window detection)
- **Phase 4:** Defensive programming patterns, can be implemented with existing PyTorch profiling tools

Phases likely needing deeper research during planning:
- **Phase 5:** Surrogate-assisted evaluation requires model selection (regression vs neural network), feature engineering for genome representation, and trade-off analysis between pre-screening cost and full evaluation savings

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Verified against official PyTorch 2.9 docs, EvoTorch 0.6.1 docs, and 2025 community practices |
| Features | HIGH | Based on QDax, MEMES algorithm papers, and fitness caching research; existing system analysis confirms gaps |
| Architecture | HIGH | Canonical patterns from GPU-accelerated EA literature; GenerationProfiler validates evaluation dominance (94%) |
| Pitfalls | HIGH | Derived from project BUG_REPORT.md (6 identified memory leaks), PyTorch FAQ, and EvoTorch best practices |

**Overall confidence:** HIGH

All research dimensions are supported by primary sources (official documentation, research papers, project-specific debugging). The evaluation bottleneck is measured (94% via GenerationProfiler), optimization ROI is documented in academic literature, and pitfalls are verified through actual bug fixes in the codebase.

### Gaps to Address

- **Surrogate model effectiveness:** Unknown if cheap proxy fitness can maintain search quality for this domain; needs validation with actual STARK benchmark during Phase 5 planning
- **Optimal cache sizing:** Fitness cache and embedding cache sizes need tuning based on actual mutation rates and diversity; monitor cache hit rates during implementation
- **torch.compile gains:** Expected 20-40% speedup from kernel fusion depends on graph break elimination; use `torch._dynamo.explain()` to validate compilation quality during Phase 5
- **EvoTorch migration decision:** Optional migration to EvoTorch for production MAP-Elites; benchmark against current implementation only if Phases 1-4 don't meet target (avoid premature rewrite)
- **Convergence threshold tuning:** Sliding window size (15 gens) and improvement threshold (0.1%) are literature defaults; may need domain-specific tuning for STARK benchmark

## Sources

### Primary (HIGH confidence)
- **PyTorch 2.9+ Official Documentation:** torch.compile, memory management, profiling, CUDA graphs
- **EvoTorch 0.6.1 Documentation:** GPU-vectorized MAP-Elites, problem parallelization, vectorized fitness patterns
- **Project BUG_REPORT.md:** 6 documented GPU memory leaks with reproduction steps and fixes
- **Project CONCERNS.md:** Tech debt audit identifying fragile areas and exception handling issues

### Secondary (MEDIUM confidence)
- **Research Papers:** EvoTorch paper (arXiv:2302.12600), MEMES algorithm (arXiv:2303.06137), QDax framework, GPU-accelerated GA survey
- **Community Resources:** ezyang's torch.compile guide (August 2025), vLLM integration blog, PyTorch optimization guides
- **Quality-Diversity Literature:** MAP-Elites surveys, fitness caching analysis (ACM 2018), termination detection strategies, adaptive population sizing

### Tertiary (LOW confidence)
- **pymoo Termination Criteria:** Reference implementation for convergence detection patterns (needs adaptation to MAP-Elites)
- **Python GC Tuning:** Threshold recommendations (need profiling to validate benefit for this workload)

### Aggregated Source Categories
- **Stack Research:** 8 official PyTorch docs, 5 EvoTorch resources, 4 research papers, 6 community guides
- **Features Research:** 8 papers/frameworks on QD algorithms, fitness caching, termination criteria, population sizing
- **Architecture Research:** 4 GPU-accelerated EA papers, 3 PyTorch memory guides, 3 MAP-Elites resources, 3 profiling guides
- **Pitfalls Research:** Project bug reports, PyTorch FAQ, 4 memory leak diagnosis guides, 3 EA-specific pitfall papers

---
*Research completed: 2026-01-29*
*Ready for roadmap: yes*
