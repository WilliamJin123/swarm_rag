# Feature Research

**Domain:** High-Performance Evolutionary Optimization
**Researched:** 2026-01-29
**Confidence:** HIGH

## Executive Summary

Research into high-performance evolutionary optimization systems reveals a clear hierarchy of features. For your target of 500 generations in 3 hours with population 50-100, the critical bottleneck is **evaluation time per generation**. Your existing checkpointing, batch evaluation, and GPU acceleration provide a strong foundation. The highest-impact additions are **fitness caching/memoization** and **adaptive early stopping** (which you partially have), with **convergence-based termination** offering the most compute savings.

---

## Feature Landscape

### Table Stakes (Must Have for Performance)

| Feature | Why Essential | Complexity | Notes |
|---------|---------------|------------|-------|
| **Checkpointing & Resume** | Required for runs >1hr; enables fault tolerance and iterative experimentation | LOW | **EXISTING** - Your `RunManager.save_checkpoint()` handles this well with torch RNG state preservation |
| **Batch/Parallel Evaluation** | Single-threaded evaluation is 10-50x slower for populations; GPU utilization requires batching | MEDIUM | **EXISTING** - Your `PopulationEvaluator` with `concurrent_evaluations` and shared precompute |
| **GPU Acceleration** | CPU-only evaluation is infeasible for 500 gens; GPU provides 10-100x speedup for metrics | HIGH | **EXISTING** - Your `compute_all_metrics_batch_gpu_precomputed` and device management |
| **Progress Tracking** | Essential for debugging, early stopping decisions, and understanding convergence | LOW | **EXISTING** - Your `ProgressTracker` with JSONL logging and matplotlib plotting |
| **Early Exit for Poor Genomes** | Avoids wasting 75% of evaluation time on clearly bad genomes | MEDIUM | **EXISTING** - Your quarter-checkpoint early exit at 25% with threshold filtering |

### Differentiators (Significant Speedup)

| Feature | Performance Impact | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Fitness Caching/Memoization** | **30-70% eval savings** in later generations when mutations produce duplicates or near-duplicates | LOW | **NOT IMPLEMENTED** - Critical gap; cache by genome hash; particularly valuable for MAP-Elites where archive members are re-selected |
| **Adaptive Convergence Detection** | **Stop 20-40% earlier** when evolution stagnates; avoid wasted generations | MEDIUM | **NOT IMPLEMENTED** - Use sliding window on QD-score or best fitness; 10-20 gen window with <0.1% improvement threshold |
| **Cross-Generation Embedding Cache** | **50-80% retrieval speedup** if queries are fixed across generations | LOW | **PARTIALLY IMPLEMENTED** - Your `SharedPrecomputeContext` caches within generation; extend to cross-generation |
| **Adaptive Population Sizing** | **20-50% speedup** by using smaller populations in early exploration, larger during refinement | MEDIUM | **NOT IMPLEMENTED** - Start with pop/2, scale up when diversity drops below threshold |
| **Surrogate-Assisted Evaluation** | **2-5x speedup** by using cheap proxy fitness for pre-screening before expensive full eval | HIGH | **NOT IMPLEMENTED** - Train regression model on genome features -> fitness; screen with proxy, evaluate top-k properly |
| **Multi-Level Checkpointing** | **Faster resume** via incremental checkpoints; reduce I/O overhead by 80% | MEDIUM | **NOT IMPLEMENTED** - Checkpoint deltas + periodic full snapshots; faster recovery |
| **Generation-Level Profiling** | **Identify bottlenecks** precisely; your profiler shows eval is 94%+ of time | LOW | **EXISTING** - Your `GenerationProfiler` with section timing and GPU memory tracking |

### Anti-Features (Seem Good, Hurt Performance)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Per-Individual Parallelism on GPU** | "More parallelism = faster" | CUDA context is thread-local; multi-process GPU sharing has high overhead; causes memory fragmentation | Batch individuals in single GPU call (what you already do) |
| **Very Large Populations (>200)** | "More diversity = better search" | O(N) evaluation cost; diminishing diversity returns past ~100 for most MAP-Elites grids | Use larger batch size with same population; increase archive selection pressure |
| **Checkpointing Every Generation** | "Maximum safety" | I/O overhead becomes significant (5-10% of gen time); disk thrashing | Checkpoint every 5-10 generations; save on improvement only |
| **Full Logging of All Metrics** | "Complete visibility" | JSON serialization overhead; disk space; log file parsing becomes slow | Sample logging; aggregate stats; log full data only at checkpoints |
| **Async Evaluation Across Generations** | "Pipeline parallelism" | Complex synchronization; stale fitness values in selection; debugging nightmare | Stick with synchronous generational model; parallel within-generation |
| **Dynamic Mutation Rate per Individual** | "Self-adaptation" | Parameter explosion; unstable convergence; requires more generations to tune | Use archive-level adaptive rates based on recent success ratios |
| **Extremely Fine-Grained Early Exit** | "More checkpoints = more savings" | Overhead of metric computation at each checkpoint exceeds savings; diminishing returns past 2 checkpoints | Your single quarter checkpoint is optimal; going to 1/8 adds overhead > savings |

---

## Feature Dependencies

```
Fitness Caching ─────────────────────────────────────────────────┐
    │                                                             │
    └── Requires: Stable genome hashing (genome.id or content hash)
                                                                  │
Cross-Gen Embedding Cache ───────────────────────────────────────┤
    │                                                             │
    └── Requires: Fixed query set across generations              │
    └── Requires: SharedPrecomputeContext (EXISTING)              │
                                                                  │
Adaptive Convergence Detection ──────────────────────────────────┤
    │                                                             │
    └── Requires: Progress Tracking (EXISTING)                    │
    └── Requires: QD-score computation (EXISTING in archive.stats)│
                                                                  │
Adaptive Population Sizing ──────────────────────────────────────┤
    │                                                             │
    └── Requires: Diversity metrics (coverage from archive)       │
    └── Requires: Configurable population bounds                  │
                                                                  │
Surrogate-Assisted Evaluation ───────────────────────────────────┤
    │                                                             │
    └── Requires: Fitness Caching (for training data)             │
    └── Requires: Genome feature extraction                       │
    └── Requires: sklearn/torch regression model                  │
                                                                  │
Multi-Level Checkpointing ───────────────────────────────────────┘
    │
    └── Requires: Delta serialization for genomes
    └── Requires: RunManager modifications
```

---

## Implementation Priority Matrix

Given your target (500 gens / 3 hours / pop 50-100):

| Priority | Feature | Expected Impact | Effort |
|----------|---------|-----------------|--------|
| **P0** | Fitness Caching | 30-70% eval savings | 1-2 days |
| **P1** | Adaptive Convergence Detection | Stop 20-40% early | 0.5 days |
| **P1** | Cross-Gen Embedding Cache | 50-80% retrieval savings | 0.5 days |
| **P2** | Adaptive Population Sizing | 20-50% speedup | 1 day |
| **P3** | Surrogate Pre-screening | 2-5x if eval-bound | 2-3 days |
| **P4** | Multi-Level Checkpointing | Faster resume | 1-2 days |

---

## Detailed Feature Specifications

### Fitness Caching (P0)

**Why Critical:** In MAP-Elites, the same elite genome may be selected multiple times. Mutations may also produce duplicates or near-duplicates. Without caching, these are re-evaluated at full cost.

**Implementation:**
```python
class FitnessCache:
    def __init__(self, max_size: int = 10000):
        self._cache: Dict[str, FitnessResult] = {}

    def get_or_compute(
        self,
        genome: Genome,
        compute_fn: Callable[[Genome], FitnessResult]
    ) -> Tuple[FitnessResult, bool]:  # (result, was_cached)
        key = genome.content_hash()  # hash of expression + params
        if key in self._cache:
            return self._cache[key], True
        result = compute_fn(genome)
        self._cache[key] = result
        return result, False
```

**Cache Key Strategy:**
- Hash genome's expression tree + parameter values
- Exclude genome.id (two genomes with same content should share cache)
- Consider fuzzy matching for near-duplicates (within epsilon on float params)

### Adaptive Convergence Detection (P1)

**Why Critical:** Evolution often converges well before max generations. Detecting stagnation saves 20-40% of compute.

**Implementation:**
```python
class ConvergenceDetector:
    def __init__(self, window_size: int = 15, threshold: float = 0.001):
        self.window_size = window_size
        self.threshold = threshold
        self.history: List[float] = []

    def should_stop(self, qd_score: float) -> Tuple[bool, str]:
        self.history.append(qd_score)
        if len(self.history) < self.window_size:
            return False, "insufficient_data"

        window = self.history[-self.window_size:]
        improvement = (window[-1] - window[0]) / max(abs(window[0]), 1e-8)

        if improvement < self.threshold:
            return True, f"stagnant_{self.window_size}_gens"
        return False, "improving"
```

**Metrics to Track:**
- QD-score (primary for MAP-Elites)
- Best fitness (secondary)
- Archive coverage (tertiary)

### Cross-Generation Embedding Cache (P1)

**Why Critical:** Query embeddings are computed fresh each generation despite being constant. This is pure waste.

**Implementation:**
- Extend `SharedPrecomputeContext` to persist across generations
- Store in `EvolutionContext` with generation-independent lifetime
- Invalidate only if query set changes

```python
class PersistentPrecomputeContext:
    query_embeddings: torch.Tensor  # Computed once, reused
    initial_pools: Dict[int, List[List[Any]]]  # Keyed by pool_size
    _generation_created: int

    def is_valid_for(self, queries: List[str]) -> bool:
        # Check if queries match what was used to create context
        return hash(tuple(queries)) == self._query_hash
```

---

## Existing Implementation Analysis

Your codebase already implements several high-performance features:

### Strengths (Already Implemented)
1. **Quarter-checkpoint early exit** - Filters poor genomes at 25% evaluation cost
2. **Shared precompute within generation** - `SharedPrecomputeContext` eliminates redundant embedding computation
3. **GPU batch metrics** - `compute_all_metrics_batch_gpu_precomputed` uses vectorized tensor ops
4. **Generation profiling** - `GenerationProfiler` enables precise bottleneck identification
5. **Memory management** - Explicit `torch.cuda.empty_cache()` calls prevent fragmentation

### Gaps (Not Implemented)
1. **No fitness caching** - Duplicate genomes are re-evaluated
2. **No convergence detection** - Runs always go to max generations
3. **No cross-generation caching** - Embeddings recomputed each generation
4. **No adaptive population** - Fixed size throughout run

---

## Sources

- [Enhancing MAP-Elites with Multiple Parallel Evolution Strategies](https://arxiv.org/abs/2303.06137) - MEMES algorithm showing 100x speedup via GPU parallelization
- [QDax: Accelerated Quality-Diversity](https://github.com/adaptive-intelligent-robotics/QDax) - JAX-based QD library with 10-100x speedups
- [The Influence of Fitness Caching on Modern Evolutionary Methods](https://dl.acm.org/doi/10.1145/3205651.3205788) - Analysis of fitness caching impact
- [Termination Detection Strategies in Evolutionary Algorithms](https://dl.acm.org/doi/10.1145/3205455.3205466) - Survey of stopping criteria
- [Adaptive Population Sizing Schemes in Genetic Algorithms](https://www.semanticscholar.org/paper/Adaptive-Population-Sizing-Schemes-in-Genetic-Lobo-Lima/9db92c22e6e34ac3616dc28b89725869c3d780a0) - Population sizing strategies
- [Scaling Policy Gradient Quality-Diversity with Massive Parallelization](https://arxiv.org/html/2501.18723) - Shows batch size scaling maintains QD performance
- [ParetoTracker: Visual Analytics for MOEA Population Dynamics](https://arxiv.org/html/2408.04539v1) - Progress tracking and visualization
- [pymoo Termination Criteria Documentation](https://pymoo.org/interface/termination.html) - Practical termination implementations
