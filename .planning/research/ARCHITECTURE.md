# Architecture Research

**Domain:** Performant Evolution Loop Architecture
**Researched:** 2026-01-29
**Confidence:** HIGH

## Executive Summary

High-performance evolutionary optimization systems follow a predictable architecture pattern: tight generation loops with aggressive memory reuse, GPU-batched fitness evaluation, minimal state transfer between generations, and strategic profiling integration. The existing SwarmRAG system has many of these components but can benefit from restructuring around cleaner component boundaries and explicit hot path optimization.

---

## Standard Architecture for Fast Evolution

### Generation Loop Structure

A high-performance generation loop follows this canonical structure:

```
GENERATION LOOP (per-gen)
  1. BREED         [CPU, ~3% time]  - Select parents, apply mutation/crossover
  2. EVALUATE      [GPU, ~90% time] - Batch fitness computation (HOT PATH)
  3. ARCHIVE       [CPU, ~2% time]  - Insert elites, compute descriptors
  4. LOG/TRACK     [CPU, ~1% time]  - Update metrics, journal
  5. CHECKPOINT    [I/O, async]     - Background save, non-blocking
```

**Key Insight:** Evaluation dominates wall-clock time (90%+). All optimization effort should focus here first. The existing system correctly identifies this via the GenerationProfiler design.

**Canonical Data Flow:**
```
Population (in-memory)
    |
    v
[Breed Phase] -- offspring genomes (lightweight objects)
    |
    v
[Compile Phase] -- retrieval kwargs (one-time per genome)
    |
    v
[Shared Precompute] -- query embeddings + initial pools (ONE GPU allocation)
    |
    v
[Batch Retrieval] -- batched traversal results (GPU tensors)
    |
    v
[Batch Metrics] -- fitness scores (single GPU kernel)
    |
    v
[Archive Insert] -- updated archive (no GPU tensors stored)
```

### Component Boundaries

The optimal architecture separates concerns into these modules:

| Component | Responsibility | GPU Touches | Memory Lifecycle |
|-----------|---------------|-------------|------------------|
| **GenomeFactory** | Create/mutate/crossover genomes | Never | Ephemeral per-gen |
| **GenomeCompiler** | Genome -> retrieval kwargs | Never | Cache until mutation |
| **SharedPrecompute** | Query embeddings + initial pools | Once per gen | Clear after evaluation |
| **BatchRetriever** | Multi-query traversal | Per genome batch | Clear after metrics |
| **MetricsComputer** | Batch fitness from results | Once per gen | Clear immediately |
| **Archive** | Store elite genomes | Never | Long-lived |
| **Checkpoint** | Async I/O | Never | Background thread |

**Current vs Optimal:**
- Current: SharedPrecomputeContext partially implements this but tensors persist across genomes
- Optimal: Strict cleanup after each phase with explicit `del` and `torch.cuda.empty_cache()`

### Memory Lifecycle

High-performance evolutionary systems follow the **allocate-late, free-early** principle:

#### Phase 1: Pre-Evaluation (Allocate Once)
```python
# Allocate shared tensors ONCE at generation start
query_embeddings = embed_all_queries(queries)  # (n_queries, dim) on GPU
initial_pools = {pool_size: compute_pools(pool_size) for pool_size in unique_sizes}
gt_tensor, gt_sizes = precompute_ground_truth(ground_truth)  # GPU tensors
```

#### Phase 2: Per-Genome Evaluation (Reuse Buffers)
```python
# Reuse pre-allocated buffers, never allocate inside loop
for genome in offspring:
    results = retrieve_with_precomputed(query_embeddings, pools[genome.pool_size])
    # results tensor: temporary, freed before next genome
    metrics = compute_metrics_gpu(results, gt_tensor, gt_sizes)
    genome.fitness = metrics
    del results  # Explicit cleanup
    torch.cuda.empty_cache()  # Prevent fragmentation
```

#### Phase 3: Post-Evaluation (Free Immediately)
```python
# Free shared tensors IMMEDIATELY after evaluation phase
del query_embeddings, initial_pools, gt_tensor, gt_sizes
torch.cuda.empty_cache()
gc.collect()  # Collect Python reference cycles
```

**Anti-Pattern in Current Code:**
The BatchedRetrievalResults class accumulates results across genomes before batch metric computation. This is correct for cross-genome batching but requires explicit cleanup:
```python
batched_results.clear()  # Current code does this
del gt_tensor_expanded, gt_sizes_expanded  # Current code does this
torch.cuda.empty_cache()  # Current code does this
```

The cleanup is present but could be more aggressive with per-genome clearing.

---

## Hot Paths

### Priority 1: Fitness Evaluation (90%+ of time)

**Location:** `PopulationEvaluator._evaluate_all_with_shared()` and `_evaluate_batch_with_shared()`

**Subcomponents by Time:**
1. **Retrieval** (70% of evaluation): `retriever.retrieve_batch_with_precomputed()`
2. **Metric Computation** (20% of evaluation): `MetricFunctions.compute_all_metrics_batch_gpu_precomputed()`
3. **Result Collection** (10% of evaluation): Building result tensors, moving data

**Optimization Opportunities:**
- Multi-query GPU batching (already designed in `2026-01-28-multi-query-gpu-batching.md`)
- Cross-genome batching for metrics (already implemented)
- Avoid per-query memory allocation during traversal

### Priority 2: Swarm Traversal (70% of evaluation)

**Location:** `SwarmRetriever._retrieve_batch_multi_query_gpu()` (once implemented)

**Hot Operations:**
1. Neighbor lookup: `graph_store.get_neighbors_batch()`
2. Similarity computation: `torch.mm()` for batched dot products
3. Pheromone operations: Lookup and scatter_add for deposits

**Optimization Focus:**
- Keep all traversal state on GPU
- Use dense tensors with masking instead of sparse structures
- Preallocate position history tensor at max size

### Priority 3: Metric Computation (20% of evaluation)

**Location:** `MetricFunctions.compute_all_metrics_batch_gpu_precomputed()`

**Current State:** Already optimized with precomputed ground truth tensors.

**Further Optimization:** Could use torch.compile() on the metric kernels.

---

## State Management Patterns

### Pattern 1: Immutable Genomes

Genomes should be immutable during evaluation. Mutations create new copies:

```python
# Good: Clone before mutating
offspring = parent.copy()
mutate_in_place(offspring)

# Bad: Mutate shared reference
mutate_in_place(parent)  # Corrupts archive!
offspring = parent
```

**Current Implementation:** Genome.copy() exists and is used correctly.

### Pattern 2: Context Manager for GPU State

Use context managers to ensure cleanup:

```python
@contextmanager
def shared_evaluation_context(queries, ground_truth, device):
    """Allocate shared tensors, cleanup on exit."""
    context = prepare_shared_context(...)
    try:
        yield context
    finally:
        context.cleanup()  # Explicit tensor deletion
        torch.cuda.empty_cache()
        gc.collect()
```

**Gap in Current Code:** SharedPrecomputeContext has tensor cleanup but no context manager pattern.

### Pattern 3: Generation Scoped Counters

Reset counters at generation boundaries:

```python
class GenerationState:
    offspring_count: int = 0

    def reset(self):
        self.offspring_count = 0  # Reset per-gen counters
```

**Fixed in Current Code:** Recent commit `2026-01-29-fix-offspring-counter-never-reset.md` addressed this.

### Pattern 4: No Tensor Storage in Long-Lived Objects

Archive should never store GPU tensors:

```python
# Good: Store serializable data
archive.add(genome)  # genome.fitness is a dataclass, not tensor

# Bad: Store tensor references
archive.grid[cell].tensor = some_gpu_tensor  # Memory leak!
```

**Current Implementation:** Archive stores Genome objects with FitnessResult (dataclass), correct.

---

## Anti-Patterns

### Anti-Pattern 1: Implicit Tensor Accumulation

**Symptom:** GPU memory grows linearly with generations.

**Cause:** Storing tensors in lists/dicts that persist across generations.

```python
# Bad
all_results = []
for gen in generations:
    results = evaluate(...)
    all_results.append(results)  # Accumulates!
```

**Fix:** Clear results after each generation:
```python
for gen in generations:
    results = evaluate(...)
    process(results)
    del results
```

### Anti-Pattern 2: Synchronous Checkpointing

**Symptom:** Evolution pauses during checkpoint saves.

**Cause:** `torch.save()` blocks the main thread.

**Fix:** Async checkpointing (designed in `2026-01-26-evolution-optimization-design.md`):
```python
checkpoint_queue.put((state, path))  # Non-blocking
```

### Anti-Pattern 3: Reference Cycles with Tensors

**Symptom:** Tensors not freed despite going out of scope.

**Cause:** Python objects with circular references hold tensors.

```python
# Bad: Creates reference cycle
class Result:
    def __init__(self, tensor, parent):
        self.tensor = tensor
        self.parent = parent
        parent.child = self  # Cycle!
```

**Fix:** Use weakref or break cycles explicitly:
```python
import weakref
class Result:
    def __init__(self, tensor, parent):
        self.tensor = tensor
        self.parent = weakref.ref(parent)
```

**Detection:** Set `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8` and enable reference cycle detector.

### Anti-Pattern 4: Broad Exception Handling in Hot Path

**Symptom:** Silent failures, incorrect metrics, hard to debug.

**Cause:** `except Exception:` catches everything.

```python
# Bad
try:
    score = compute_score(tensor)
except Exception:
    score = 0.0  # Hides real bugs
```

**Fix:** Catch specific exceptions, fail fast in hot path:
```python
try:
    score = compute_score(tensor)
except (ValueError, RuntimeError) as e:
    logger.error(f"Score computation failed: {e}")
    raise  # Don't silently continue
```

### Anti-Pattern 5: Per-Query Allocation in Batch Loop

**Symptom:** High allocation overhead, memory fragmentation.

**Cause:** Creating new tensors inside per-query loops.

```python
# Bad
for query in queries:
    emb = torch.tensor([...], device=device)  # New allocation each time!
```

**Fix:** Pre-allocate and index:
```python
# Good
all_embs = torch.zeros((n_queries, dim), device=device)
for i, query in enumerate(queries):
    all_embs[i] = ...  # Write to pre-allocated slot
```

---

## Profiling Integration Points

### Generation-Level Profiling (Implemented)

`GenerationProfiler` provides section-level timing and memory tracking:

```python
profiler.start_generation(gen)
with profiler.section("breeding"): ...
with profiler.section("evaluation"): ...
print(profiler.end_generation())  # "[1247ms | GPU:2841MB] breed=42ms | eval=1180ms"
```

**Activation:** `EVOLUTION_PROFILE=1`

### Step-Level Profiling (Existing)

`StepProfiler` tracks hot sections within swarm traversal:
- Embedding lookup
- Neighbor fetch
- Score computation
- Pheromone operations

**Activation:** `SWARM_PROFILE=1`

### Memory Profiling Integration Points

**Suggested Additions:**

1. **Peak Memory Tracking:**
```python
torch.cuda.reset_peak_memory_stats()
# ... evaluation ...
peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
```

2. **Tensor Allocation Counting:**
```python
stats = torch.cuda.memory_stats()
allocations = stats['allocation.all.current']
```

3. **Memory Snapshot (Debugging):**
```python
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
```

---

## Population Data Structures

### Archive Grid (MAP-Elites)

**Structure:** Dict mapping behavioral descriptor tuples to elite genomes.

```python
grid: Dict[Tuple[int, ...], Genome] = {}
```

**Memory Characteristics:**
- O(bins^dimensions) potential cells
- Only filled cells consume memory
- Genome objects are lightweight (no tensors)

**Scaling Concern:** With 1000x1000 grid and 100% fill, could store 1M genomes.

**Mitigation:** Periodic pruning, secondary storage for old cells.

### Population List (Per-Generation)

**Structure:** List of offspring genomes for evaluation.

```python
offspring: List[Genome] = []  # Typically 10-50 per generation
```

**Memory Characteristics:**
- Small, ephemeral
- Cleared after archive insertion

### Batched Results (Temporary)

**Structure:** Results tensor for cross-genome metric batching.

```python
# (total_queries_all_genomes, max_k) on GPU
retrieved_ids: torch.Tensor
```

**Memory Characteristics:**
- Largest temporary allocation
- For 50 genomes, 100 queries, k=20: 50 * 100 * 20 * 8 bytes = 800KB
- Cleared immediately after metric computation

---

## Refactoring Priorities

Based on the analysis, here are the suggested refactoring priorities:

### Priority 1: Implement Multi-Query GPU Batching (HIGH IMPACT)

**Status:** Designed, not implemented
**Expected Impact:** 5-10x speedup on evaluation
**Effort:** Medium (tasks defined in design doc)

### Priority 2: Add Context Manager for SharedPrecomputeContext (MEDIUM IMPACT)

**Status:** Partial cleanup exists
**Expected Impact:** Prevent memory leaks, cleaner code
**Effort:** Low

```python
class SharedPrecomputeContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()
        torch.cuda.empty_cache()
        gc.collect()
```

### Priority 3: Implement Async Checkpointing (MEDIUM IMPACT)

**Status:** Designed, not implemented
**Expected Impact:** 5-10 minutes saved over 500 generations
**Effort:** Low (design complete)

### Priority 4: Split SwarmRetriever (LOW IMPACT, HIGH MAINTENANCE)

**Status:** Not started
**Expected Impact:** Better maintainability, easier testing
**Effort:** High (3097 lines to refactor)

Suggested split:
- `swarm_retriever.py`: Orchestration, public API
- `state_manager.py`: TraversalState operations
- `movement_logic.py`: Step execution, score computation
- `pheromone_ops.py`: Pheromone deposit, decay, lookup
- `result_ranking.py`: Final ranking and result formatting

### Priority 5: Specific Exception Handling in Hot Paths (LOW IMPACT)

**Status:** Flagged in CONCERNS.md
**Expected Impact:** Better debuggability
**Effort:** Low

---

## Recommended Architecture for 500 gens / 3 hours

To achieve 500 generations in 3 hours with population 50-100:

**Time Budget:**
- 3 hours = 10,800 seconds
- 500 generations = 21.6 seconds per generation
- With 50-100 offspring: 0.2-0.4 seconds per genome evaluation

**Required Speedups:**
1. Multi-query GPU batching: Process 100 queries in ~100ms instead of ~1000ms
2. Shared precompute: Eliminate redundant embedding computation (~10% savings)
3. Async checkpointing: Remove I/O from critical path (~5% savings)
4. Early exit at 25%: Filter bad genomes 3x faster (~20% savings for half of genomes)

**Architecture Constraints:**
- All shared tensors allocated once per generation
- No per-query memory allocation inside traversal loop
- Explicit cleanup after each evaluation phase
- Generation profiler enabled for monitoring

---

## Sources

### GPU-Accelerated Evolutionary Algorithms
- [EvoGP: A GPU-accelerated Framework for Tree-Based Genetic Programming](https://arxiv.org/html/2501.17168v1)
- [GPU-based island model for evolutionary algorithms](https://dl.acm.org/doi/10.1145/1830483.1830685)
- [EvoRL: A GPU-accelerated Framework for Evolutionary Reinforcement Learning](https://dl.acm.org/doi/10.1145/3750053)
- [Parallel Genetic Algorithms with GPU Computing](https://www.intechopen.com/chapters/69121)

### MAP-Elites and Quality-Diversity
- [Multi-Emitter MAP-Elites: Improving quality, diversity and convergence speed](https://arxiv.org/abs/2007.05352)
- [Quality Diversity Algorithms - Jean-Baptiste Mouret](https://members.loria.fr/jbmouret/qd.html)
- [MAP-Elites: Quality-Diversity Search](https://www.emergentmind.com/topics/map-elites-algorithm)
- [DCRL-MAP-Elites](https://github.com/adaptive-intelligent-robotics/DCRL-MAP-Elites)

### PyTorch Memory Management
- [Understanding GPU Memory 2: Finding and Removing Reference Cycles](https://pytorch.org/blog/understanding-gpu-memory-2/)
- [Memory Leakage with PyTorch](https://medium.com/@raghadalghonaim/memory-leakage-with-pytorch-23f15203faa3)
- [PyTorch Memory Management](https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/)
- [CUDA Caching Allocator GC-awareness Issue](https://github.com/pytorch/pytorch/issues/50185)

### Python Profiling
- [Profiling in Python: How to Find Performance Bottlenecks](https://realpython.com/python-profiling/)
- [10 Profilers to Unmask Python's Hot Paths](https://medium.com/@bhagyarana80/10-profilers-to-unmask-pythons-hot-paths-08fb0998e4a3)
- [Python Performance Profiling and Optimization](https://monadical.com/posts/python-performance-profiling.html)

### Evolutionary Computation Libraries
- [DEAP: Distributed Evolutionary Algorithms in Python](https://github.com/DEAP/deap)
- [PyPop7: Population-based Black-Box Optimization](https://github.com/Evolutionary-Intelligence/pypop)
- [pymoo: Multi-objective Optimization in Python](https://pymoo.org/)

---

*Architecture research completed: 2026-01-29*
