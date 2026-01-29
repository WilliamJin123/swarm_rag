# Pitfalls Research

**Domain:** Evolutionary Optimization Performance
**Researched:** 2026-01-29
**Confidence:** HIGH (based on project-specific bugs, PyTorch documentation, and industry best practices)

---

## Critical Pitfalls

### Pitfall 1: Tensor Reference Retention in Loops

**What goes wrong:**
Intermediate GPU tensors created inside evaluation loops maintain references through Python variables, preventing CUDA memory from being freed. Over hundreds of generations, memory fragments and eventually exhausts.

**Why it happens:**
- PyTorch's caching allocator keeps tensors in GPU memory until Python references are deleted
- Variables assigned in loop bodies (e.g., `results_tensor = compute(...)`) persist until scope exit
- Expanded/broadcast tensors (`tensor.expand()`) create views that keep originals alive
- List/dict accumulation of tensor references (`all_results.append(tensor)`) builds unbounded history

**How to avoid:**
1. Explicitly `del` intermediate tensors after use: `del expanded_tensor`
2. Call `torch.cuda.empty_cache()` after each genome evaluation, not just at loop end
3. Pre-allocate reusable buffers once per generation, reuse with `.fill_()` or `.zero_()`
4. Convert tensors to Python types immediately when values are final: `total += float(loss_tensor)`

**Warning signs:**
- GPU memory (via `nvidia-smi` or `torch.cuda.memory_allocated()`) grows monotonically across generations
- Later generations run slower than early ones with identical population sizes
- OOM errors that don't occur on small test runs

**Phase to address:** Performance Testing Phase - implement memory guards with per-generation assertions

---

### Pitfall 2: Step-Level Buffer Creation Without Reuse

**What goes wrong:**
Per-step buffers (index mappings, score accumulators, feature tensors) are allocated fresh each step within traversal loops. With 25 agents x 5 steps x 100 queries x 50 genomes = 625,000 allocations per generation, this causes severe fragmentation.

**Why it happens:**
- Natural coding pattern: create tensor where needed, trust GC
- PyTorch's allocator fragments memory when allocation sizes vary
- Developers underestimate cumulative cost of "small" per-step allocations

**How to avoid:**
1. Pre-allocate all per-query buffers once at traversal start
2. Use `.fill_(-1)` or `.zero_()` to reset instead of creating new tensors
3. Maintain a buffer pool keyed by (dtype, size) for reuse across queries
4. Profile with `torch.cuda.memory_stats()` to detect fragmentation

**Warning signs:**
- `torch.cuda.memory_reserved()` >> `torch.cuda.memory_allocated()` (fragmentation)
- Allocation time increases over generations
- Memory spikes during traversal that don't recede

**Phase to address:** Core Development Phase - design buffer reuse into traversal architecture

---

### Pitfall 3: Random Tensor Generation Without Output Buffers

**What goes wrong:**
Calls like `torch.rand_like(scores)` or `torch.randn(size, device="cuda")` allocate new GPU memory each invocation. In hot paths (heuristic evaluation, mutation noise), this creates thousands of short-lived tensors.

**Why it happens:**
- Convenience APIs (`torch.rand_like`) don't support output buffers
- Random jitter/exploration noise added as afterthought
- Developers assume small tensors are "free"

**How to avoid:**
1. Pre-allocate jitter buffers: `self._jitter_buffer = torch.empty_like(template)`
2. Use `torch.rand(size, out=buffer)` where available
3. For `rand_like`, manually: `buffer.uniform_()` on pre-allocated tensor
4. Consider CPU random generation with single GPU transfer for non-critical paths

**Warning signs:**
- Profiler shows many small allocations in heuristic/mutation code
- Feature registry or scoring functions dominate allocation counts
- Memory churn (high alloc + high free counts) in `torch.cuda.memory_stats()`

**Phase to address:** Optimization Phase - audit all random generation in hot paths

---

### Pitfall 4: Shared Context Lifetime Mismanagement

**What goes wrong:**
Shared pre-compute contexts (embeddings, ground truth tensors, initial pools) are created once per generation but expanded copies for per-genome computation are never cleaned up. The "optimization" of sharing computation backfires when derived tensors accumulate.

**Why it happens:**
- Shared context is a good optimization for base computation
- But broadcast/expand operations for per-genome use create new tensor views
- Developers assume "views" are free (they reference original memory but prevent GC)
- `empty_cache()` called at wrong granularity (generation-end vs genome-end)

**How to avoid:**
1. Delete expanded tensors immediately after metrics computation
2. Call `torch.cuda.empty_cache()` inside per-genome loops, not outside
3. Use explicit scope boundaries with context managers for tensor lifetime
4. Consider recomputing cheap expansions vs caching expensive ones

**Warning signs:**
- Memory stable for first few genomes, then grows within generation
- Shared precompute module shows growing memory despite "sharing"
- Profiler shows memory delta positive for every genome evaluation

**Phase to address:** Integration Phase - instrument shared context with memory guards

---

### Pitfall 5: Fitness History Accumulation with Gradient Graphs

**What goes wrong:**
When computing aggregate fitness (`total_fitness += genome_fitness`), if `genome_fitness` is a tensor with gradient history, the entire computation graph is retained. Memory grows linearly with population size.

**Why it happens:**
- PyTorch tensors retain computation graphs by default for autograd
- Summation operations like `+=` on tensors preserve history
- Even with `torch.no_grad()`, direct tensor accumulation retains references
- Common in training loops, transferred to evolution without adaptation

**How to avoid:**
1. Always convert to Python scalar: `total_fitness += float(genome_fitness)`
2. Use `.detach()` before any accumulation: `fitness_list.append(f.detach())`
3. Wrap evaluation in `torch.no_grad()` context (but still need explicit detach for accumulation)
4. Use `torch.stack()` on detached list at end instead of incremental addition

**Warning signs:**
- Memory proportional to population size (not constant)
- Memory freed only when fitness aggregation variable goes out of scope
- OOM on large populations that work with small ones

**Phase to address:** Core Development Phase - establish fitness handling conventions early

---

### Pitfall 6: Pheromone/State Buffer Unbounded Growth

**What goes wrong:**
State buffers (pheromone tables, visitation counts, score caches) grow to accommodate the largest seen case and never shrink. After processing a few large graphs, memory stays high even for small ones.

**Why it happens:**
- Buffers sized to `max_node_id + padding` for safety
- Minimum sizes hardcoded for perceived efficiency
- No shrinking logic because "reallocation is expensive"
- Hashtables/dicts grow but never release capacity

**How to avoid:**
1. Size buffers to actual graph size, not arbitrary minimums
2. Implement periodic buffer right-sizing (e.g., every 10 generations)
3. Use weak references for caches that should be evictable
4. Track buffer memory separately and alert on growth

**Warning signs:**
- Memory baseline shifts upward after processing different datasets
- Buffer sizes in profiler don't match actual data sizes
- Memory high even when processing empty/trivial inputs

**Phase to address:** Production Hardening Phase - add adaptive buffer sizing

---

### Pitfall 7: Thread Pool Over-Subscription

**What goes wrong:**
Thread pools created with generous limits (`max_workers=32`) compete for CPU resources. Combined with PyTorch's internal parallelism, this causes context switching overhead and cache thrashing.

**Why it happens:**
- Assumption: "more threads = more parallelism = faster"
- PyTorch internally uses thread pools for CPU operations
- NumPy and other libraries also spawn threads
- Total thread count = explicit pools + PyTorch + NumPy + system

**How to avoid:**
1. Cap thread pools to `min(requested, os.cpu_count() // 2)`
2. Set `torch.set_num_threads()` explicitly to avoid interference
3. Profile with system tools to see actual thread count
4. Consider process-based parallelism for CPU-bound work

**Warning signs:**
- CPU utilization near 100% but throughput plateaus or decreases
- High system time (context switching) in profiler
- Performance inversely correlated with worker count past a point

**Phase to address:** Performance Tuning Phase - establish parallelism configuration guidelines

---

### Pitfall 8: Expression Tree Interpreter Overhead

**What goes wrong:**
Expression trees (for genome strategies) are evaluated interpretively for each genome on each query. The Python interpreter overhead dominates actual computation time as trees grow.

**Why it happens:**
- Expression trees provide flexibility and evolvability
- Initial implementation uses recursive Python evaluation
- "It works" mindset delays compilation optimization
- Tree structure varies per genome, complicating batching

**How to avoid:**
1. Cache compiled versions of frequently-used subtrees
2. Compile entire trees to PyTorch operations using `torch.jit.script`
3. Batch evaluation across genomes with similar tree structures
4. Consider switching to weighted-sum mode for production runs

**Warning signs:**
- CPU time dominates GPU time despite GPU evaluation
- Evaluation time grows superlinearly with tree depth
- Profiler shows most time in tree traversal, not tensor ops

**Phase to address:** Optimization Phase - implement tree compilation or mode switching

---

### Pitfall 9: `empty_cache()` Timing Anti-Pattern

**What goes wrong:**
`torch.cuda.empty_cache()` is called too infrequently (only at generation boundaries) or too frequently (every operation). Both hurt performance.

**Why it happens:**
- Too infrequent: memory accumulates, OOM before cleanup point
- Too frequent: CUDA allocator loses warmed cache, reallocation overhead
- Developers cargo-cult the call without understanding allocator behavior

**How to avoid:**
1. Call after completing each genome's full evaluation (not per-query, not per-generation)
2. Pair with explicit `del` of large tensors
3. Monitor actual memory delta to tune frequency
4. Use memory profiler to find optimal placement

**Warning signs:**
- Memory spikes between cache clears exceed available GPU RAM
- Performance degrades when adding `empty_cache()` calls
- Inconsistent timing - some generations fast, some slow

**Phase to address:** Performance Testing Phase - establish cache clearing cadence with benchmarks

---

### Pitfall 10: Archive/History Unbounded Accumulation

**What goes wrong:**
MAP-Elites archives, fitness history, decision logs, and profiler data grow without bounds over long runs. Eventually, CPU memory is exhausted even if GPU is fine.

**Why it happens:**
- Archives designed to "never forget" elite solutions
- Logging/tracing added for debugging, never removed
- Rolling windows have maximums but no enforcement
- History kept for "analysis later" that never happens

**How to avoid:**
1. Implement hard caps on archive size with oldest-eviction
2. Use rolling windows with strict size enforcement
3. Periodic serialization to disk with in-memory pruning
4. Make history retention configurable, default to conservative limits

**Warning signs:**
- CPU memory grows even when GPU is stable
- Python process RSS increases linearly with generations
- Checkpoint files grow larger each save

**Phase to address:** Production Hardening Phase - implement archive management policies

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Profiling overhead in hot paths | Profiler data collection > actual work | Disable profiling in production; use sampling | Always-on profiling in release builds |
| Synchronous GPU operations | Low GPU utilization, high latency | Use async operations, avoid `.item()` in loops | Every `.cpu()` or `.item()` forces sync |
| CPU-GPU data ping-pong | Memory bandwidth saturation | Keep data on one device for entire pipeline | Per-step device transfers |
| Over-eager tensor creation | Allocation dominates computation | Pre-allocate, reuse, pool buffers | Dynamic shapes in inner loops |
| Silent dtype promotion | Memory doubles unexpectedly | Explicit dtypes, use `torch.set_default_dtype()` | Mixed float32/float64 operations |
| Logging in hot loops | I/O blocks compute | Use async logging, reduce verbosity | DEBUG logging enabled in production |
| Repeated identical computations | CPU waste | Memoize/cache deterministic functions | Same query evaluated multiple times |
| Large batch sizes exceeding memory | OOM on first batch | Start small, increase adaptively | Batch size tuned on different hardware |
| Checkpoint serialization blocking | Periodic stalls | Async checkpoint writes | Large models with frequent saves |
| Exception handling in hot paths | Try/except overhead | Validate upfront, avoid exceptions for control flow | Per-element exception handling |

---

## "Looks Done But Isn't" Checklist

These optimizations appear complete but have hidden issues:

### Memory Management
- [ ] **Pre-allocation exists but not verified** - Buffers are created but still recreated somewhere in the path
- [ ] **`del` statements present but after scope would GC anyway** - Explicit deletes that don't actually help
- [ ] **`empty_cache()` called but memory still grows** - Tensor references retained elsewhere
- [ ] **Shared context "optimizes" but derived tensors leak** - Base optimization, derived accumulation

### Parallelism
- [ ] **Thread pool exists but workers idle** - Work not actually parallelizable
- [ ] **GPU utilization reported high but includes idle spin** - Check compute vs memory bound
- [ ] **Batch processing implemented but batch size = 1** - Configuration not updated

### Profiling
- [ ] **Profiler shows fast operations but end-to-end slow** - Profiler overhead or missing sections
- [ ] **Hotspot identified but optimization had no effect** - Hotspot was symptom, not cause
- [ ] **Memory "stable" but baseline keeps shifting** - Looking at wrong metric (reserved vs allocated)

### Correctness vs Performance
- [ ] **Fast but wrong** - Optimization changed semantics (e.g., removed necessary sync)
- [ ] **Caching works on test data but not real data** - Cache keys don't capture all variation
- [ ] **Parallel version matches serial for simple cases** - Race conditions in complex cases

### Evolution-Specific
- [ ] **Fitness evaluation "vectorized" but genome-by-genome inside** - Outer batch, inner loop
- [ ] **Early exit saves time but kills diversity** - Good genomes exited because of noise
- [ ] **Shared precompute exists but most genomes don't share** - Pool sizes vary, no sharing benefit
- [ ] **Archive pruning implemented but threshold never reached** - Configured for larger runs

---

## Detection Strategies

### Memory Monitoring Protocol

```python
# Minimum viable memory monitoring for evolution loops
import torch

def assert_memory_stable(gen: int, baseline_mb: float, tolerance_mb: float = 50):
    """Call at end of each generation."""
    if torch.cuda.is_available():
        current_mb = torch.cuda.memory_allocated() / 1024 / 1024
        if current_mb > baseline_mb + tolerance_mb:
            raise MemoryError(
                f"Gen {gen}: Memory drift detected. "
                f"Baseline: {baseline_mb:.1f}MB, Current: {current_mb:.1f}MB"
            )
```

### Latency Regression Detection

```python
# Track per-generation timing to catch regressions
def assert_latency_stable(gen: int, gen_time_ms: float, baseline_ms: float, tolerance_pct: float = 20):
    """Call at end of each generation."""
    max_allowed = baseline_ms * (1 + tolerance_pct / 100)
    if gen_time_ms > max_allowed:
        raise PerformanceError(
            f"Gen {gen}: Latency regression. "
            f"Baseline: {baseline_ms:.0f}ms, Current: {gen_time_ms:.0f}ms"
        )
```

---

## Sources

### Project-Specific
- Bug report: `docs/BUG_REPORT.md` - 6 critical GPU memory leaks identified
- Concerns audit: `.planning/codebase/CONCERNS.md` - Tech debt and fragile areas
- Fix TODOs: `.planning/todos/done/2026-01-29-fix-gpu-memory-*.md` - Applied fixes

### PyTorch Official
- [Understanding GPU Memory 2: Reference Cycles](https://pytorch.org/blog/understanding-gpu-memory-2/)
- [PyTorch FAQ: GPU Memory Management](https://docs.pytorch.org/docs/stable/notes/faq.html)

### Industry Research
- [Gaggle: Genetic Algorithms on GPU](https://dl.acm.org/doi/abs/10.1145/3583133.3596356)
- [EvoTorch: Scalable Evolutionary Computation](https://arxiv.org/pdf/2302.12600)
- [EvoGP: GPU-accelerated Genetic Programming](https://arxiv.org/html/2501.17168v1)

### Best Practices
- [PyTorch Gotchas](https://coolnesss.github.io/2019-02-05/pytorch-gotchas)
- [Premature Optimization](https://www.geeksforgeeks.org/software-engineering/premature-optimization/)
- [Memory Leak Diagnosis in Python](https://www.tutorialspoint.com/python/python_diagnosing_and_fixing_memory_leaks.htm)
- [Gradient Accumulation Memory Issues](https://medium.com/biased-algorithms/gradient-accumulation-in-pytorch-36962825fa44)

---

*Research completed: 2026-01-29*
