# Stack Research

**Domain:** PyTorch Evolutionary Optimization Performance
**Researched:** 2026-01-29
**Confidence:** HIGH (verified against official PyTorch 2.9/2.10 docs and 2025 community best practices)

## Executive Summary

This research focuses on optimizing a MAP-Elites evolutionary search system running on PyTorch 2.9.1. The target is 500 generations in 3 hours with population 50-100. Query latency (50ms) is acceptable; the bottleneck is evolution loop overhead and potential memory accumulation across generations.

Key findings:
1. **EvoTorch 0.6.1** provides production-ready MAP-Elites with GPU vectorization
2. **torch.compile with reduce-overhead** can eliminate Python/CUDA launch overhead
3. **Memory discipline** (detach, del, periodic gc) prevents generation-over-generation accumulation
4. **Batch fitness evaluation** with static shapes maximizes GPU utilization

---

## Recommended Stack

### Core Technologies (for optimization)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **EvoTorch** | 0.6.1 | GPU-vectorized evolutionary algorithms | Native MAP-Elites implementation that processes entire populations in batch. Eliminates serial mutation/evaluation loops. Built directly on PyTorch tensors. |
| **torch.compile** | PyTorch 2.9+ | JIT compilation | Fuses kernels, reduces Python overhead. Use `mode="reduce-overhead"` for CUDA graphs when shapes are static. |
| **Ray** | 2.x | Multi-process parallelization | EvoTorch's recommended parallelization backend. Zero-copy tensor sharing via shared memory. Better than multiprocessing for tensor workloads. |
| **PyTorch Memory Profiler** | Built-in 2.9+ | Memory leak detection | `torch.cuda.memory._record_memory_history()` + online visualizer for tracking allocations across generations. |

### Profiling Tools

| Tool | Purpose | When to Use |
|------|---------|-------------|
| **torch.profiler** | CPU/GPU timing, kernel analysis | Profile one evolution loop iteration to find bottlenecks. Use `profile_memory=True`. |
| **torch.cuda.memory._record_memory_history()** | Memory allocation traces | Run for 10-20 generations to detect memory accumulation patterns. |
| **torch._dynamo.explain()** | Graph break identification | Before deploying torch.compile to identify compilation barriers. |
| **nvidia-smi / nvitop** | GPU utilization monitoring | Continuous monitoring during evolution runs. Low utilization (<70%) indicates CPU-bound work. |
| **TORCH_LOG=perf_hints** | torch.compile optimization hints | Set as env var when running evolution to get compile-time warnings. |
| **torch.cuda.synchronize() + time.perf_counter()** | Accurate GPU timing | Wrap critical sections; don't trust wall-clock time without sync. |

### Memory Management Techniques

#### Preventing Memory Leaks in Long-Running Evolution

**1. Detach Tensors When Copying Across Generations**
```python
# BAD: Keeps computation graph attached
new_genome = old_genome.clone()

# GOOD: Breaks gradient history, creates independent copy
new_genome = old_genome.detach().clone()
```
**Confidence:** HIGH - This is the #1 cause of memory accumulation in evolutionary algorithms.

**2. Use Scalars for Fitness Aggregation**
```python
# BAD: Accumulates gradient history
total_fitness += fitness_tensor

# GOOD: Extract scalar value
total_fitness += float(fitness_tensor)
# or
total_fitness += fitness_tensor.item()
```
**Confidence:** HIGH - Documented in PyTorch FAQ as common mistake.

**3. Explicit Memory Cleanup Per Generation**
```python
def run_generation():
    # ... generation logic ...

    # End of generation cleanup
    del intermediate_tensors
    torch.cuda.empty_cache()  # Returns cached memory to CUDA
    gc.collect()  # Python cleanup
```
**Confidence:** MEDIUM - empty_cache() has overhead; profile to determine optimal frequency (every N generations vs every generation).

**4. Use torch.no_grad() for Fitness Evaluation**
```python
@torch.no_grad()
def evaluate_fitness(population):
    # No gradient tracking needed for evolutionary fitness
    return compute_metrics(population)
```
**Confidence:** HIGH - Evolutionary algorithms don't need gradients; this saves memory and compute.

**5. Control Variable Scope in Loops**
```python
# BAD: intermediates live until function returns
for gen in range(500):
    offspring = mutate(population)
    fitness = evaluate(offspring)
    population = select(offspring, fitness)
    # offspring still in scope!

# GOOD: Explicit scope control
for gen in range(500):
    population = run_single_generation(population)  # All intermediates go out of scope
```
**Confidence:** HIGH - Python scoping keeps loop variables alive.

**6. GC Tuning for Long-Running Loops**
```python
import gc

# Option A: Disable GC during generation, run manually
gc.disable()
for gen in range(500):
    run_generation()
    if gen % 10 == 0:
        gc.collect()
gc.enable()

# Option B: Increase thresholds to reduce GC frequency
gc.set_threshold(50000, 500, 100)  # Default is (700, 10, 10)
```
**Confidence:** MEDIUM - Only tune after profiling shows GC overhead.

### GPU Optimization Techniques

#### 1. Batch Fitness Evaluation (Critical)

**Problem:** Evaluating fitness one genome at a time underutilizes GPU.

**Solution:** EvoTorch's vectorized fitness function pattern:
```python
from evotorch import Problem

def vectorized_fitness(solutions: torch.Tensor) -> torch.Tensor:
    """
    Args:
        solutions: Shape [batch_size, genome_length]
    Returns:
        fitnesses: Shape [batch_size]
    """
    # All solutions evaluated in parallel
    return compute_batch_fitness(solutions)

problem = Problem(
    "max",
    vectorized_fitness,
    solution_length=genome_length,
    num_actors=4,  # Ray parallelization
    device="cuda:0"  # GPU execution
)
```
**Confidence:** HIGH - This is the primary GPU optimization for evolutionary algorithms.

#### 2. torch.compile for Fitness Functions

```python
@torch.compile(mode="reduce-overhead", fullgraph=True)
def compiled_fitness_evaluation(population_tensor):
    # Ensure no graph breaks inside:
    # - No print/logging
    # - No data-dependent control flow
    # - No Python I/O
    return compute_fitness_batch(population_tensor)
```

**Mode Selection:**
| Mode | Use When | Trade-off |
|------|----------|-----------|
| `"default"` | First optimization pass | Fast compile, moderate speedup |
| `"reduce-overhead"` | **Static batch sizes** | Uses CUDA graphs, eliminates launch overhead |
| `"max-autotune"` | Final tuning pass | Slow compile, maximum speedup |
| `"max-autotune-no-cudagraphs"` | Dynamic shapes | Autotuning without CUDA graph constraints |

**Confidence:** HIGH - Official PyTorch recommendation.

**Important:** `reduce-overhead` mode requires static shapes. For MAP-Elites with fixed population size, this is ideal.

#### 3. CUDA Graphs for Repetitive Workloads

For fitness evaluation that runs identically each generation:
```python
# Warmup
static_input = torch.zeros(population_size, genome_length, device="cuda")
for _ in range(3):
    _ = fitness_function(static_input)

# Capture graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = fitness_function(static_input)

# Replay (much faster than individual launches)
for gen in range(500):
    static_input.copy_(population_tensor)  # Fill with actual data
    g.replay()
    fitness = static_output.clone()
```
**Confidence:** MEDIUM - Requires static shapes and careful memory management. torch.compile with reduce-overhead does this automatically.

#### 4. Static Shape Optimization

```python
# Pad variable-size populations to fixed size
MAX_POPULATION = 100

def pad_population(pop):
    current_size = pop.shape[0]
    if current_size < MAX_POPULATION:
        padding = torch.zeros(MAX_POPULATION - current_size, pop.shape[1], device=pop.device)
        return torch.cat([pop, padding], dim=0)
    return pop
```
**Confidence:** HIGH - Avoids recompilation in torch.compile.

#### 5. DataLoader Optimization (if applicable)

If fitness evaluation involves data loading (e.g., STARK benchmark queries):
```python
dataloader = DataLoader(
    dataset,
    batch_size=query_batch_size,
    num_workers=4,  # Start with 4 * num_GPUs
    pin_memory=True,  # Faster CPU->GPU transfer
    persistent_workers=True,  # Avoid worker respawn overhead
    prefetch_factor=2  # Pre-load next batches
)
```
**Confidence:** HIGH - Standard PyTorch optimization.

### Efficient Population Management

#### 1. Tensor-Based Population Storage

```python
# Store entire population as single tensor
population = torch.zeros(pop_size, genome_length, device="cuda")
fitness = torch.zeros(pop_size, device="cuda")
features = torch.zeros(pop_size, num_features, device="cuda")

# Vectorized operations on entire population
mutated = population + torch.randn_like(population) * mutation_rate
```
**Confidence:** HIGH - EvoTorch design pattern.

#### 2. In-Place Operations Where Safe

```python
# Allocate once
mutation_noise = torch.empty(pop_size, genome_length, device="cuda")

for gen in range(500):
    # Reuse allocation
    mutation_noise.normal_(0, mutation_rate)
    offspring = population + mutation_noise
```
**Confidence:** HIGH - Reduces allocation overhead.

#### 3. Avoid Python-Level Iteration Over Population

```python
# BAD: Python loop
fitnesses = []
for genome in population:
    fitnesses.append(evaluate_single(genome))

# GOOD: Vectorized
fitnesses = evaluate_batch(population)  # Returns tensor
```
**Confidence:** HIGH - This is the core principle.

---

## What NOT to Do

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **Serial genome evaluation** | Wastes GPU parallelism; CPU launch overhead dominates | Batch evaluate entire population with vectorized fitness function |
| **Keeping gradient history** | Memory grows linearly with generations | Use `detach().clone()`, `torch.no_grad()`, `float()` for scalars |
| **Dynamic population sizes** | Forces torch.compile recompilation | Pad to fixed size or use bucketing |
| **Python lists of tensors** | Can't vectorize, poor memory locality | Single 2D tensor `[pop_size, genome_length]` |
| **`torch.cuda.empty_cache()` every iteration** | Has overhead (~1ms) | Call every 10-50 generations or based on profiling |
| **Ignoring graph breaks** | Silently degrades torch.compile performance | Use `torch._dynamo.explain()` and `fullgraph=True` to identify |
| **`num_workers` too high in DataLoader** | Memory explosion from worker processes | Start with 4 * num_GPUs, tune down if memory issues |
| **Mixing CPU and GPU tensors** | Implicit transfers kill performance | Keep entire evolution loop on GPU |
| **torch.compile on dynamic control flow** | Graph breaks, no speedup | Refactor to use `torch.where()`, `torch.cond()` |
| **Using multiprocessing for tensor sharing** | Expensive pickle serialization | Use Ray with zero-copy object store |

---

## Performance Targets vs Techniques

| Target | Primary Technique | Expected Impact |
|--------|-------------------|-----------------|
| 500 gens / 3 hours | Vectorized batch fitness evaluation | 10-50x speedup vs serial |
| Memory stability | detach().clone() + no_grad() | Prevents accumulation |
| GPU utilization > 80% | Increase batch size until memory-bound | Direct correlation |
| Reduce overhead | torch.compile(mode="reduce-overhead") | 20-40% for small kernels |

---

## Implementation Priority

1. **Immediate** (< 1 day):
   - Wrap fitness evaluation in `@torch.no_grad()`
   - Ensure `detach().clone()` on all genome copies
   - Profile with `torch.profiler` to identify current bottlenecks

2. **Short-term** (1-3 days):
   - Refactor to vectorized batch fitness evaluation
   - Add `torch.compile` to fitness function (start with `mode="default"`)
   - Implement memory monitoring for 100+ generation runs

3. **Medium-term** (3-7 days):
   - Consider EvoTorch migration for production MAP-Elites
   - Tune torch.compile mode (`reduce-overhead` vs `max-autotune`)
   - Profile and optimize DataLoader if query loading is bottleneck

---

## Sources

### Official PyTorch Documentation
- [torch.compile Documentation (PyTorch 2.9)](https://docs.pytorch.org/docs/stable/torch.compiler.html)
- [torch.compile FAQ](https://docs.pytorch.org/docs/stable/torch.compiler_faq.html)
- [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)
- [PyTorch Profiler Tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [PyTorch FAQ - Memory Management](https://docs.pytorch.org/docs/stable/notes/faq.html)
- [Dynamic Shapes in torch.compile](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html)
- [CUDA Graph Trees](https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html)
- [PyTorch 2.9 Release Blog](https://pytorch.org/blog/pytorch-2-9/)

### EvoTorch
- [EvoTorch Documentation](https://docs.evotorch.ai/)
- [EvoTorch MAP-Elites Reference](https://docs.evotorch.ai/v0.4.1/reference/evotorch/algorithms/mapelites/)
- [EvoTorch Problem Parallelization](https://docs.evotorch.ai/latest/user_guide/problem_parallelization/)
- [EvoTorch GitHub](https://github.com/nnaisense/evotorch)
- [EvoTorch PyPI](https://pypi.org/project/evotorch/)

### Community Resources
- [State of torch.compile for Training (August 2025) - ezyang's blog](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)
- [Accelerating PyTorch with CUDA Graphs - PyTorch Blog](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [vLLM torch.compile Integration (August 2025)](https://blog.vllm.ai/2025/08/20/torch-compile.html)
- [PyTorch GPU Optimization Guide - Medium](https://medium.com/@ishita.verma178/pytorch-gpu-optimization-step-by-step-guide-9dead5164ca2)
- [Understanding Tensor.detach() in PyTorch](https://medium.com/biased-algorithms/understanding-tensor-detach-in-pytorch-a-practical-guide-e859a7713f28)
- [Python Garbage Collection Tuning](https://emitechlogic.com/python-garbage-collection/)

### Research Papers
- [EvoTorch: Scalable Evolutionary Computation in Python (arXiv)](https://arxiv.org/pdf/2302.12600)
- [Enabling Population-Level Parallelism in Tree-Based Genetic Programming for GPU Acceleration (2025)](https://arxiv.org/abs/2501.17168)
- [Enhancing MAP-Elites with Multiple Parallel Evolution Strategies](https://arxiv.org/abs/2303.06137)
- [Accelerated Quality-Diversity through Massive Parallelism](https://openreview.net/pdf?id=znNITCJyTI)

---

*Generated: 2026-01-29 | Confidence: HIGH*
