# Phase 6: Performance Validation - Research

**Researched:** 2026-01-31
**Domain:** Performance benchmarking, PyTorch profiling, evolution validation
**Confidence:** HIGH

## Summary

This phase validates that the full optimization stack (memory guards, fitness cache, embedding cache, convergence detection, async checkpointing) achieves the target of 500 generations in 3 hours with population 50-100. The research focused on:

1. **Existing infrastructure** - The codebase already has comprehensive profiling (`GenerationProfiler`), memory logging (`MemoryLogger`), and timing utilities (`Benchmarker`). The primary task is creating a benchmark harness that orchestrates these components.

2. **PyTorch Profiler API** - Official documentation confirms `torch.profiler.profile` with `ProfilerActivity.CUDA` for GPU-level insights, `profile_memory=True` for memory tracking, and `export_chrome_trace()` for detailed analysis.

3. **Benchmark design** - Single run with population 75, cold-start timing, convergence early-stop allowed with time extrapolation to 500 generations. Pass criteria: projected time < 3 hours and peak VRAM <= 4GB.

**Primary recommendation:** Create a `PerformanceBenchmark` class that wraps the existing `EvolutionEngine`, collects per-generation timing, and produces a JSON report with pass/fail determination.

## Standard Stack

The established tools for this domain (all already in the codebase):

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch.profiler | PyTorch 2.x | GPU-level profiling | Official PyTorch profiler, captures CUDA kernels |
| torch.cuda | PyTorch 2.x | Memory/timing APIs | `memory_allocated()`, `synchronize()`, `max_memory_allocated()` |
| time.perf_counter | stdlib | High-res timing | Sub-microsecond precision, standard Python |
| json | stdlib | Report serialization | Standard format for results |

### Supporting (Already in Codebase)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| GenerationProfiler | internal | Per-generation timing | Enable with `EVOLUTION_PROFILE=1` |
| MemoryLogger | internal | Per-generation memory stats | Already integrated in orchestrator |
| FitnessCache | internal | Cache hit tracking | Already integrated in evaluator |
| EmbeddingCache | internal | Embedding cache stats | Already integrated in evaluator |
| ConvergenceDetector | internal | Early termination | Already integrated in orchestrator |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| torch.profiler | NVIDIA Nsight | Nsight is more detailed but requires separate tooling |
| Custom timing | pytest-benchmark | pytest-benchmark is for unit tests, not full evolution runs |
| JSON output | CSV | JSON captures nested structure better |

**Installation:**
```bash
# No new dependencies needed - all tools already available
```

## Architecture Patterns

### Recommended Benchmark Structure
```
benchmark/
    __init__.py
    performance_benchmark.py  # Main benchmark class
    report_schema.py          # Pydantic models for JSON schema
    run_benchmark.py          # CLI entry point
```

### Pattern 1: Benchmark Harness Wrapping EvolutionEngine
**What:** Create a `PerformanceBenchmark` class that configures and runs `EvolutionEngine` with timing instrumentation
**When to use:** For the performance validation benchmark
**Example:**
```python
# Source: Project architecture pattern
class PerformanceBenchmark:
    """
    Validates full optimization stack performance.

    Wraps EvolutionEngine to:
    1. Configure with benchmark parameters (pop=75, gens=500)
    2. Collect per-generation timing
    3. Track cache hit rates
    4. Monitor peak GPU memory
    5. Produce JSON report with pass/fail
    """

    def __init__(
        self,
        population_size: int = 75,
        max_generations: int = 500,
        time_limit_hours: float = 3.0,
        memory_limit_gb: float = 4.0,
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.time_limit_seconds = time_limit_hours * 3600
        self.memory_limit_bytes = memory_limit_gb * 1024**3

        self.generation_times: List[float] = []
        self.peak_memory_bytes: int = 0

    def run(self) -> BenchmarkResult:
        """Run benchmark and return results."""
        # Configure engine with benchmark settings
        # Run evolution with timing hooks
        # Collect metrics
        # Produce report
        pass
```

### Pattern 2: Extrapolation for Early Convergence
**What:** When convergence triggers early, extrapolate total time from average generation time
**When to use:** Benchmark allows early-stop, need to project to 500 generations
**Example:**
```python
# Source: CONTEXT.md requirement
def extrapolate_time(
    actual_generations: int,
    actual_time_seconds: float,
    target_generations: int = 500
) -> Tuple[float, bool]:
    """
    Extrapolate total time if convergence triggered early.

    Returns:
        Tuple of (projected_time_seconds, was_extrapolated)
    """
    if actual_generations >= target_generations:
        return actual_time_seconds, False

    avg_gen_time = actual_time_seconds / actual_generations
    projected_time = avg_gen_time * target_generations

    return projected_time, True
```

### Pattern 3: System Info Collection
**What:** Capture GPU, CUDA, Python, PyTorch versions for reproducibility
**When to use:** At benchmark start, include in JSON report
**Example:**
```python
# Source: PyTorch documentation
import sys
import torch

def collect_system_info() -> dict:
    """Collect system information for benchmark report."""
    info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_total_memory_gb"] = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )
        info["cudnn_version"] = torch.backends.cudnn.version()

    return info
```

### Anti-Patterns to Avoid
- **Running multiple trials for validation:** Decision is single run - no statistical replication needed
- **Warm-up before timing:** Decision includes cold-start in timing
- **Auto-tuning on failure:** Decision is profile/report only, no auto-tune
- **Ignoring convergence early-stop:** Must extrapolate to 500 gens if stopped early

## Don't Hand-Roll

Problems that have existing solutions in the codebase:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-generation timing | Custom timing loop | `GenerationProfiler` | Already tracks timing + memory per section |
| Memory tracking | Manual `memory_allocated()` calls | `MemoryLogger` | Already integrated, exports JSON |
| Cache hit rates | Custom counters | `FitnessCache.total_stats`, `EmbeddingCacheStats` | Already track hits/misses |
| Early termination | Custom convergence check | `ConvergenceDetector` | Already configured with window=40, threshold=0.1% |
| GPU detection | Manual CUDA checks | `get_device()`, `get_gpu_memory_info()` | Already handles CUDA/MPS/CPU |

**Key insight:** The codebase has excellent instrumentation. The benchmark task is primarily orchestration and reporting, not building new profiling infrastructure.

## Common Pitfalls

### Pitfall 1: Not Using torch.cuda.synchronize() for GPU Timing
**What goes wrong:** GPU timing appears faster than reality because operations are asynchronous
**Why it happens:** CUDA launches kernels asynchronously; CPU-side timing doesn't wait for completion
**How to avoid:** Call `torch.cuda.synchronize()` before timing measurements
**Warning signs:** Suspiciously fast timing, inconsistent results

```python
# Correct pattern (used in existing Benchmarker class)
torch.cuda.synchronize()
start = time.perf_counter()
# ... GPU operations ...
torch.cuda.synchronize()
end = time.perf_counter()
```

### Pitfall 2: Using memory_reserved() Instead of memory_allocated()
**What goes wrong:** Over-reporting memory usage due to CUDA caching
**Why it happens:** `memory_reserved()` includes unused cached memory
**How to avoid:** Use `memory_allocated()` for actual usage (per prior decision [01-01])
**Warning signs:** Memory appears to grow indefinitely when it's just cache

### Pitfall 3: Forgetting to Reset Peak Memory Stats
**What goes wrong:** Peak memory reflects lifetime max, not per-generation max
**Why it happens:** `max_memory_allocated()` accumulates across generations
**How to avoid:** Call `torch.cuda.reset_peak_memory_stats()` at generation start
**Warning signs:** Peak memory only ever increases

### Pitfall 4: Running Benchmark in Debug Mode
**What goes wrong:** Performance is artificially slow
**Why it happens:** Debug mode disables optimizations, adds overhead
**How to avoid:** Run with `PYTHONOPTIMIZE=1` or ensure no debug flags
**Warning signs:** Unexpectedly slow generation times

### Pitfall 5: Including Data Loading in Timing
**What goes wrong:** Data loading variance skews benchmark results
**Why it happens:** Disk I/O is highly variable
**How to avoid:** Load STARK Prime data before starting benchmark timing
**Warning signs:** First generation is much slower than subsequent ones

## Code Examples

Verified patterns from the codebase and official documentation:

### Per-Generation Timing with Profiler
```python
# Source: swarm_rag_module/swarm_rag/evolution/execution/profiler.py
profiler = GenerationProfiler(enabled=True, max_generations=500)

for gen in range(n_generations):
    profiler.start_generation(gen)

    with profiler.section("breeding"):
        offspring = breed()

    with profiler.section("evaluation"):
        evaluate(offspring)

    with profiler.section("archive_insert"):
        archive.add_all(offspring)

    # Get live stats
    logger.info(profiler.end_generation())

# Final summary
print(profiler.summary())
profiler.save("profiler_data.json")
```

### PyTorch Profiler for GPU Insights
```python
# Source: PyTorch official docs (https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
from torch.profiler import profile, ProfilerActivity, schedule

def trace_handler(prof):
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace(f"trace_{prof.step_num}.json")

my_schedule = schedule(skip_first=10, wait=5, warmup=1, active=3, repeat=2)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=my_schedule,
    on_trace_ready=trace_handler,
    profile_memory=True,
    record_shapes=True
) as prof:
    for gen in range(50):
        run_generation()
        prof.step()
```

### Collecting Cache Statistics
```python
# Source: swarm_rag_module/swarm_rag/evolution/execution/evaluator.py
# FitnessCache stats
cache_stats = evaluator._fitness_cache.total_stats
fitness_hit_rate = cache_stats.hit_rate

# EmbeddingCache stats
embed_cache = EmbeddingCacheProvider.get()
if embed_cache:
    embed_stats = embed_cache.stats
    embed_hit_rate = embed_stats.hit_rate
    time_saved = embed_stats.compute_time_saved_sec
```

### Memory Tracking Pattern
```python
# Source: swarm_rag_module/swarm_rag/evolution/execution/memory_logger.py
memory_logger = MemoryLogger(log_dir="./benchmark_logs", warning_threshold=0.70)

for gen in range(n_generations):
    stats = memory_logger.log_generation(
        generation=gen,
        cache_stats=fitness_cache.finalize_generation(gen),
        embed_cache_stats=embed_cache.finalize_generation(gen)
    )

    if stats.peak_mb > 4096:  # 4GB limit per CONTEXT.md
        raise MemoryError(f"Peak VRAM {stats.peak_mb:.1f}MB exceeds 4GB limit")

# Export final stats
memory_logger.export_stats("memory_stats.json")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual timing loops | `GenerationProfiler` with sections | Already in codebase | Structured timing with memory tracking |
| Separate cache counters | Integrated `FitnessCache`/`EmbeddingCache` stats | Already in codebase | Automatic hit rate tracking |
| Manual memory polling | `MemoryLogger` per-generation | Already in codebase | Automatic delta and trend tracking |
| Fixed evolution length | `ConvergenceDetector` early-stop | Already in codebase | Automatic stagnation detection |

**Deprecated/outdated:**
- N/A - The codebase uses current best practices

## Open Questions

Things that couldn't be fully resolved:

1. **Exact STARK Prime Data Loading**
   - What we know: Need full STARK Prime dataset for realistic performance
   - What's unclear: Exact loading pattern used in existing tests
   - Recommendation: Check existing integration tests for data loading pattern

2. **PyTorch Profiler Overhead**
   - What we know: CUDA profiling adds overhead per documentation
   - What's unclear: Exact overhead magnitude on this workload
   - Recommendation: Run with profiler off for final timing, with profiler on only for debugging bottlenecks

## Sources

### Primary (HIGH confidence)
- `swarm_rag_module/swarm_rag/evolution/execution/profiler.py` - GenerationProfiler implementation
- `swarm_rag_module/swarm_rag/evolution/execution/memory_logger.py` - MemoryLogger implementation
- `swarm_rag_module/swarm_rag/evolution/execution/evaluator.py` - Cache integration patterns
- `swarm_rag_module/swarm_rag/evolution/orchestrators/map_elites.py` - Evolution loop structure
- `swarm_rag_module/swarm_rag/utils/benchmark.py` - Existing Benchmarker class
- [PyTorch Profiler Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) - Official profiler documentation

### Secondary (MEDIUM confidence)
- [PyTorch Profiler with TensorBoard](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) - Visualization options

### Tertiary (LOW confidence)
- N/A

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All tools already in codebase and verified
- Architecture: HIGH - Clear patterns from existing code
- Pitfalls: HIGH - Documented in PyTorch official docs and visible in codebase patterns

**Research date:** 2026-01-31
**Valid until:** 60 days (stable domain, no rapidly changing dependencies)

---

## Recommended JSON Schema for Benchmark Results

Based on CONTEXT.md (Claude's Discretion: "Exact JSON schema for benchmark results"):

```json
{
  "$schema": "benchmark-results-v1",
  "timestamp": "2026-01-31T12:00:00Z",
  "pass": true,

  "config": {
    "population_size": 75,
    "target_generations": 500,
    "time_limit_hours": 3.0,
    "memory_limit_gb": 4.0
  },

  "results": {
    "actual_generations": 342,
    "termination_reason": "convergence",
    "was_extrapolated": true,

    "timing": {
      "total_seconds": 6123.5,
      "projected_seconds": 8956.2,
      "avg_generation_ms": 17912.4,
      "min_generation_ms": 15234.1,
      "max_generation_ms": 23456.7,
      "p99_generation_ms": 22345.6
    },

    "memory": {
      "peak_allocated_mb": 3245.6,
      "peak_reserved_mb": 4012.3,
      "final_allocated_mb": 2890.1
    },

    "cache_stats": {
      "fitness_cache_hit_rate": 0.342,
      "fitness_cache_total_lookups": 25650,
      "embedding_cache_hit_rate": 0.987,
      "embedding_cache_time_saved_sec": 1234.5
    }
  },

  "system_info": {
    "python_version": "3.11.4",
    "pytorch_version": "2.1.0+cu118",
    "cuda_version": "11.8",
    "cudnn_version": 8700,
    "gpu_name": "NVIDIA GeForce RTX 3080",
    "gpu_total_memory_gb": 10.0
  },

  "pass_criteria": {
    "time_pass": true,
    "time_reason": "Projected 8956s < 10800s (3h limit)",
    "memory_pass": true,
    "memory_reason": "Peak 3245MB < 4096MB (4GB limit)"
  }
}
```

## Recommended Console Summary Format

Based on CONTEXT.md (Claude's Discretion: "Console summary formatting"):

```
================================================================================
PERFORMANCE VALIDATION BENCHMARK - RESULTS
================================================================================

Configuration:
  Population size:     75
  Target generations:  500
  Time limit:          3.0 hours
  Memory limit:        4.0 GB

Results:
  Actual generations:  342 (converged)
  Total time:          1h 42m 3.5s
  Projected time:      2h 29m 16.2s (extrapolated to 500 gens)

  Generation timing:
    Average:  17912 ms
    Min:      15234 ms
    Max:      23456 ms
    P99:      22345 ms

  Memory:
    Peak allocated:  3245.6 MB
    Peak reserved:   4012.3 MB

  Cache performance:
    Fitness cache:   34.2% hit rate (25650 lookups)
    Embedding cache: 98.7% hit rate (saved 1234.5s)

Pass Criteria:
  [PASS] Time:   2h 29m < 3h 00m
  [PASS] Memory: 3245 MB < 4096 MB

================================================================================
OVERALL: PASS
================================================================================

Report saved to: .planning/phases/06-performance-validation/benchmark-results.json
```

## Extrapolation Strategy

Based on CONTEXT.md (Claude's Discretion: "How to extrapolate timing if convergence triggers early"):

When convergence triggers early (before 500 generations):

1. **Calculate average generation time** from actual generations completed
2. **Project total time** as `avg_gen_time * 500`
3. **Flag as extrapolated** in results
4. **Document assumption**: Linear extrapolation assumes generation time is roughly constant

```python
def extrapolate_to_target(
    actual_gens: int,
    actual_time_sec: float,
    target_gens: int = 500
) -> Tuple[float, bool]:
    """
    Extrapolate benchmark time to target generations.

    Uses simple linear extrapolation based on average generation time.
    This is valid because:
    - Memory usage is bounded (cache eviction, memory guards)
    - Generation work is roughly constant (fixed population size)
    - No progressive slowdown expected (addressed in earlier phases)

    Args:
        actual_gens: Generations actually completed
        actual_time_sec: Wall clock time for actual generations
        target_gens: Target generation count (500)

    Returns:
        Tuple of (projected_time_seconds, was_extrapolated)
    """
    if actual_gens >= target_gens:
        return actual_time_sec, False

    avg_gen_time = actual_time_sec / actual_gens
    projected = avg_gen_time * target_gens

    return projected, True
```

**Important**: If actual generations < 50, extrapolation confidence is LOW. Consider warning in report.
