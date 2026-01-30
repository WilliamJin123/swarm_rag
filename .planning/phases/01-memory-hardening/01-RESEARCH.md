# Phase 1: Memory Hardening - Research

**Researched:** 2026-01-29
**Domain:** PyTorch CUDA Memory Management / GPU Memory Stability
**Confidence:** HIGH

## Summary

This research investigates PyTorch CUDA memory management patterns to ensure GPU memory remains stable across 500+ generations of evolutionary runs without OOM crashes or memory accumulation. The project already has a `MemoryProfiler` class and some scattered `torch.cuda.empty_cache()` calls, but lacks systematic tensor lifecycle guards, configurable memory thresholds, and dedicated memory logging.

The standard approach involves three complementary techniques: (1) strict tensor lifecycle management using `torch.no_grad()` wrappers, explicit `del` statements, and `.detach()` calls; (2) real-time memory monitoring with configurable warning/hard-stop thresholds; and (3) buffer pre-allocation and reuse for repetitive operations like traversal. PyTorch's caching allocator complicates monitoring since `nvidia-smi` shows reserved (cached) memory, not actual tensor allocations. The key is using `torch.cuda.memory_allocated()` for accurate tracking.

**Primary recommendation:** Wrap all genome evaluation code in `torch.no_grad()` context, implement a `MemoryGuard` context manager that tracks per-evaluation memory delta and enforces thresholds, and pre-allocate traversal buffers at the start of each generation based on the maximum pool size.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.x | GPU tensor operations, memory management | Built-in CUDA memory API |
| psutil | 5.x+ | CPU/process memory tracking | Already in codebase |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| (none needed) | - | PyTorch's built-in memory API is sufficient | - |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom logging | pytorch-memlab | External dependency, more features than needed |
| Manual monitoring | NVIDIA SMI watch | Less precise, shows reserved not allocated |

**Installation:**
```bash
# No new dependencies required - using PyTorch built-in and existing psutil
```

## Architecture Patterns

### Recommended Project Structure
```
swarm_rag/
├── utils/
│   ├── memory.py           # Existing - extend with new guards
│   └── device.py           # Existing - no changes needed
├── evolution/
│   ├── execution/
│   │   ├── evaluator.py    # Add memory guards around evaluation
│   │   └── memory_guard.py # NEW: MemoryGuard context manager
│   └── orchestrators/
│       └── map_elites.py   # Add per-generation memory logging
└── core/
    └── swarm_retriever.py  # Add buffer pre-allocation
```

### Pattern 1: Evaluation Memory Guard
**What:** Context manager that wraps genome evaluation, tracks memory delta, and enforces thresholds
**When to use:** Every genome evaluation in the evolution loop
**Example:**
```python
# Source: Phase 1 design pattern
class MemoryGuard:
    """Context manager enforcing memory thresholds during evaluation."""

    def __init__(
        self,
        warning_threshold: float = 0.70,  # 70% of total VRAM
        hard_stop_threshold: float = 0.85,  # 85% of total VRAM
        cleanup_on_exit: bool = True
    ):
        self.warning_threshold = warning_threshold
        self.hard_stop_threshold = hard_stop_threshold
        self.cleanup_on_exit = cleanup_on_exit
        self._before_allocated = 0
        self._total_vram = 0

    def __enter__(self):
        if torch.cuda.is_available():
            self._before_allocated = torch.cuda.memory_allocated()
            self._total_vram = torch.cuda.get_device_properties(0).total_memory
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not torch.cuda.is_available():
            return False

        after_allocated = torch.cuda.memory_allocated()
        delta = after_allocated - self._before_allocated
        usage_ratio = after_allocated / self._total_vram

        if usage_ratio >= self.hard_stop_threshold:
            # Trigger checkpoint and clean exit
            raise MemoryThresholdExceeded(
                f"Hard stop: {usage_ratio:.1%} >= {self.hard_stop_threshold:.1%}"
            )

        if usage_ratio >= self.warning_threshold:
            logger.warning(
                f"Memory warning: {usage_ratio:.1%} >= {self.warning_threshold:.1%}"
            )

        if self.cleanup_on_exit:
            gc.collect()
            torch.cuda.empty_cache()

        return False
```

### Pattern 2: Per-Generation Memory Stats
**What:** Log comprehensive memory stats at generation boundaries for trend detection
**When to use:** Start and end of each generation in orchestrator
**Example:**
```python
# Source: Phase 1 design pattern
@dataclass
class GenerationMemoryStats:
    generation: int
    timestamp: float
    allocated_mb: float
    cached_mb: float
    peak_mb: float
    delta_mb: float  # Change from previous generation

    def to_log_line(self) -> str:
        return (
            f"gen={self.generation} "
            f"alloc={self.allocated_mb:.1f}MB "
            f"cached={self.cached_mb:.1f}MB "
            f"peak={self.peak_mb:.1f}MB "
            f"delta={self.delta_mb:+.1f}MB"
        )

class MemoryLogger:
    def __init__(self, log_path: str):
        self._log_path = log_path
        self._prev_allocated = 0
        self._handler = None

    def log_generation(self, generation: int):
        if not torch.cuda.is_available():
            return

        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        cached = torch.cuda.memory_reserved() / (1024 * 1024)
        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        delta = allocated - self._prev_allocated

        stats = GenerationMemoryStats(
            generation=generation,
            timestamp=time.time(),
            allocated_mb=allocated,
            cached_mb=cached,
            peak_mb=peak,
            delta_mb=delta
        )

        self._write_stats(stats)
        self._prev_allocated = allocated

        # Reset peak for next generation
        torch.cuda.reset_peak_memory_stats()
```

### Pattern 3: Buffer Pre-allocation
**What:** Create fixed-size buffers at generation start and reuse them
**When to use:** Traversal operations that need scratch space for scoring, ranking
**Example:**
```python
# Source: Phase 1 design pattern
class TraversalBufferPool:
    """Pre-allocated buffers for graph traversal operations."""

    def __init__(self, max_pool_size: int, device: str = "cuda"):
        self.device = device
        self.max_pool_size = max_pool_size

        # Pre-allocate scoring buffer (reused for each step)
        self._score_buffer = torch.zeros(
            max_pool_size, dtype=torch.float32, device=device
        )
        # Pre-allocate index buffer
        self._index_buffer = torch.zeros(
            max_pool_size, dtype=torch.long, device=device
        )

    def get_score_buffer(self, size: int) -> torch.Tensor:
        """Return a view into the pre-allocated buffer."""
        return self._score_buffer[:size]

    def get_index_buffer(self, size: int) -> torch.Tensor:
        return self._index_buffer[:size]

    def clear(self):
        """Zero out buffers (not deallocate) for next use."""
        self._score_buffer.zero_()
        self._index_buffer.zero_()
```

### Anti-Patterns to Avoid
- **Creating tensors in loops:** Allocating new tensors inside hot paths (per-step, per-query) causes fragmentation and OOM. Pre-allocate outside loop and reuse views.
- **Accumulating history:** Storing tensors with gradient history across iterations. Always use `.detach()` or `float()` when extracting values.
- **Delayed cleanup:** Waiting until OOM to clean up. Proactive cleanup after each evaluation prevents accumulation.
- **Ignoring cache vs allocated:** Using `nvidia-smi` or `memory_reserved()` for threshold checks. Use `memory_allocated()` for accurate tensor usage.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Memory tracking | Custom nvidia-smi parsing | `torch.cuda.memory_allocated()` | Built-in, accurate, fast |
| Peak detection | Manual max tracking | `torch.cuda.max_memory_allocated()` | Already implemented in PyTorch |
| Memory snapshots | Custom allocation tracking | `torch.cuda.memory._record_memory_history()` | Official tool for debugging |
| Process memory | Custom /proc parsing | `psutil.Process().memory_info()` | Cross-platform, reliable |

**Key insight:** PyTorch's CUDA memory API is comprehensive and well-tested. Custom memory tracking adds complexity without benefit. Focus on using the built-in functions correctly and wrapping them in domain-specific guards.

## Common Pitfalls

### Pitfall 1: Confusion Between Reserved and Allocated Memory
**What goes wrong:** Developers use `nvidia-smi` or `memory_reserved()` for threshold checks, see high usage, panic, and add excessive `empty_cache()` calls that hurt performance.
**Why it happens:** PyTorch's caching allocator reserves more memory than actually used to speed up future allocations.
**How to avoid:** Always use `torch.cuda.memory_allocated()` for threshold checks. The cached memory is fine - it will be reused.
**Warning signs:** Memory appears high in nvidia-smi but code runs fine; excessive `empty_cache()` calls slowing down code.

### Pitfall 2: Gradient History Accumulation
**What goes wrong:** Metrics/losses accumulated as `total_loss += loss` retain full computational graph, causing memory to grow linearly with iterations.
**Why it happens:** Tensors with `requires_grad=True` keep history for backward pass. Adding them accumulates history.
**How to avoid:** Use `float(loss)` or `loss.item()` when accumulating scalars. Use `.detach()` when storing tensor results.
**Warning signs:** Memory grows steadily over iterations; OOM after many iterations but not immediately.

### Pitfall 3: Forgotten References in Collections
**What goes wrong:** Appending tensors to lists for batch processing but never clearing lists, causing memory to accumulate.
**Why it happens:** Python garbage collector can't free tensors still referenced by lists/dicts.
**How to avoid:** Clear collections after use with `.clear()` or `del`. Process in smaller batches.
**Warning signs:** Lists grow unboundedly; `gc.collect()` doesn't help; memory freed only after function returns.

### Pitfall 4: Synchronous vs Asynchronous CUDA Operations
**What goes wrong:** Memory appears to not be freed because CUDA operations are asynchronous.
**Why it happens:** `empty_cache()` may not immediately free memory if GPU is still executing operations.
**How to avoid:** Call `torch.cuda.synchronize()` before `empty_cache()` if precise timing is needed.
**Warning signs:** Memory frees inconsistently; debugging shows different behavior than production.

### Pitfall 5: Fragmentation from Variable-Size Allocations
**What goes wrong:** Memory available per `memory_allocated()` but OOM occurs because free memory is fragmented.
**Why it happens:** Alternating large and small allocations create gaps that can't fit new large allocations.
**How to avoid:** Use consistent tensor sizes where possible. Pre-allocate maximum size and use views. Use `max_split_size_mb` env var if needed.
**Warning signs:** OOM with significant free memory; issue worsens over time; fresh process runs fine.

## Code Examples

Verified patterns from official sources:

### Complete no_grad Wrapper for Evaluation
```python
# Source: PyTorch official docs - no_grad
@torch.no_grad()
def evaluate_genome(genome: Genome, retriever: SwarmRetriever, ...) -> FitnessResult:
    """
    Evaluate a genome with gradient tracking disabled.

    The @torch.no_grad() decorator ensures:
    - No gradient computation overhead
    - No gradient storage overhead
    - All intermediate tensors can be freed immediately
    """
    # All operations inside are gradient-free
    results = retriever.retrieve_batch(...)
    metrics = compute_metrics(results, ground_truth)

    # Safe to return - no gradient history attached
    return FitnessResult(quality=metrics['Hit@5'], ...)
```

### Memory Threshold Check Pattern
```python
# Source: PyTorch CUDA Memory Usage docs
def check_memory_thresholds(
    warning_pct: float = 0.70,
    hard_stop_pct: float = 0.85
) -> tuple[bool, bool, float]:
    """
    Check current GPU memory against thresholds.

    Returns:
        (is_warning, is_critical, usage_pct)
    """
    if not torch.cuda.is_available():
        return False, False, 0.0

    # Use allocated (actual tensor usage), not reserved (cached)
    allocated = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory

    usage_pct = allocated / total

    return (
        usage_pct >= warning_pct,
        usage_pct >= hard_stop_pct,
        usage_pct
    )
```

### Clean Tensor Extraction Pattern
```python
# Source: PyTorch FAQ - Don't accumulate history
def extract_metrics_safely(output_tensor: torch.Tensor) -> dict:
    """
    Extract metric values without retaining gradient history.

    Critical: Always detach or convert to Python scalar
    before storing results outside the evaluation scope.
    """
    return {
        # For scalar values - use .item()
        'loss': output_tensor.item(),

        # For tensor values that need to persist - use .detach().cpu()
        'embeddings': output_tensor.detach().cpu(),

        # For computing further - use float() or .detach()
        'score': float(output_tensor),
    }
```

### Complete Cleanup Sequence
```python
# Source: Community best practice, verified against PyTorch docs
def cleanup_after_evaluation():
    """
    Complete cleanup sequence after genome evaluation.

    Order matters:
    1. Delete local references
    2. Run Python garbage collection
    3. Clear CUDA cache
    4. (Optional) Synchronize for debugging
    """
    # 1. Delete known large tensors
    del results, embeddings, scores  # Replace with actual vars

    # 2. Collect Python garbage (including tensor refs)
    import gc
    gc.collect()

    # 3. Release CUDA cached memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. For debugging memory timing issues only
    # torch.cuda.synchronize()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual memory tracking | `torch.cuda.memory._record_memory_history()` | PyTorch 2.0+ | Official memory snapshot visualization |
| `nvidia-smi` monitoring | `torch.cuda.memory_stats()` | PyTorch 1.x | Accurate per-process tracking |
| `no_grad()` context | `inference_mode()` context | PyTorch 1.9+ | Stricter, faster inference mode |
| Manual fragmentation fix | `PYTORCH_CUDA_ALLOC_CONF` | PyTorch 1.8+ | Built-in allocator tuning |

**Deprecated/outdated:**
- Using `volatile=True` for inference (deprecated, use `no_grad()` or `inference_mode()`)
- Parsing nvidia-smi output for memory tracking (use PyTorch API)

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal empty_cache frequency**
   - What we know: Too frequent hurts performance (allocation overhead), too rare risks fragmentation
   - What's unclear: Exact optimal frequency for this workload (per-genome vs per-generation)
   - Recommendation: Start with per-genome cleanup, measure impact, adjust based on profiling

2. **inference_mode vs no_grad**
   - What we know: `inference_mode()` is stricter and faster than `no_grad()`
   - What's unclear: Whether it's compatible with all operations in swarm traversal
   - Recommendation: Start with `no_grad()` (proven safe), evaluate `inference_mode()` in Phase 6

3. **Buffer pre-allocation sizing**
   - What we know: Pre-allocating max size prevents fragmentation
   - What's unclear: Actual max pool sizes used across genomes (varies by config)
   - Recommendation: Use `get_unique_pool_sizes()` from existing code to determine sizes at generation start

## Sources

### Primary (HIGH confidence)
- [PyTorch CUDA Memory Usage Documentation](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) - Memory tracking API reference
- [PyTorch no_grad Documentation](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html) - Gradient disabling
- [PyTorch FAQ - Avoid History Accumulation](https://docs.pytorch.org/docs/stable/notes/faq.html) - Memory leak prevention
- [PyTorch CUDA Semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) - Caching allocator behavior

### Secondary (MEDIUM confidence)
- [PyTorch GPU Memory Visualization Blog](https://pytorch.org/blog/understanding-gpu-memory-1/) - Memory snapshot tooling
- [Community Memory Management Guide](https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/) - Practical patterns
- [GeeksforGeeks PyTorch Memory Optimization](https://www.geeksforgeeks.org/deep-learning/how-to-optimize-memory-usage-in-pytorch/) - Best practices compilation

### Tertiary (LOW confidence)
- WebSearch community discussions on fragmentation handling (needs validation with profiling)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Using only PyTorch built-in APIs, well documented
- Architecture: HIGH - Patterns derived from official PyTorch documentation
- Pitfalls: HIGH - Documented in official FAQ and confirmed by multiple sources

**Research date:** 2026-01-29
**Valid until:** 2026-03-29 (PyTorch memory API is stable, 60 days validity)
