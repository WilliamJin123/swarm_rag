---
phase: 01-memory-hardening
plan: 01
subsystem: memory-management
tags: [pytorch, cuda, memory-guard, no-grad, threshold-enforcement]

dependency-graph:
  requires: []
  provides:
    - MemoryGuard context manager
    - MemoryThresholdExceeded exception
    - evaluation_no_grad decorator
    - check_memory_thresholds function
    - MEMORY_WARNING_THRESHOLD constant
    - MEMORY_HARD_STOP_THRESHOLD constant
  affects:
    - 01-02 (memory logger uses thresholds)
    - 01-03 (buffer pools use memory guard)
    - all evaluation code paths

tech-stack:
  added: []
  patterns:
    - Context manager for resource tracking
    - Decorator pattern for evaluation wrapping
    - Environment variable configuration

key-files:
  created:
    - swarm_rag_module/swarm_rag/evolution/execution/memory_guard.py
  modified:
    - swarm_rag_module/swarm_rag/utils/memory.py

decisions:
  - id: mem-01
    choice: "Use memory_allocated() not memory_reserved() for thresholds"
    rationale: "memory_allocated() reflects actual tensor usage; memory_reserved() includes caching allocator overhead"
  - id: mem-02
    choice: "Default thresholds 70% warning, 85% hard stop"
    rationale: "Leave headroom for spike allocations; hard stop before OOM crash"
  - id: mem-03
    choice: "Environment variable override for thresholds"
    rationale: "Allow runtime tuning without code changes for different GPU configurations"

metrics:
  duration: "~15 minutes"
  completed: "2026-01-30"
---

# Phase 01 Plan 01: MemoryGuard Context Manager Summary

**One-liner:** GPU memory threshold enforcement via MemoryGuard context manager with configurable warning/hard-stop thresholds and automatic CUDA cache cleanup.

## What Was Built

### 1. MemoryGuard Context Manager
`swarm_rag_module/swarm_rag/evolution/execution/memory_guard.py`

A context manager that wraps code sections to:
- Track GPU memory usage (before/after allocation)
- Enforce configurable warning threshold (default 70% VRAM)
- Enforce hard stop threshold (default 85% VRAM) with exception
- Automatically clean up CUDA cache on exit (gc.collect + empty_cache)
- Provide memory delta properties (delta_bytes, delta_mb, usage_ratio)

```python
with MemoryGuard(warning_threshold=0.70, hard_stop_threshold=0.85, label="eval") as guard:
    result = evaluate_genome(genome)
print(f"Memory delta: {guard.delta_mb:.2f} MB")
```

### 2. MemoryThresholdExceeded Exception
Custom exception raised when hard stop threshold is exceeded, containing:
- usage_ratio: Current memory usage as ratio
- threshold: The threshold that was exceeded
- delta_bytes: Memory change during the guarded operation
- label: Operation identifier

### 3. evaluation_no_grad Decorator
Combines `torch.no_grad()` with optional MemoryGuard tracking:

```python
@evaluation_no_grad(track_memory=True)
def evaluate_genome(genome, retriever):
    # All operations are gradient-free
    # Memory is tracked and cleaned up
    return compute_fitness(...)
```

### 4. Threshold Configuration
Extended `swarm_rag/utils/memory.py` with:
- `MEMORY_WARNING_THRESHOLD`: Module constant (default 0.70, env override)
- `MEMORY_HARD_STOP_THRESHOLD`: Module constant (default 0.85, env override)
- `check_memory_thresholds()`: Function returning (is_warning, is_critical, usage_pct)

## Key Implementation Details

- **Uses memory_allocated() not memory_reserved()**: Per RESEARCH.md, memory_allocated() gives accurate tensor usage while memory_reserved() includes caching allocator overhead
- **Threshold checks on __exit__**: Memory is checked after the guarded code completes, not continuously
- **Cleanup before raising**: On hard stop, cleanup is performed before raising exception to free memory
- **CUDA availability checks**: All operations gracefully handle non-CUDA environments

## Commits

| Hash | Description |
|------|-------------|
| a88fc43 | feat(01-01): create MemoryGuard context manager with threshold enforcement |
| aa6ad6c | feat(01-01): add environment-based threshold configuration to memory.py |
| 69a5fcb | feat(01-01): add evaluation_no_grad decorator combining no_grad with MemoryGuard |

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

All verification criteria passed:
1. All components import without errors
2. MemoryGuard delta_bytes property returns int
3. MemoryThresholdExceeded raised when threshold=0.0
4. Environment variable override works (tested with MEMORY_WARNING_THRESHOLD=0.5)
5. Key links verified:
   - torch.cuda.memory_allocated() used in threshold checks
   - torch.cuda.empty_cache() called on cleanup

## Next Phase Readiness

Ready for:
- **01-02**: Memory logger can use MEMORY_WARNING_THRESHOLD/MEMORY_HARD_STOP_THRESHOLD constants
- **01-03**: Buffer pools can wrap allocations in MemoryGuard

No blockers identified.
