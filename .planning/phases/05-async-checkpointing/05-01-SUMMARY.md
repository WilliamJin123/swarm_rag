---
phase: 05-async-checkpointing
plan: 01
subsystem: storage
tags: [async, threading, checkpointing, atomic-writes, queue]

dependency-graph:
  requires: []
  provides: [async-checkpoint-writer, non-blocking-saves, atomic-writes]
  affects: [06-unified-run-logging, 07-adaptive-population]

tech-stack:
  added: []
  patterns:
    - unbounded-queue-for-all-checkpoints
    - non-daemon-thread-for-graceful-shutdown
    - sentinel-value-for-shutdown-signal
    - temp-file-atomic-replace
    - retry-with-cleanup

key-files:
  created: []
  modified:
    - swarm_rag_module/swarm_rag/evolution/storage/run_manager.py

decisions:
  - id: 05-01-unbounded-queue
    choice: "Queue() unbounded instead of Queue(maxsize=1)"
    reason: "Queue all checkpoints, never drop per CONTEXT.md"
  - id: 05-01-non-daemon-thread
    choice: "daemon=False for background thread"
    reason: "Allow graceful shutdown - wait for queue drain"
  - id: 05-01-atomic-writes
    choice: "tempfile.mkstemp + os.replace pattern"
    reason: "Prevent corrupt checkpoint files on failure"
  - id: 05-01-no-checkpoint-rotation
    choice: "Disable _cleanup_old_checkpoints"
    reason: "Keep all checkpoints per CONTEXT.md"
  - id: 05-01-deep-copy-order
    choice: "Check dict/list/tuple before generic .copy()"
    reason: "Dict has .copy() (shallow) - must handle recursively first"

metrics:
  duration: 10min
  completed: 2026-01-31
---

# Phase 05 Plan 01: Async Checkpointing Refactor Summary

**One-liner:** Refactored AsyncCheckpointWriter with unbounded queue, non-daemon thread, atomic writes via mkstemp+replace, and comprehensive logging.

## What Changed

### AsyncCheckpointWriter Refactored
- Changed from `Queue(maxsize=1)` to unbounded `Queue()` - all checkpoints queued
- Changed from `daemon=True` to `daemon=False` - graceful shutdown waits for drain
- Added `_prepare_checkpoint_data()` with recursive deep copy handling tensors
- Implemented atomic writes: `tempfile.mkstemp()` + `os.replace()`
- Added retry logic (1 retry with 0.5s delay on failure)
- Added `CheckpointStats` dataclass for performance tracking
- Removed `_cleanup_old_checkpoints` (per CONTEXT: keep all)
- Removed `flush()` method (shutdown handles drain)

### Logging Added
- "Checkpoint queued (gen N)" on queue
- "Checkpoint queue: N pending" warning if queue depth > 1
- "Checkpoint saved (gen N, X.XMB, X.XXs)" on completion
- "Checkpointing: N saves, Xs total, avg Ys" on shutdown

### RunManager Integration
- `save_checkpoint()` now deep-copies state before queuing
- `close()` calls `shutdown(timeout=30.0)` and logs status
- `_cleanup_old_checkpoints()` made a no-op for API compatibility

## Tasks Completed

| Task | Name | Commit | Key Changes |
|------|------|--------|-------------|
| 1 | Refactor AsyncCheckpointWriter class | 638bbec | Unbounded queue, non-daemon thread, atomic writes, CheckpointStats |
| 2 | Fix deep copy and integration | 524adba | Fixed _deep_copy_state ordering (dict/list before .copy()) |

## Verification Results

**Syntax check:** PASS
**Import check:** PASS
**Pattern counts:**
- daemon=False: 1
- os.replace: 4
- mkstemp: 2
- "Checkpoint queued": 1
- "Checkpoint saved": 2

**Integration tests:**
- Non-daemon thread: PASS
- Tensor handling (detach/clone/cpu): PASS
- Unbounded queue (5 items queued): PASS
- Shutdown queue drain (3 checkpoints written): PASS

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed deep copy ordering in _deep_copy_state**
- **Found during:** Task 2 integration testing
- **Issue:** Dict objects have a `.copy()` method that returns a shallow copy. The original code checked `hasattr(obj, 'copy')` before `isinstance(obj, dict)`, causing dicts to take the shallow copy branch instead of recursive deep copy.
- **Fix:** Reordered checks: dict/list/tuple checked before generic `.copy()` method
- **Files modified:** run_manager.py
- **Commit:** 524adba

## Key Implementation Details

### Atomic Write Pattern
```python
fd, temp_path = tempfile.mkstemp(suffix=".tmp", prefix="ckpt_", dir=checkpoint_dir)
os.close(fd)  # Close fd before torch.save
torch.save(state, temp_path)
os.replace(temp_path, target_path)  # Atomic on most filesystems
```

### Tensor Deep Copy
```python
if isinstance(obj, torch.Tensor):
    return obj.detach().clone().cpu()
```

### Shutdown with Stats
```python
def shutdown(self, timeout: float = 30.0) -> bool:
    self._queue.put(None)  # Sentinel
    self._thread.join(timeout=timeout)
    # Log: "Checkpointing: N saves, Xs total, avg Ys"
    return not self._thread.is_alive()
```

## Next Phase Readiness

**Phase 06 (Unified Run Logging):** Ready. AsyncCheckpointWriter provides the foundation for non-blocking I/O patterns that can be extended to logging.

**Phase 07 (Adaptive Population):** Ready. Checkpoint infrastructure is now robust enough for longer evolution runs with larger populations.
