# Phase 5: Async Checkpointing - Research

**Researched:** 2026-01-30
**Domain:** Python threading, queue-based producer-consumer, atomic file operations
**Confidence:** HIGH

## Summary

This phase implements non-blocking checkpoint saves during evolution. The standard approach uses Python's `threading.Thread` with a `queue.Queue` for producer-consumer communication. The evolution loop (producer) queues checkpoint data, while a background thread (consumer) writes to disk without blocking the main loop.

The existing codebase already has an `AsyncCheckpointWriter` implementation in `run_manager.py`, but it needs refinement to match the CONTEXT.md decisions:
1. Queue all checkpoints (current implementation drops stale ones)
2. Wait for queue drain on shutdown (current implementation uses daemon thread)
3. Add comprehensive logging (current logging is minimal)
4. Implement retry logic on write failures

**Primary recommendation:** Refactor the existing `AsyncCheckpointWriter` to use a non-daemon thread with unlimited queue, sentinel-based shutdown, and enhanced logging/retry logic.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `threading.Thread` | stdlib | Background worker thread | Built-in, well-documented, sufficient for single-consumer pattern |
| `queue.Queue` | stdlib | Thread-safe producer-consumer queue | FIFO, blocking get, join() for drain verification |
| `threading.Event` | stdlib | Shutdown signaling | Clean signal mechanism, timeout-aware waiting |
| `copy.deepcopy` | stdlib | Deep copy checkpoint data | Handles nested objects, respects pickle protocol |
| `torch.save` | PyTorch | Serialize checkpoint data | Device-aware, handles tensors/genomes natively |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `tempfile.NamedTemporaryFile` | stdlib | Atomic write temp file | When need OS-managed cleanup |
| `os.replace` | stdlib | Atomic file rename | Cross-platform atomic overwrite |
| `logging` | stdlib | Status/feedback logging | All checkpoint operations |
| `time.perf_counter` | stdlib | High-resolution timing | Measuring checkpoint write duration |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `threading.Thread` | `ThreadPoolExecutor` | Overkill for single consumer; harder shutdown control |
| `queue.Queue` | `collections.deque` with Lock | Less convenient; manual locking needed |
| `copy.deepcopy` | Manual cloning | Error-prone for complex nested structures |
| Sentinel shutdown | `shutdown()` method | Only available in Python 3.13+; sentinel is universally compatible |

**Installation:**
```bash
# No additional packages needed - all stdlib + existing PyTorch
```

## Architecture Patterns

### Recommended Project Structure
```
swarm_rag_module/swarm_rag/evolution/
  storage/
    run_manager.py       # Contains RunManager + AsyncCheckpointWriter
    __init__.py          # Exports RunManager
```

### Pattern 1: Producer-Consumer with Sentinel Shutdown

**What:** Single background thread consumes from unbounded queue. Shutdown signaled via sentinel value (None).

**When to use:** Single consumer, need to process all items before shutdown, items are independent.

**Example:**
```python
# Source: Python docs + verified pattern
import threading
import queue
import logging
from typing import Optional, Tuple, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

SENTINEL = None  # Shutdown signal

@dataclass
class CheckpointStats:
    """Statistics for checkpoint operations."""
    total_saves: int = 0
    total_time_seconds: float = 0.0

    @property
    def avg_time_seconds(self) -> float:
        return self.total_time_seconds / self.total_saves if self.total_saves > 0 else 0.0

class AsyncCheckpointWriter:
    def __init__(self):
        self._queue: queue.Queue = queue.Queue()  # Unbounded
        self._shutdown_event = threading.Event()
        self._stats = CheckpointStats()

        # Non-daemon: allows graceful shutdown
        self._thread = threading.Thread(
            target=self._writer_loop,
            daemon=False,
            name="AsyncCheckpointWriter"
        )
        self._thread.start()

    def _writer_loop(self):
        """Consumer loop - processes all queued items until sentinel."""
        while True:
            try:
                item = self._queue.get(timeout=1.0)

                if item is SENTINEL:
                    self._queue.task_done()
                    break

                self._process_checkpoint(item)
                self._queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Checkpoint write error: {e}")
                self._queue.task_done()

    def _process_checkpoint(self, item: Tuple[dict, str, str]):
        """Write checkpoint with retry logic."""
        state, gen_path, latest_path = item
        start = time.perf_counter()

        success = self._write_with_retry(state, gen_path, latest_path)

        elapsed = time.perf_counter() - start
        if success:
            self._stats.total_saves += 1
            self._stats.total_time_seconds += elapsed

    def queue_checkpoint(self, state: dict, gen_path: str, latest_path: str):
        """Queue checkpoint for async write."""
        queue_depth = self._queue.qsize()
        if queue_depth > 1:
            logger.warning(f"Checkpoint queue: {queue_depth} pending")

        self._queue.put((state, gen_path, latest_path))
        logger.info(f"Checkpoint queued (gen {state.get('generation', '?')})")

    def shutdown(self, timeout: float = 30.0) -> bool:
        """Graceful shutdown: drain queue then stop thread."""
        logger.info("Shutting down checkpoint writer, draining queue...")

        # Signal shutdown
        self._queue.put(SENTINEL)

        # Wait for thread to finish
        self._thread.join(timeout=timeout)

        if self._thread.is_alive():
            logger.error("Checkpoint writer did not terminate in time")
            return False

        # Log summary
        logger.info(
            f"Checkpointing: {self._stats.total_saves} saves, "
            f"{self._stats.total_time_seconds:.1f}s total, "
            f"avg {self._stats.avg_time_seconds:.2f}s"
        )
        return True
```

### Pattern 2: Atomic Write with Temp File

**What:** Write to temp file in same directory, then atomic rename.

**When to use:** Preventing partial/corrupt files on crash or interruption.

**Example:**
```python
# Source: Python os.replace docs + tempfile docs
import os
import tempfile
import torch

def atomic_save(state: dict, target_path: str) -> None:
    """Save checkpoint atomically using temp file + rename."""
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)

    # Create temp file in same directory (ensures same filesystem)
    fd, temp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix="checkpoint_",
        dir=target_dir
    )

    try:
        # Close fd, let torch.save handle file
        os.close(fd)
        torch.save(state, temp_path)

        # Atomic replace (works on Windows with os.replace)
        os.replace(temp_path, target_path)

    except Exception:
        # Clean up temp file on failure
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
```

### Pattern 3: Deep Copy with Tensor Handling

**What:** Deep copy checkpoint data, ensuring tensors are detached and on CPU.

**When to use:** Before queuing data to background thread to prevent GPU memory pinning and data races.

**Example:**
```python
# Source: PyTorch docs on clone/detach + copy module docs
import copy
import torch
from typing import Any

def prepare_checkpoint_data(state: dict) -> dict:
    """
    Deep copy checkpoint state, moving tensors to CPU.

    This prevents:
    1. Data races if main thread modifies objects
    2. GPU memory being pinned by background thread
    3. CUDA context issues in background thread
    """
    def _copy_value(val: Any) -> Any:
        if isinstance(val, torch.Tensor):
            # Detach from graph, clone data, move to CPU
            return val.detach().clone().cpu()
        elif isinstance(val, dict):
            return {k: _copy_value(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [_copy_value(v) for v in val]
        elif hasattr(val, 'copy') and callable(val.copy):
            # Genome and other dataclasses with copy method
            return val.copy()
        else:
            # Immutable or already safe
            return copy.deepcopy(val)

    return _copy_value(state)
```

### Anti-Patterns to Avoid

- **Daemon threads for checkpoint writing:** Program may exit before checkpoints complete, causing data loss. Use non-daemon thread with explicit shutdown.

- **Dropping queued checkpoints:** If new checkpoint arrives while one is writing, dropping the old one loses data. Queue all checkpoints.

- **Direct write without temp file:** Crash during write creates corrupt checkpoint. Always use atomic write pattern.

- **Sharing mutable state with background thread:** Main thread may modify objects while background thread serializes. Deep copy before queuing.

- **GPU tensors in background thread:** Background thread accessing GPU memory can cause CUDA errors or pin memory. Move to CPU before queuing.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Thread-safe queue | Locked list | `queue.Queue` | Built-in locking, join/task_done semantics |
| Atomic file write | Manual remove+rename | `os.replace` with temp file | Cross-platform atomic semantics |
| Deep copy with tensors | Manual field copying | `copy.deepcopy` + tensor handling | Handles nested structures correctly |
| Shutdown signaling | Boolean flag | Sentinel + `queue.Queue` | No race conditions, works with blocking get |
| Time measurement | `time.time()` | `time.perf_counter()` | Higher resolution, monotonic |

**Key insight:** Python's stdlib provides robust thread-safe primitives. The complexity is in orchestrating them correctly, not in the primitives themselves.

## Common Pitfalls

### Pitfall 1: GIL Misconceptions
**What goes wrong:** Assuming background thread runs truly in parallel with main thread for CPU-bound work.
**Why it happens:** Python's GIL prevents parallel execution of Python bytecode.
**How to avoid:** For I/O-bound work (file writing), threading is fine - GIL is released during I/O syscalls. Checkpoint writing is I/O-bound.
**Warning signs:** Background thread doesn't speed up CPU-intensive operations.

### Pitfall 2: Queue.get() Blocking Forever
**What goes wrong:** Background thread blocks on `queue.get()` forever during shutdown.
**Why it happens:** Using infinite timeout without shutdown mechanism.
**How to avoid:** Use timeout with `queue.get(timeout=1.0)` or sentinel value.
**Warning signs:** Program hangs on exit, thread doesn't terminate.

### Pitfall 3: Partial Checkpoint on Crash
**What goes wrong:** Checkpoint file is corrupt or incomplete after crash during write.
**Why it happens:** Direct write to final path; crash mid-write leaves partial file.
**How to avoid:** Write to temp file, then atomic rename with `os.replace`.
**Warning signs:** `torch.load` fails with "unexpected EOF" or pickle errors.

### Pitfall 4: Race Condition on Shutdown
**What goes wrong:** Queue has items, but thread exits before processing them.
**Why it happens:** Setting shutdown flag while items still pending; using daemon thread.
**How to avoid:** Use sentinel value (None) as last queue item; use non-daemon thread; wait for queue.join() before thread.join().
**Warning signs:** Final checkpoints not written, log shows "N pending" at shutdown.

### Pitfall 5: CUDA Context in Background Thread
**What goes wrong:** Background thread tries to access GPU tensor, gets CUDA error.
**Why it happens:** Tensors on GPU passed to background thread; CUDA context is thread-local.
**How to avoid:** Deep copy tensors to CPU before queuing: `tensor.detach().clone().cpu()`.
**Warning signs:** "CUDA error: invalid device ordinal" or "CUDA context has been destroyed".

### Pitfall 6: Windows Atomic Rename Limitations
**What goes wrong:** `os.rename` fails with FileExistsError on Windows when destination exists.
**Why it happens:** Unlike Unix, Windows `os.rename` doesn't overwrite existing files.
**How to avoid:** Use `os.replace` instead - it atomically replaces on all platforms.
**Warning signs:** FileExistsError on Windows when updating latest.pkl.

## Code Examples

Verified patterns for implementation:

### Deep Copy Checkpoint State
```python
# Source: Python docs + PyTorch docs
import copy
import torch

def deep_copy_state(state: dict) -> dict:
    """Create a deep copy of checkpoint state, handling tensors specially."""
    copied = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            copied[key] = value.detach().clone().cpu()
        elif isinstance(value, list):
            copied[key] = [
                g.copy() if hasattr(g, 'copy') else copy.deepcopy(g)
                for g in value
            ]
        elif hasattr(value, 'copy') and callable(value.copy):
            copied[key] = value.copy()
        else:
            copied[key] = copy.deepcopy(value)
    return copied
```

### Retry Logic for Write Failures
```python
# Source: Standard retry pattern
import logging
import time

logger = logging.getLogger(__name__)

def write_with_retry(
    state: dict,
    path: str,
    max_retries: int = 1,
    retry_delay: float = 0.5
) -> bool:
    """Write checkpoint with retry on failure."""
    for attempt in range(max_retries + 1):
        try:
            atomic_save(state, path)
            return True
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    f"Checkpoint write failed (attempt {attempt + 1}), "
                    f"retrying in {retry_delay}s: {e}"
                )
                time.sleep(retry_delay)
            else:
                logger.error(f"Checkpoint write failed after {max_retries + 1} attempts: {e}")
                return False
    return False
```

### Logging Integration
```python
# Source: Logging best practices
import logging
import os
import time

logger = logging.getLogger(__name__)

def log_checkpoint_complete(
    generation: int,
    path: str,
    elapsed_seconds: float
):
    """Log checkpoint completion with size and timing."""
    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info(
        f"Checkpoint saved (gen {generation}, "
        f"{size_mb:.1f}MB, {elapsed_seconds:.2f}s)"
    )

def log_shutdown_summary(stats: CheckpointStats):
    """Log checkpoint summary on shutdown."""
    logger.info(
        f"Checkpointing: {stats.total_saves} saves, "
        f"{stats.total_time_seconds:.1f}s total, "
        f"avg {stats.avg_time_seconds:.2f}s"
    )
```

### Thread Daemon Status Recommendation

For this use case, **use daemon=False**:
```python
# Non-daemon thread - allows graceful shutdown
self._thread = threading.Thread(
    target=self._writer_loop,
    daemon=False,  # IMPORTANT: False allows queue drain before exit
    name="AsyncCheckpointWriter"
)
```

**Rationale:** Daemon threads are abruptly terminated when the main program exits, potentially losing queued checkpoints. Non-daemon threads allow the program to wait for all checkpoints to complete during graceful shutdown.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Synchronous checkpoint writes | Async background thread | Always available | Non-blocking evolution loop |
| `os.rename` for atomic writes | `os.replace` | Python 3.3+ | Cross-platform atomic overwrite |
| Manual locking with deque | `queue.Queue` | Always preferred | Built-in thread safety |
| Boolean shutdown flag | Sentinel value | Best practice | No race conditions |

**Deprecated/outdated:**
- `os.rename` for overwriting: Use `os.replace` instead (fails on Windows if dest exists)
- Daemon threads for I/O workers: Use non-daemon with explicit shutdown for data integrity
- `queue.Queue(maxsize=1)` with drop: Use unbounded queue for checkpoints (user decision)

## Open Questions

Things that couldn't be fully resolved:

1. **Archive serialization approach**
   - What we know: Archive contains genomes in grid dict; genomes have `.copy()` method
   - What's unclear: Whether to include full archive in checkpoint or rebuild on restore
   - Recommendation: Include full archive - rebuilding requires re-evaluation, which is expensive

2. **Optimal temp file naming convention**
   - What we know: Need unique names to avoid collisions; same directory ensures same filesystem
   - What's unclear: Whether to use random suffix or predictable pattern
   - Recommendation: Use `tempfile.mkstemp(suffix=".tmp", prefix="ckpt_", dir=checkpoint_dir)` - OS handles uniqueness

3. **Queue implementation choice**
   - What we know: Both `queue.Queue` and simple list with Lock work
   - What's unclear: Performance difference for typical checkpoint frequency
   - Recommendation: Use `queue.Queue` - cleaner API, built-in join/task_done, standard pattern

## Sources

### Primary (HIGH confidence)
- [Python threading docs](https://docs.python.org/3/library/threading.html) - Thread, Event, daemon behavior
- [Python queue docs](https://docs.python.org/3/library/queue.html) - Queue, join(), sentinel pattern
- [Python copy docs](https://docs.python.org/3/library/copy.html) - deepcopy behavior
- [Python tempfile docs](https://docs.python.org/3/library/tempfile.html) - mkstemp, NamedTemporaryFile
- [PyTorch tensor operations](https://discuss.pytorch.org/t/clone-and-detach-in-v0-4-0/16861) - clone, detach behavior

### Secondary (MEDIUM confidence)
- [os.replace vs os.rename comparison](https://www.pythontutorials.net/blog/difference-between-os-replace-and-os-rename/) - Cross-platform behavior
- [PyTorch async checkpointing blog](https://pytorch.org/blog/6x-faster-async-checkpointing/) - GIL considerations
- [Janus library](https://github.com/aio-libs/janus) - Thread-safe async queues (alternative reference)

### Tertiary (LOW confidence)
- [Background task processing patterns](https://danielsarney.com/blog/python-background-task-processing-2025-handling-asynchronous-work-modern-applications/) - General patterns

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All stdlib components with official documentation
- Architecture patterns: HIGH - Standard producer-consumer, well-documented
- Pitfalls: HIGH - Known issues with official workarounds
- Code examples: HIGH - Verified against official docs

**Research date:** 2026-01-30
**Valid until:** 2026-03-01 (60 days - stable stdlib components)
