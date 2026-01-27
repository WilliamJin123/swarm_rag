# Evolution Loop Optimization Design

**Date:** 2026-01-26
**Status:** Ready for implementation

## Overview

Two optimizations to reduce evolution loop latency and support 500+ generations within a few hours:

1. **Early exit at 25%** - Filter bad genomes faster
2. **Async checkpointing** - Remove I/O from critical path

## Optimization 1: Early Exit at 25%

### Current State
- Single checkpoint at 50% (halfway) of queries
- Threshold: 0.30 quality score
- Bad genomes waste 50% of query budget before rejection

### Proposed Change
- Move checkpoint to 25% (quarter) of queries
- Keep same threshold (0.30)
- Bad genomes rejected 3x faster

### Implementation

**File:** `swarm_rag_module/swarm_rag/evolution/execution/evaluator.py`

Change in `_evaluate_single_with_shared()`, `_evaluate_single()`, and related methods:

```python
# From:
halfway = n_queries // 2

# To:
quarter = n_queries // 4
```

Update logging messages from "halfway" to "quarter" for clarity.

### Expected Impact
- Broken genomes (quality < 0.30): Exit at 25% instead of 50%
- ~50% faster rejection of bad genomes
- No impact on good genomes (still get full evaluation)

---

## Optimization 2: Async Checkpointing

### Current State
- `save_checkpoint()` does 2 synchronous `torch.save()` calls
- Blocks evolution loop during disk I/O
- 500 generations @ checkpoint every 5 gens = 100 blocking saves
- Estimated overhead: 5-10 minutes over full run

### Proposed Change
- Background thread handles checkpoint I/O
- Main thread queues checkpoint and continues immediately
- Graceful shutdown flushes pending checkpoint

### Architecture

```
Main Thread                    Background Thread
-----------                    -----------------
save_checkpoint() called
  |
  +-> Build state dict (fast, in-memory)
  |
  +-> Queue checkpoint data --------->  Receives data
  |                                      |
  +-> Return immediately                 +-> torch.save(gen_XXX.pkl)
      (evolution continues)              +-> Atomic latest.pkl update
                                         +-> Cleanup old checkpoints
```

### Implementation

**File:** `swarm_rag_module/swarm_rag/evolution/types/config.py`

Add to `StorageConfig`:
```python
async_checkpoints: bool = True  # Enable async checkpoint writing
```

**File:** `swarm_rag_module/swarm_rag/evolution/storage/run_manager.py`

Add new class:
```python
class AsyncCheckpointWriter:
    """Background thread for non-blocking checkpoint writes."""

    def __init__(self):
        self._queue = Queue(maxsize=1)
        self._thread = Thread(target=self._writer_loop, daemon=True)
        self._shutdown = Event()
        self._thread.start()

    def _writer_loop(self):
        """Background loop that processes checkpoint writes."""
        while not self._shutdown.is_set():
            try:
                item = self._queue.get(timeout=0.5)
                if item is None:  # Shutdown signal
                    break
                state, gen_path, latest_path, keep_n = item
                self._write_checkpoint(state, gen_path, latest_path, keep_n)
                self._queue.task_done()
            except Empty:
                continue

    def _write_checkpoint(self, state, gen_path, latest_path, keep_n):
        """Perform the actual checkpoint write."""
        # Save numbered checkpoint
        torch.save(state, gen_path)

        # Atomic update of latest.pkl
        temp = latest_path + ".tmp"
        torch.save(state, temp)
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.rename(temp, latest_path)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(os.path.dirname(gen_path), keep_n)

    def save(self, state, gen_path, latest_path, keep_n):
        """Queue checkpoint for async write. Drops old pending if queue full."""
        try:
            self._queue.put_nowait((state, gen_path, latest_path, keep_n))
        except Full:
            # Drop old pending checkpoint, queue new one
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            self._queue.put_nowait((state, gen_path, latest_path, keep_n))

    def flush(self):
        """Block until pending checkpoint is written."""
        self._queue.join()

    def shutdown(self):
        """Stop the background thread gracefully."""
        self._shutdown.set()
        self._queue.put(None)  # Wake up thread
        self._thread.join(timeout=5.0)
```

Modify `RunManager.__init__()`:
```python
def __init__(self, config: "StorageConfig", device: torch.device = None):
    self.config = config
    # ... existing code ...

    # Initialize async writer if enabled
    self._async_writer = None
    if getattr(config, 'async_checkpoints', True):
        self._async_writer = AsyncCheckpointWriter()
```

Modify `RunManager.save_checkpoint()`:
```python
def save_checkpoint(self, population, best_genome, generation, ...):
    state = {
        "generation": generation,
        "population": population,
        # ... rest of state dict ...
    }

    gen_path = self.config.checkpoint_path_for_gen(generation)
    latest_path = self.config.latest_checkpoint_path

    if self._async_writer:
        self._async_writer.save(
            state, gen_path, latest_path,
            self.config.keep_n_checkpoints
        )
        logger.info(f"Checkpoint queued: {gen_path}")
    else:
        # Existing synchronous path
        torch.save(state, gen_path)
        # ... rest of existing code ...
```

Add cleanup method:
```python
def close(self):
    """Flush pending checkpoints and shutdown async writer."""
    if self._async_writer:
        self._async_writer.flush()
        self._async_writer.shutdown()
```

### File Types Preserved
- `.pkl` for checkpoints (torch.save)
- `.json` for config/genomes/metrics
- `.jsonl` for logs

### Expected Impact
- Checkpoint I/O no longer blocks evolution loop
- ~5-10 minutes saved over 500 generation run
- Graceful handling of rapid successive checkpoints

---

## Files Modified

| File | Changes |
|------|---------|
| `evolution/types/config.py` | Add `async_checkpoints` to `StorageConfig` |
| `evolution/storage/run_manager.py` | Add `AsyncCheckpointWriter`, modify `save_checkpoint()`, add `close()` |
| `evolution/execution/evaluator.py` | Change `halfway` to `quarter` in early exit logic |

## Files NOT Modified

- Swarm retrieval code (`core/swarm_retriever.py`)
- Query sampling (handled externally by `evolve_stark.py`)
- Metric computation (`eval/metric_functions.py`)

## Testing

1. Run short evolution (10 generations) and verify:
   - Early exit triggers at 25% for bad genomes
   - Checkpoints written correctly in background
   - `latest.pkl` always valid (atomic update)

2. Verify checkpoint loading still works:
   - Load checkpoint from previous run
   - Resume evolution correctly

3. Verify graceful shutdown:
   - Final checkpoint written before exit
