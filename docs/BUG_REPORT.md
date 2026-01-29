# Code Audit: Bug Report and Code Smells

**Date:** 2026-01-28
**Scope:** Full codebase audit focusing on evolution loop, GPU memory management, and design issues

---

## Critical: GPU Memory Leak (Causes Slowdown Over Generations)

### 1. Shared Context Tensors Not Released Between Genomes
**File:** `swarm_rag_module/swarm_rag/evolution/execution/evaluator.py:369-377`

`ground_truth_tensor` and `gt_sizes` tensors in `SharedPrecomputeContext` are only deleted at the end of `_evaluate_all_with_shared`, but expanded copies (`gt_tensor_expanded`, `gt_sizes_expanded`) created in `_batch_compute_metrics_all_genomes` are never explicitly deleted.

**Fix:** Add `del gt_tensor_expanded; del gt_sizes_expanded` after use and call `torch.cuda.empty_cache()` inside the loop.

### 2. Step-Level Tensor Accumulation in Batched Traversal
**File:** `swarm_rag_module/swarm_rag/core/swarm_retriever.py:1092-1094`

`id_to_idx_tensor` (size: `_max_node_id + 1`) is created fresh every step inside `_step_agents_batched` but never explicitly deleted, causing fragmentation over long traversals.

**Fix:** Pre-allocate `id_to_idx_tensor` once per query and reuse with `.fill_(-1)` each step.

### 3. Feature Registry Creates New Random Tensors Every Step
**File:** `swarm_rag_module/swarm_rag/core/swarm_retriever.py:1147`

`"random_jitter": torch.rand_like(semantic_scores) * 0.1` creates new GPU tensors every step without cleanup, accumulating memory.

**Fix:** Pre-allocate jitter buffer once per query and use `torch.rand_like(_, out=buffer)`.

### 4. Pheromone Buffer Never Shrinks
**File:** `swarm_rag_module/swarm_rag/core/swarm_retriever.py:160`

`_pheromone_buffer_size = max(self._max_node_id + 1024, 150000)` hardcodes minimum 150k floats (~600KB per query) regardless of actual graph size.

**Fix:** Set buffer to `_max_node_id + 1` without the arbitrary minimum.

### 5. BatchedRetrievalResults.clear() Doesn't Free GPU Memory
**File:** `swarm_rag_module/swarm_rag/evolution/execution/shared_precompute.py:334-340`

`clear()` deletes Python reference but doesn't call `torch.cuda.empty_cache()`, leaving memory in CUDA cache.

**Fix:** Add `if torch.cuda.is_available(): torch.cuda.empty_cache()` after deletion.

### 6. Empty Cache Called Too Late in Evaluation Loop
**File:** `swarm_rag_module/swarm_rag/evolution/execution/evaluator.py:341, 429`

`torch.cuda.empty_cache()` is called after each genome in `_evaluate_all_with_shared`, but the real memory hogs (intermediate metric tensors) accumulate inside `_batch_compute_metrics_all_genomes` before this call.

**Fix:** Move `empty_cache()` inside `_batch_compute_metrics_all_genomes` after each genome's metrics are computed.

---

## High: Logic Bugs and Incorrect Behavior

### 7. Lambda Closure Bug in Feature Getters
**File:** `swarm_rag_module/swarm_rag/evolution/execution/weighted_sum.py:68-84`

Lambdas in `_build_feature_getters` capture loop variable `name` by reference, not value. All fallback lambdas will return the same (last) value of `name`.

```python
# Bug: All fallbacks use same `name`
self._feature_getters[name] = lambda ctx, n=name: torch.zeros(1)
```

**Fix:** Already has `n=name` default arg which is correct, but the pattern at line 84 should be verified to use the default arg.

### 8. Offspring Counter Never Reset
**File:** `swarm_rag_module/swarm_rag/evolution/map_elites/loop.py:35`

`self._offspring_counter` increments indefinitely across generations, causing genome IDs like `g999_c50000` instead of `g999_c1`.

**Fix:** Reset counter at the start of each generation's `step()` method.

### 9. Archive Uses eval() for Checkpoint Restoration
**File:** `swarm_rag_module/swarm_rag/evolution/map_elites/archive.py:408`

`key = eval(key_str)` is a security risk and can crash on malformed input.

**Fix:** Use `ast.literal_eval(key_str)` instead.


### 11. select_random Returns Stored Reference
**File:** `swarm_rag_module/swarm_rag/evolution/map_elites/archive.py:286-291`

`select_random()` returns `self.grid[key]` directly, but `add()` stores copies. If caller mutates returned genome, archive integrity is compromised.

**Fix:** Return `self.grid[key].copy()` to maintain immutability guarantee.

---

## Medium: Hardcoded Magic Numbers

### 12. Early Exit Threshold Hardcoded
**File:** `swarm_rag_module/swarm_rag/evolution/execution/evaluator.py:38`

`DEFAULT_EARLY_EXIT_THRESHOLD: float = 0.30` is module-level constant that should be configurable per-run.

**Fix:** Move to `ResourceConfig` and pass through constructor.


### 15. Mutation Probabilities Hardcoded
**File:** `swarm_rag_module/swarm_rag/evolution/execution/weighted_sum.py:261-265`

```python
PROB_WEIGHT = 0.60
PROB_BIAS = 0.15
# ...
```

These should be configurable for different evolution strategies.

**Fix:** Move to `GeneticConfig` dataclass.

### 16. Profiler Max Samples Hardcoded
**File:** `swarm_rag_module/swarm_rag/core/swarm_retriever.py:20`

`max_samples_per_section: int = 1000` limits profiling granularity without user control.

**Fix:** Allow override via environment variable or constructor.

---

## Medium: Code Duplication

### 17. Duplicate Seed Genome Definitions
**Files:**
- `swarm_rag_module/swarm_rag/evolution/seed_genomes.py:22-138` (expression tree format)
- `swarm_rag_module/swarm_rag/evolution/execution/weighted_sum.py:497-627` (weighted sum format)

Two completely separate seed definitions for the same strategies.

**Fix:** Create single source of seed configurations and convert to appropriate format at runtime.

### 18. Duplicate Default Parameter Definitions
**Files:**
- `swarm_rag_module/swarm_rag/core/swarm_retriever.py:81-103` (`_DEFAULT_PARAMS`)
- `swarm_rag_module/swarm_rag/evolution/types/genome.py:54-61` (`DEFAULT_PARAMS`)

Same defaults defined twice with risk of divergence.

**Fix:** Single authoritative source in swarm_retriever, import where needed elsewhere.

### 19. Duplicate Device Resolution Logic
**Files:**
- `swarm_rag_module/swarm_rag/evolution/types/config.py:36-39`
- `swarm_rag_module/swarm_rag/utils/device.py`
- `swarm_rag_module/swarm_rag/evolution/execution/evaluator.py:129-134`

Device detection repeated in multiple places with subtle differences.

**Fix:** Use single `get_device()` from utils everywhere.

---

## Medium: Design Smells

### 20. EvolutionContext Mixes Config and State
**File:** `swarm_rag_module/swarm_rag/evolution/types/config.py:603-664`

`EvolutionContext` holds both immutable config and mutable runtime state (generation, stagnation_count, etc.), violating single responsibility.

**Fix:** Split into `EvolutionConfig` (immutable) and `EvolutionState` (mutable).

### 21. PopulationEvaluator Has 17 Constructor Parameters
**File:** `swarm_rag_module/swarm_rag/evolution/execution/evaluator.py:78-97`

Too many parameters indicates the class is doing too much.

**Fix:** Use builder pattern

### 22. WeightTensors Default Factory Calls get_device()
**File:** `swarm_rag_module/swarm_rag/evolution/types/config.py:51-59`

`default_factory=lambda: torch.zeros(1, 4, device=_get_default_device())` triggers CUDA initialization at import time.

**Fix:** Use `"cpu"` as default device in dataclass, move to target device explicitly when used.

### 23. SharedPrecomputeContext Has Nullable Required Fields
**File:** `swarm_rag_module/swarm_rag/evolution/execution/shared_precompute.py:21-48`

`ground_truth_tensor` and `gt_sizes` are `Optional` but required for GPU-accelerated metrics, leading to runtime checks everywhere.

**Fix:** Create two classes: `SharedPrecomputeContextCPU` and `SharedPrecomputeContextGPU`.

### 24. FitnessCalculator Has Two Init Patterns
**File:** `swarm_rag_module/swarm_rag/evolution/execution/fitness.py:80-101`

Can init with either `weights` or `config`, creating divergent code paths and confusion.

**Fix:** Single constructor taking `FitnessConfig`, remove legacy weights and usage of legacy weights anywhere within the codebase.

### 25. Genome Mode as String Literal
**File:** `swarm_rag_module/swarm_rag/evolution/types/genome.py:77`

`mode: Literal["weighted_sum", "expression_tree"]` uses string literals instead of enum.

**Fix:** Create `GenomeMode` enum for type safety and IDE support.

---

## Low: Minor Issues

### 26. Logger Spam in Hot Path
**File:** `swarm_rag_module/swarm_rag/core/swarm_retriever.py:903-904`

`logger.debug(f"Agent {agent_id} at {current_loc}...")` called every other step per agent. With 25 agents x 5 steps x 100 queries = 12,500 log calls per genome even at DEBUG level.

**Fix:** Move to trace level or remove.

### 27. Progress Logging Uses Magic Interval
**File:** `swarm_rag_module/swarm_rag/core/swarm_retriever.py:447-448`

`if (i + 1) % 10 == 0` hardcodes progress log interval.

**Fix:** Use tqdm.

### 28. Type Annotation Missing Self-Reference
**File:** `swarm_rag_module/swarm_rag/evolution/map_elites/loop.py:146`

`-> Optional["StrategicDirective"]` uses string quote for forward reference but imports inside method.

**Fix:** Use `from __future__ import annotations` at module level.

### 29. Inconsistent Error Handling
**File:** `swarm_rag_module/swarm_rag/evolution/execution/evaluator.py:1326`

GPU metric failures silently fall back to CPU without logging the actual error message to warn level.

**Fix:** GPU failures should just fail without a CPU fallback and explcitly fail instead of silently. this applies accross the whole codebase as cpu<->gpu overhead would be too much to deal with anyways.

### 30. Thread Pool Not Bounded by Available Cores
**File:** `swarm_rag_module/swarm_rag/evolution/map_elites/loop.py:124`

`ThreadPoolExecutor(max_workers=max_workers)` doesn't check against `os.cpu_count()`.

**Fix:** Use `min(max_workers, os.cpu_count() or 4)`.

---

## Summary

| Severity | Count | Key Areas |
|----------|-------|-----------|
| Critical (Memory Leak) | 6 | GPU tensors, batched evaluation, step buffers |
| High (Logic Bugs) | 5 | Closures, counters, archive safety |
| Medium (Hardcoded) | 5 | Thresholds, probabilities, limits |
| Medium (Duplication) | 3 | Seeds, defaults, device logic |
| Medium (Design) | 6 | Mixed concerns, too many params, nullable fields |
| Low | 5 | Logging, types, error handling |

**Total: 30 issues identified**

The GPU memory leak is almost certainly caused by issues #1-6. Start there for the slowdown investigation.
