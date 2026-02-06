# Global Refactor Audit Report

**Date:** 2026-02-05
**Goal:** Clean modular code, simple API interfaces, clean data handoffs

---

## Executive Summary

The codebase has a **solid architectural foundation** (clean layering, no circular deps, good registry pattern) but has accumulated significant complexity debt in three core areas:

1. **Giant files with mixed concerns** (evaluator.py 68KB, strategies.py 47KB, weighted_sum.py 31KB, config.py 32KB)
2. **`Dict[str, Any]` used everywhere** instead of typed contracts for data handoffs
3. **Mutable shared state scattered across modules** (genome mutation in-place, singleton caches, dual-source-of-truth config)

The refactor should focus on **simplifying interfaces**, **making data handoffs explicit and typed**, and **breaking apart the 5-6 god files**.

---

## Part 1: GOD OBJECTS & OVERSIZED FILES

### 1.1 PopulationEvaluator (evaluator.py - 68KB)
**Location:** `evolution/execution/evaluator.py`
**Problem:** Single class handles evaluation orchestration, memory management, caching, metric batching, decision tracking, early exit logic, shared precompute coordination.
- Constructor takes 17 parameters
- Three representations of same config (EvaluatorConfig dataclass, __init__ params, Builder with 10+ setters)
- Modifies genomes in-place during evaluation (sets `.metrics`, `.fitness`, `.evaluated`)

**Refactor:**
- Extract `EvaluationOrchestrator` (main loop only)
- Move caching to standalone `CacheManager`
- Move early-exit logic to a `EarlyExitPolicy` strategy
- Accept a single config object instead of 17 params
- Return evaluation results instead of mutating genomes

### 1.2 SwarmRetriever (swarm_retriever.py - 1000+ LOC)
**Location:** `core/swarm_retriever.py`
**Problem:** Combines vector search, graph traversal, agent simulation, pheromone management, query preprocessing, batch orchestration, profiling, buffer pool management, and threading.
- 20+ methods mixing different responsibilities
- `TraversalBufferPool` is a thin wrapper that just returns tensor views (over-abstraction)
- `HeuristicContext` is a god-dataclass with 10+ heterogeneous fields passed to every heuristic

**Refactor:**
- Split into `VectorSearch`, `GraphTraversal`, `AgentSimulator`, `ResultsRanker`
- Replace `TraversalBufferPool` with a simple dict or inline allocation
- Split `HeuristicContext` into focused context types per heuristic category

### 1.3 strategies.py (47KB)
**Location:** `evolution/execution/strategies.py`
**Problem:** All genetic operators (mutations, crossovers, selections, initialization) in one file. Deep nesting (5+ levels) in crossover functions. Magic probabilities hardcoded (`0.7`, `0.5`).

**Refactor:**
- Split into `mutations.py`, `crossovers.py`, `selections.py`, `initialization.py`
- Extract magic probabilities to config constants
- Flatten nested crossover logic

### 1.4 config.py (32KB)
**Location:** `evolution/types/config.py`
**Problem:** 7 nested config dataclasses + `EvolutionContext` which mixes immutable config, quasi-immutable data, and mutable runtime state. `EvolutionContext` has dual source of truth (top-level fields AND `._state` property that creates stale snapshots).

**Refactor:**
- Split into `config.py` (immutable configs only) and `state.py` (mutable runtime state)
- Remove `EvolutionContext` dual representation - single `EvolutionState` class
- Each config should be self-contained (no 7-level nesting)

### 1.5 weighted_sum.py (31KB)
**Location:** `evolution/execution/weighted_sum.py`

**Refactor:** Audit for extractable concerns; may be acceptable given GPU kernel complexity.

---

## Part 2: DATA HANDOFFS - Dict[str, Any] Everywhere

### 2.1 Genome Compilation Returns Untyped Dict
**Location:** `evolution/types/genome.py:327-401`
**Chain:** `Genome -> GenomeCompiler.compile() -> Dict[str, Any] -> **unpacked into retrieve_batch()`
**Problem:** No validation that required keys exist or types are correct. Compile() called multiple times per genome despite unchanged input.

**Fix:** Define `CompiledGenome` TypedDict/dataclass as the compile output type. Cache compilation result.

### 2.2 AgentGroupConfig Uses Dict[str, Any] for Strategies
**Location:** `interfaces/types.py:25-67`
```python
class AgentGroupConfig(TypedDict):
    movement_strategies: Dict[str, Any]  # Should be typed
    deposit_strategies: Dict[str, Any]   # Should be typed
```

**Fix:** Define `StrategyConfig` with proper types for function references and weights.

### 2.3 Protocol Methods Return Dict[str, Any]
**Location:** `evolution/protocols.py:49-88`
- `summarize_for_strategic() -> Dict[str, Any]`
- `summarize_for_tactical() -> Dict[str, Any]`
- `to_summary_dict() -> Dict[str, Any]`

**Fix:** Define TypedDict schemas for each return type.

### 2.4 Genome.metrics Is a Raw Dict
**Location:** `evolution/types/genome.py:81`
```python
metrics: Dict[str, float] = field(default_factory=dict)
```
Directly assigned by evaluator. No validation. No guarantee of which keys exist.

**Fix:** Define `EvaluationMetrics` dataclass with explicit fields (hit_at_1, hit_at_5, mrr, recall_at_20, etc.)

### 2.5 Missing Genome.from_dict()
**Location:** `evolution/types/genome.py:207-241`
`to_dict()` exists but no symmetric `from_dict()`. JSON checkpoints rely on pickle or manual dict unpacking. Inconsistent enum handling in serialization.

**Fix:** Add `Genome.from_dict()` classmethod.

---

## Part 3: MUTABLE STATE & COUPLING

### 3.1 Genome Mutated In-Place During Evaluation
**Location:** `evolution/execution/evaluator.py:1211-1217`
```python
genome.metrics = final_metrics
genome.fitness = self.fitness_calc.calculate(final_metrics, genome)
genome.evaluated = True
```
Four separate fields represent evaluation state (fitness, metrics, latency, evaluated) - can get out of sync.

**Fix:** Return immutable `EvaluationResult` objects instead of mutating genomes. Or at minimum, use a single `genome.set_evaluation_result(result)` method that sets all fields atomically.

### 3.2 Singleton Caches Without Thread Safety
**Location:** `evolution/execution/embedding_cache.py:570-605`
`EmbeddingCacheProvider._instance` is a class variable accessed without locks. `clear()` can race with active evaluation threads.

**Fix:** Add `threading.Lock`, or pass cache instances via dependency injection instead of singletons.

### 3.3 Genome._compiled_cache Stale After Mutation
**Location:** `evolution/types/genome.py:85, 164-166`
Changes to `genome.strategies` don't invalidate `_compiled_cache`. Cache is annotated as `CompiledStrategies` but defaults to `dict`.

**Fix:** Invalidate cache on any strategy mutation, or make strategies immutable and return new genomes on mutation.

### 3.4 Feature Detection Via hasattr()
**Location:** `evolution/execution/evaluator.py:418-422`, `evolution/execution/shared_precompute.py:148-163`
```python
if hasattr(self.retriever, 'retrieve_batch_with_precomputed'):
    ...
if hasattr(retriever, '_get_cached_query_embeddings_batch'):  # Private method!
    ...
```

**Fix:** Define capabilities in the protocol/interface. Use explicit flags or methods.

---

## Part 4: INCONSISTENT APIs

### 4.1 Cache Stats Have Different Field Names
**Location:** `evolution/execution/fitness_cache.py` vs `evolution/execution/embedding_cache.py`
- FitnessCache: `hits`, `misses`, `hit_rate`
- EmbeddingCache: `cache_hits`, `cache_misses`, `generation_hits`, `generation_misses`, `hit_rate`

**Fix:** Create shared `CacheStats` protocol/base class.

### 4.2 Null Object Pattern Incomplete
**Location:** `evolution/protocols.py:126-220`
`NullJournal.record_mutation()` returns `None` while protocol expects `Any`. `NullTracker.__init__()` accepts `**kwargs` but ignores them.

**Fix:** Ensure null implementations fully satisfy protocol contracts.

### 4.3 VectorStore.compute_neighbor_similarities Returns Optional
**Location:** `interfaces/abstract_classes.py:84-107`
Returns `None` to signal "use fallback" - forces every caller to handle both cases.

**Fix:** Always return tensor. Implementations handle GPU vs CPU internally.

### 4.4 SwarmRetrieverAdapter Return Type Depends on Hidden State
**Location:** `evolution/adapters/swarm_adapter.py:148-230`
Return type varies based on `self.use_new_api` flag. Type annotation says `List[Any]` but content differs.

**Fix:** Return consistent type regardless of API mode.

### 4.5 Embedding Providers Return Different Types
**Location:** `integrations/cohere_embed.py` vs `integrations/gemini_embed.py`
- Cohere `embed_query_batch` returns `torch.Tensor`
- Gemini `embed_query_batch` returns `List[torch.Tensor]`

**Fix:** Standardize return type in `EmbeddingProvider` ABC.

---

## Part 5: CODE QUALITY

### 5.1 Bare Except Clauses (CRITICAL)
| File | Line | Fix |
|------|------|-----|
| `integrations/cohere_embed.py` | 9 | `except (ImportError, ModuleNotFoundError):` |
| `integrations/gemini_embed.py` | 9 | `except (ImportError, ModuleNotFoundError):` |

### 5.2 Silent Exception Swallowing (12+ instances)
| File | Lines | Issue |
|------|-------|-------|
| `core/swarm_retriever.py` | 1492-1500 | 3x `except Exception: pass` in `_capture_heuristic_scores()` |
| `utils/memory.py` | 205, 218, 371 | GPU memory returns 0 silently |
| `utils/device.py` | 132 | GPU detection returns empty dict |
| `evolution/storage/run_manager.py` | 99 | deepcopy failure returns shallow copy |
| `evolution/execution/strategies.py` | 403 | Crossover failure hidden |
| `interfaces/registry.py` | 109 | Enum parsing error swallowed |

**Fix:** Add specific exception types and logging. Document intentional fallbacks.

### 5.3 Code Duplication
| Location | Issue |
|----------|-------|
| `swarm_retriever.py:1490-1502` | Same try/except pattern repeated 3x - extract to loop |
| `tests/evolution/test_evaluator.py` | MockRetriever, MockEvaluator, MockFitnessCalc defined 3x each - move to conftest |
| `strategies.py` | `subtree_crossover()` and `root_mix_crossover()` share identical nesting pattern |

### 5.4 Magic Numbers
| File | Value | Should Be |
|------|-------|-----------|
| `strategies.py:378` | `0.7` (crossover probability) | `CROSSOVER_BIAS` constant |
| `strategies.py:408` | `0.5` (parent selection) | `PARENT_SELECTION_PROB` constant |
| `swarm_retriever.py:2330` | `0.3/0.4/0.3` (heuristic weights) | Config parameter |
| `fitness.py:240` | `1e-10` (epsilon) | Named constant |
| `evaluator.py:73` | `4` (concurrent evaluations) | Already in config, use it |

### 5.5 Late Imports
| File | Line | Fix |
|------|------|-----|
| `evaluator.py` | 313 | Move `from ...utils.device import get_device` to top |
| `evaluator.py` | 332 | Move `from .weighted_sum import WeightedSumCompiler` to top |
| `fitness.py` | 233 | Move `import math` to top |

### 5.6 Test Quality
- Module-level globals for temp dirs instead of pytest fixtures
- Empty mock classes (`class MockRetriever: pass`) don't implement interfaces
- Integration tests have minimal assertions (verify existence, not behavior)

---

## Part 6: MODULE ORGANIZATION

### 6.1 evolution/execution/ Is Too Large (16 files)
**Proposed split:**
```
evolution/execution/
  evaluator.py          # Keep - evaluation orchestration
  factory.py            # Keep - genome creation
  fitness/
    calculator.py       # From fitness.py
    strategies.py       # From fitness_strategies.py
    cache.py            # From fitness_cache.py
  optimization/
    embedding_cache.py
    shared_precompute.py
    stratified_sampler.py
    weighted_sum.py
  genetics/
    strategies.py       # From strategies.py (genetic ops)
    llm_strategies.py
  monitoring/
    memory_guard.py
    memory_logger.py
    profiler.py
    tracker.py
```

### 6.2 evolution/llm/__init__.py Exports 50+ Items
**Fix:** Only export `LLMBridge` as public API. Everything else is internal.

### 6.3 core/__init__.py Exposes Internals
`HeuristicRegistry` and `HeuristicContext` are exported but should be internal implementation details.

---

## Prioritized Implementation Plan

### Phase 1: Type Safety & Data Contracts (Lowest risk, highest clarity gain)
1. [x] Define `CompiledGenome` dataclass to replace `Dict[str, Any]` from compile()
2. [x] Define `EvaluationMetrics` dataclass to replace `genome.metrics: Dict`
3. [x] Add `Genome.from_dict()` for symmetric serialization
4. [x] Standardize embedding provider return types
5. [x] Fix bare except clauses (2 files, trivial)

### Phase 2: Evaluator Decomposition (Highest impact single file)
1. [x] Extract evaluation result as atomic setter (genome.set_evaluation_result()) - 10 sites updated
2. [x] Split PopulationEvaluator: extracted EarlyExitPolicy (early_exit.py) + CacheCoordinator (cache_coordinator.py)
3. [x] Single config object instead of 17 constructor params (EvaluatorConfig dataclass + **kwargs compat)
4. [x] Thread-safe caching (EmbeddingCacheProvider._lock)

### Phase 3: File Splits (Organizational clarity)
1. [x] Split `strategies.py` into mutations/crossovers/selections/initialization + genetic_registry.py
2. [x] Fix `config.py` dual state - EvolutionContext properties delegate to EvolutionState
3. [x] Reorganize `evolution/execution/` into subdirectories (fitness/, optimization/, genetics/, monitoring/)
4. [x] Clean up `evolution/llm/` exports (68 -> 5 in __all__)

### Phase 4: Interface Cleanup
1. [x] Remove `Optional` returns from VectorStore (returns -inf tensor, caller uses isinf check)
2. [x] Unify cache stats interfaces (CacheStatsProtocol)
3. [x] Fix null object implementations (NullMutationRecord sentinel, NullTracker **kwargs)
4. [x] Remove `hasattr()` feature detection - replaced with getattr(..., None) in evaluator + shared_precompute
5. [x] Consistent adapter return types

### Phase 5: Code Quality Sweep
1. [x] Add logging to all silent exception handlers (6/6 sites)
2. [x] Extract magic numbers to named constants/config
3. [x] Deduplicate repeated patterns (_crossover_preamble, _pick_parent_tree, heuristic capture loop)
4. [x] Move late imports to module level
5. [x] Upgrade test fixtures to pytest patterns

### Phase 6: Module Organization (from Part 6)
1. [x] Clean up `core/__init__.py` exports (removed HeuristicContext, HeuristicRegistry from __all__)
2. [x] Reorganize `evolution/execution/` into subdirectories + moved misplaced root files (focused_mutation→genetics/, protocols→llm/)

### All items complete (25/25)
