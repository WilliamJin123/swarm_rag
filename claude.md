# SwarmRAG Evolution Module - Simplified Architecture

## Overview

The evolution module uses **MAP-Elites** as the default (and only) evolutionary paradigm with optional **LLM-guided mutations**. The architecture has been simplified from 43 files to ~31 files with organized nested configuration dataclasses.

---

## Quick Start

```python
from swarm_rag.evolution import EvolutionEngine, EvolutionConfig, MapElitesConfig, LLMConfig

# Create config with nested dataclasses
config = EvolutionConfig(
    n_generations=100,
    map_elites=MapElitesConfig(bins=[20, 15], initial_fill=200),
    llm=LLMConfig(enabled=True, provider='cerebras', model='zai-glm-4.7')
)

# Create engine
engine = EvolutionEngine(
    retriever=retriever,
    fitness_calculator=fitness_calc,
    evaluator=evaluator,
    train_query_ids=train_ids,
    train_ground_truth=train_gt,
    val_query_ids=val_ids,
    val_ground_truth=val_gt,
    config=config
)

# Run evolution
best_genome = engine.optimize()
```

---

## Configuration System

### New Dataclass Structure (Replaces flat 50+ field dict)

```python
@dataclass
class EvolutionConfig:
    n_generations: int = 50
    fitness_strategy: str = "lexicographic"  # lexicographic, pareto, phased

    resources: ResourceConfig        # concurrent_evaluations, max_workers
    map_elites: MapElitesConfig      # dimensions, bins, ranges, initial_fill, batch_size
    genetic: GeneticConfig           # mutation/crossover strategies, rates, boltzmann params
    llm: LLMConfig                   # enabled, provider, model, env_path
    checkpoint: CheckpointConfig     # log_path, checkpoint_path, frequencies
```

### Config Dataclasses

| Class | Key Fields |
|-------|------------|
| `ResourceConfig` | `concurrent_evaluations`, `max_workers_per_retrieval` |
| `MapElitesConfig` | `dimensions`, `bins`, `ranges`, `initial_fill`, `batch_size` |
| `GeneticConfig` | `mutation_strategy`, `crossover_strategy`, `base_mutation_rate`, `boltzmann.*` |
| `LLMConfig` | `enabled`, `provider`, `model`, `env_path` |
| `CheckpointConfig` | `log_path`, `plot_path`, `checkpoint_path`, `checkpoint_frequency` |

### Backwards Compatibility

```python
# Legacy flat dict still works
legacy_config = {"n_generations": 100, "map_elites_enabled": True, ...}
engine = EvolutionEngine(..., config=legacy_config)  # Auto-converted

# Convert between formats
flat = config.to_flat_dict()
config = EvolutionConfig.from_flat_dict(flat)
```

---

## Module Structure

```
swarm_rag_module/swarm_rag/evolution/
├── __init__.py                  # Exports EvolutionEngine, config classes
├── engine.py                    # Main facade (MAP-Elites only)
│
├── types/
│   ├── config.py                # EvolutionConfig + nested dataclasses
│   ├── genome.py                # Genome, SwarmParams
│   ├── expressions.py           # ExpressionNode, ExpressionEvolution
│   └── fitness_results.py       # FitnessResult
│
├── orchestrators/
│   ├── base.py                  # BaseOrchestrator (logging, checkpointing)
│   └── map_elites.py            # MAPElitesOrchestrator
│
├── execution/
│   ├── evaluator.py             # PopulationEvaluator
│   ├── fitness.py               # FitnessCalculator
│   ├── strategies.py            # GeneticRegistry (mutation, crossover, selection)
│   ├── llm_strategies.py        # LLM-guided mutations
│   ├── fitness_strategies.py    # Lexicographic, Pareto, Phased
│   ├── factory.py               # GenomeFactory
│   └── tracker.py               # ProgressTracker
│
├── map_elites/
│   ├── archive.py               # MapElitesArchive
│   ├── loop.py                  # MapElitesLoop (breeding)
│   └── descriptors/             # Behavioral descriptor registry
│
└── llm/
    ├── provider.py              # LLMProvider protocol
    ├── factory.py               # LLMProviderFactory
    ├── utils.py                 # apply_llm_edits, genome_to_json_context
    └── providers/
        └── universal.py         # UniversalLLMProvider (keycycle-based)
```

---

## CLI Usage (evolve_stark.py)

```bash
# Quick test
python stark/evolve_stark.py --preset toy

# Full run with LLM
python stark/evolve_stark.py --preset full --llm

# Custom settings
python stark/evolve_stark.py --dataset amazon --gens 50 --pop 30 --llm --llm-provider openai

# Available presets: toy, fast, full, llm, amazon, mag
```

---

## Key Files for Development

| Purpose | File |
|---------|------|
| Main entry point | `stark/evolve_stark.py` |
| Evolution engine | `evolution/engine.py` |
| Configuration | `evolution/types/config.py` |
| MAP-Elites orchestrator | `evolution/orchestrators/map_elites.py` |
| Population evaluation | `evolution/execution/evaluator.py` |
| LLM mutations | `evolution/execution/llm_strategies.py` |
| Genetic operators | `evolution/execution/strategies.py` |
| Archive | `evolution/map_elites/archive.py` |

---

## Evolution Flow

```
1. Initialize (GenomeFactory creates random population)
   └─> PopulationEvaluator evaluates in parallel
   └─> FitnessStrategy assigns fitness scores
   └─> Seed MapElitesArchive with initial population

2. For each generation:
   a. MapElitesLoop.step(archive)
      └─> Select random parents from archive
      └─> Apply crossover (optional)
      └─> Apply mutation (standard OR LLM-guided)
      └─> Return offspring batch

   b. Evaluate offspring (PopulationEvaluator)

   c. Archive insertion
      └─> Compute behavioral descriptors
      └─> Map to grid cell
      └─> Replace if better than current occupant

   d. Track global best, log stats, checkpoint

3. Return best genome
```

---

## LLM Integration

LLM mutations are enabled by:
1. Setting `config.llm.enabled = True`
2. OR using `--llm` CLI flag
3. OR using `mutation_strategy: llm_mutation` in preset

The LLM receives the genome's:
- Current parameters
- Performance metrics (quality, cost, latency)
- Strategy configuration

And returns proposed changes to improve performance.

Falls back to standard mutation if LLM fails.

---

## Removed Components (Simplification)

- StandardGAOrchestrator (MAP-Elites only now)
- EvolutionLoop (for GA, replaced by MapElitesLoop)
- Extensions (Niching, Immigration - diversity handled by archive)
- MultiArchiveManager (category-aware evolution removed)
- CategoryAwareEvaluator
- Legacy LLMOptimizer (use UniversalLLMProvider)
- 9 dead config fields

---

## GPU Acceleration (Implemented)

### Overview

GPU acceleration has been implemented to address the CPU bottleneck in retrieval evaluation (70-85% of runtime). The implementation replaces FAISS with **PyTorch-native GPU vector search** to avoid CPU-GPU data transfers.

### Implementation Status: COMPLETE

All planned GPU acceleration features have been implemented and tested.

### Files Created/Modified

| File | Status | Description |
|------|--------|-------------|
| `swarm_rag/utils/device.py` | **NEW** | GPU device utilities with auto-fallback |
| `swarm_rag/utils/__init__.py` | **NEW** | Package init (migrated from utils.py) |
| `swarm_rag/integrations/gpu_vector_store.py` | **NEW** | PyTorch GPU vector store |
| `swarm_rag/integrations/stark.py` | **MODIFIED** | Added `StarkGPUVectorStore`, `create_stark_vector_store()` |
| `swarm_rag/core/swarm_retriever.py` | **MODIFIED** | Added `use_gpu` param, GPU-aware methods |
| `swarm_rag/core/heuristics.py` | **MODIFIED** | GPU-aware dot product, tensor support |
| `swarm_rag/__init__.py` | **MODIFIED** | Exports `GPUVectorStore`, `get_device` |

### Quick Start

```python
from swarm_rag import GPUVectorStore, get_device, SwarmRetriever
from swarm_rag.integrations.stark import create_stark_vector_store, StarkGPUVectorStore

# Check device
print(get_device())  # "cuda" or "cpu"

# Create GPU-accelerated vector store (auto-detects GPU)
vector_store = create_stark_vector_store(doc_embs, use_gpu="auto")

# Or explicitly use GPU store
gpu_store = StarkGPUVectorStore(doc_embs, use_gpu=True)

# SwarmRetriever with GPU support
retriever = SwarmRetriever(
    vector_store=gpu_store,
    graph_store=graph,
    embedding_provider=embedder,
    use_gpu=True  # NEW parameter
)
print(retriever.is_gpu_enabled)  # True if GPU active
```

### Environment Variable Control

```bash
# Force CPU mode
export SWARM_RAG_DEVICE=cpu

# Force CUDA (fails if unavailable)
export SWARM_RAG_DEVICE=cuda

# Auto-detect (default)
export SWARM_RAG_DEVICE=auto
```

### API Reference

#### `swarm_rag/utils/device.py`

```python
get_device(force_cpu=False) -> str        # Returns "cuda" or "cpu"
get_array_module() -> module              # Returns cupy or numpy
ensure_tensor(data, device, dtype) -> Tensor
to_numpy(data) -> np.ndarray
clear_device_cache()                      # Reset cached detection
get_gpu_memory_info() -> dict             # Memory stats if GPU
```

#### `swarm_rag/integrations/gpu_vector_store.py`

```python
class GPUVectorStore(VectorStore):
    @classmethod
    def from_dict(doc_embs, device=None)  # Create from dict

    def search(query_vec, limit) -> List[Dict]
    def search_batch(query_vecs, limit) -> List[List[Dict]]
    def fetch(node_id) -> np.ndarray
    def fetch_batch(node_ids) -> np.ndarray
    def fetch_batch_gpu(node_ids) -> Tuple[Tensor, List[int]]
    def compute_similarities(query_vec, candidate_ids) -> Tuple[Tensor, List[int]]
```

#### `swarm_rag/integrations/stark.py` (new additions)

```python
class StarkGPUVectorStore(VectorStore):
    def __init__(doc_embs, use_gpu=True, device=None)
    # Wraps GPUVectorStore with FAISS fallback

def create_stark_vector_store(doc_embs, use_gpu="auto", shared_name=None)
    # Factory: use_gpu = "auto" | "always" | "never"
```

### Testing the Implementation

```python
# Test 1: Device detection
from swarm_rag.utils.device import get_device, get_array_module
print(f"Device: {get_device()}")
print(f"Array module: {get_array_module().__name__}")

# Test 2: GPU vector store
import torch
from swarm_rag.integrations.gpu_vector_store import GPUVectorStore

embeddings = torch.randn(1000, 768)
ids = list(range(1000))
store = GPUVectorStore(embeddings, ids)
results = store.search(torch.randn(768), limit=10)
print(f"Top result: {results[0]}")

# Test 3: Heuristics with torch tensors
from swarm_rag.core.heuristics import Heuristics, HeuristicContext
import numpy as np

ctx = HeuristicContext(
    query_vec=torch.tensor([1.0, 0.0, 0.0]),
    target_vecs=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    target_ids=[0, 1]
)
scores = Heuristics.semantic_similarity(ctx)
print(f"Scores (torch): {scores}")  # tensor([1.0, 0.5])

# Test 4: Full integration with STaRK
from swarm_rag.integrations.stark import create_stark_vector_store
# store = create_stark_vector_store(doc_embs, use_gpu="auto")
```

### Backward Compatibility

- All existing APIs unchanged
- `SWARM_RAG_DEVICE=cpu` forces CPU mode
- Auto-fallback: GPU unavailable → CPU + FAISS
- Heuristics work with both numpy arrays and torch tensors
- `SwarmRetriever` default `use_gpu=True` auto-detects

### venv Management for GPU Packages

```powershell
# Fresh GPU environment setup
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip wheel setuptools

# Step 1: PyTorch CUDA first (must match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Step 2: CuPy (must match CUDA version)
pip install cupy-cuda12x

# Step 3: PyTorch-dependent packages
pip install torch-geometric sentence-transformers

# Step 4: Rest of dependencies
pip install -r requirements.txt
```

### Expected Speedup

| Component | Speedup |
|-----------|---------|
| Embedding generation | 10-20x |
| Vector search (PyTorch GPU) | 12-36x |
| Heuristic calculations | 5-10x |
| **Overall** | **4-7x** |

### Next Steps - COMPLETED

All next steps have been implemented:

1. **Benchmark** - DONE: `swarm_rag/utils/benchmark.py`
2. **Batch optimization** - DONE: `SwarmRetriever.retrieve_batch_optimized()`
3. **Memory profiling** - DONE: `swarm_rag/utils/memory.py`
4. **CuPy integration** - DONE: `swarm_rag/utils/device.py` (cupy_* functions)

---

## Benchmarking Utilities

### Quick Start

```python
from swarm_rag.utils import (
    Benchmarker, benchmark_vector_search,
    run_all_benchmarks, print_benchmark_summary
)

# Simple benchmark
benchmarker = Benchmarker(warmup_iterations=3, n_iterations=10)
result = benchmarker.run("my_operation", lambda: my_function())
print(result)  # BenchmarkResult with timing stats

# CPU vs GPU comparison
comparison = benchmarker.compare(
    "vector_search",
    cpu_func=lambda: cpu_store.search(query, 100),
    gpu_func=lambda: gpu_store.search(query, 100)
)
print(f"GPU speedup: {comparison.speedup}x")

# Run all benchmarks
results = run_all_benchmarks(n_docs=50000)
print_benchmark_summary(results)
```

### Available Functions

```python
# Benchmarker class
Benchmarker(warmup_iterations=3, n_iterations=10, sync_cuda=True)
  .run(name, func, device, n_ops) -> BenchmarkResult
  .compare(name, cpu_func, gpu_func) -> ComparisonResult

# Pre-built benchmarks
benchmark_vector_search(n_docs, dim, n_queries, top_k) -> ComparisonResult
benchmark_batch_similarity(n_candidates, dim, batch_sizes) -> List[ComparisonResult]
benchmark_heuristics(n_candidates, dim) -> List[ComparisonResult]
run_all_benchmarks(n_docs, dim, verbose) -> Dict
```

---

## Memory Profiling

### Quick Start

```python
from swarm_rag.utils import (
    MemoryProfiler, memory_guard, estimate_tensor_memory,
    get_gpu_memory_info, clear_gpu_cache
)

# Basic profiling
profiler = MemoryProfiler()
profiler.snapshot("start")
# ... do work ...
profiler.snapshot("end")
profiler.print_report()

# Context manager
with profiler.track("my_operation"):
    result = expensive_operation()

# Decorator
@profiler.profile
def my_function():
    return process_data()

# Memory guard with limit
with memory_guard(max_gpu_mb=4000) as profiler:
    train_model()
print(f"Peak GPU: {profiler.get_peak_gpu_memory()} MB")

# Estimate memory before allocation
mem_mb = estimate_tensor_memory((100000, 768), dtype=np.float32)
print(f"Tensor will use {mem_mb:.1f} MB")
```

### Available Functions

```python
# MemoryProfiler class
MemoryProfiler(track_cpu=True, track_gpu=True)
  .snapshot(label) -> MemorySnapshot
  .track(label) -> context manager
  .profile(func) -> decorator
  .get_peak_gpu_memory() -> float (MB)
  .get_peak_cpu_memory() -> float (MB)
  .print_report()

# Utilities
get_gpu_memory_info() -> Dict[str, float]  # allocated_mb, cached_mb, total_mb
clear_gpu_cache()                           # Empty GPU memory cache
memory_guard(max_gpu_mb, cleanup) -> context manager
estimate_tensor_memory(shape, dtype) -> float (MB)
```

---

## CuPy Integration

### Quick Start

```python
from swarm_rag.utils import (
    is_cupy_available, get_array_module,
    to_cupy, cupy_to_numpy,
    cupy_cosine_similarity, cupy_topk
)

# Check availability
if is_cupy_available():
    print("CuPy is available!")

# Device-agnostic code
xp = get_array_module()  # Returns cupy or numpy
arr = xp.array([1, 2, 3])
result = xp.sum(arr)

# Cosine similarity (uses GPU when available)
scores = cupy_cosine_similarity(query_vec, candidate_matrix)

# Top-k selection
top_scores, top_indices = cupy_topk(scores, k=10)
```

### Available Functions

```python
# Device-agnostic array operations
get_array_module() -> module          # cupy or numpy
is_cupy_available() -> bool
to_cupy(data) -> cupy/numpy array
cupy_to_numpy(data) -> numpy array

# GPU-accelerated operations (fall back to numpy if no GPU)
cupy_matmul(a, b)                     # Matrix multiplication
cupy_dot(a, b)                        # Dot product
cupy_norm(a, axis, keepdims)          # L2 norm
cupy_normalize(a, axis, eps)          # L2 normalize
cupy_cosine_similarity(query, candidates) -> scores
cupy_topk(scores, k) -> (top_scores, top_indices)
sync_device()                         # Synchronize CUDA
```

---

## Batch Optimization in SwarmRetriever

### New Method: `retrieve_batch_optimized()`

Provides GPU-accelerated batch retrieval with:
- Batch initial searches (single GPU operation for all queries)
- Vectorized ranking
- Better GPU memory utilization

```python
retriever = SwarmRetriever(
    vector_store=gpu_store,
    graph_store=graph,
    embedding_provider=embedder,
    use_gpu=True
)

# Standard batch retrieval
results = retriever.retrieve_batch(queries, max_workers=4)

# Optimized batch retrieval (GPU-accelerated)
results = retriever.retrieve_batch_optimized(
    queries,
    use_vectorized_ranking=True  # Use GPU-optimized ranking
)
```

### New Internal Methods

```python
# GPU-aware vector fetching
_fetch_vectors_batch_gpu(node_ids) -> (tensor/array, valid_ids)

# Vectorized ranking (uses GPU similarity computation)
_ranking_vectorized(trajectories, query_vec, ranking_func, top_k, n_agents)

# Batch initial search
_batch_initial_search(query_vecs, pool_size) -> List[List[int]]

# Batch similarity computation
_compute_batch_similarities_gpu(query_vecs, candidate_ids_per_query)
```

---

## Test Coverage

All new features are tested in `tests/unit/test_gpu_utils.py`:

```bash
pytest swarm_rag_module/tests/unit/test_gpu_utils.py -v

# Tests include:
# - Device utilities (6 tests)
# - CuPy integration (8 tests)
# - Benchmarker (4 tests)
# - MemoryProfiler (7 tests)
# - Batch optimization (2 tests)
# - GPU-aware heuristics (3 tests)
# - GPUVectorStore (4 tests)
# Total: 34 tests
```

---

## Dataclass Config Migration (January 2026)

### Overview

Fixed runtime errors caused by dict-style config access after migration to dataclass-based configuration. All genetic strategies and LLM utilities now properly use dataclass attribute access.

### Files Modified

| File | Changes |
|------|---------|
| `evolution/execution/strategies.py` | Fixed 5 functions with dict-style access |
| `evolution/llm/utils.py` | Fixed HeuristicKey enum to string conversion |

### Specific Fixes

#### `strategies.py` - Genetic Operators

1. **`_mutate_params_standard()`**:
   - `ctx.config.get('swarmrag_param_ranges', {})` → `ctx.config.genetic.param_ranges`
   - `if key in ranges` → `if hasattr(ranges, key)`
   - `ranges[key]` → `getattr(ranges, key)`

2. **`_randomize_all_params()`**: Same pattern as above

3. **`_randomize_ratios()`**: Removed dict access, uses hardcoded default `(0.1, 1.0)` for group ratios

4. **`boltzmann_selection()`**: All Boltzmann config access updated:
   - `ctx.config.get('boltzmann_temperature', 1.0)` → `ctx.config.genetic.boltzmann.temperature`
   - `ctx.config.get('boltzmann_adaptive', True)` → `ctx.config.genetic.boltzmann.adaptive`
   - `ctx.config.get('boltzmann_alpha', 0.95)` → `ctx.config.genetic.boltzmann.alpha`
   - `ctx.config.get('boltzmann_min_temp', 0.1)` → `ctx.config.genetic.boltzmann.min_temp`
   - `ctx.config.get('boltzmann_max_temp', 5.0)` → `ctx.config.genetic.boltzmann.max_temp`
   - `ctx.config.get('boltzmann_diversity_threshold', 0.05)` → `ctx.config.genetic.boltzmann.diversity_threshold`

5. **`aggressive_mutation()`**: Same param_ranges fixes as `_mutate_params_standard()`

#### `llm/utils.py` - LLM Context Building

**`_get_available_heuristics()`**: Fixed HeuristicKey enum serialization error:
```python
# Before (error: "sequence item 0: expected str instance, HeuristicKey found")
return {
    "movement": list(HeuristicRegistry.all_movement().keys()),
    ...
}

# After (works correctly)
return {
    "movement": [str(k) for k in HeuristicRegistry.all_movement().keys()],
    ...
}
```

### Config Access Patterns

When accessing config in genetic strategies, use these patterns:

```python
# Genetic config
ctx.config.genetic.mutation_strategy
ctx.config.genetic.crossover_strategy
ctx.config.genetic.base_mutation_rate
ctx.config.genetic.selection_k
ctx.config.genetic.expr_max_depth
ctx.config.genetic.n_agent_groups

# Boltzmann selection config
ctx.config.genetic.boltzmann.temperature
ctx.config.genetic.boltzmann.adaptive
ctx.config.genetic.boltzmann.alpha
ctx.config.genetic.boltzmann.min_temp
ctx.config.genetic.boltzmann.max_temp

# Parameter ranges (SwarmParamRanges dataclass)
ranges = ctx.config.genetic.param_ranges
if hasattr(ranges, key):
    min_v, max_v = getattr(ranges, key)

# MAP-Elites config
ctx.config.map_elites.batch_size
ctx.config.map_elites.bins
ctx.config.map_elites.initial_fill

# Top-level config
ctx.config.n_generations
ctx.config.fitness_strategy
```

### Verification

Tested with: `python stark/evolve_stark.py --preset toy --llm --gpu auto --scratch`

Results:
- Evolution completed successfully (3 generations)
- LLM mutations working (Cerebras API calls successful)
- MAP-Elites coverage increased from 5.56% to 9.44%
- No runtime errors
