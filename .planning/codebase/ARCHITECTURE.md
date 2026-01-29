# Architecture

**Analysis Date:** 2026-01-29

## Pattern Overview

**Overall:** Hierarchical modular architecture with three interconnected layers:
1. **Core Retrieval Layer** - SwarmRetriever implements ant-colony-inspired multi-agent search
2. **Evolution Layer** - MAP-Elites quality-diversity optimization with optional LLM guidance
3. **Integration Layer** - Pluggable vector/graph stores and embedding providers

**Key Characteristics:**
- Plugin-based abstraction for storage backends (VectorStore, GraphStore, EmbeddingProvider)
- Tensor-first design for GPU acceleration (PyTorch primitives throughout)
- Dual-mode genome evolution: linear (weighted_sum) and symbolic (expression_tree)
- Progressive batching: sequential → single-batch → multi-batch retrieval modes
- Profiler-instrumented hot paths with torch.compile() support

## Layers

**Core Retrieval Layer:**
- Purpose: Execute multi-agent swarm traversal over knowledge graphs with pheromone-based exploration
- Location: `swarm_rag_module/swarm_rag/core/`
- Contains: SwarmRetriever (main orchestrator), Heuristics registry, step profilers
- Depends on: VectorStore, GraphStore, EmbeddingProvider interfaces
- Used by: EvolutionEngine, agents during fitness evaluation

**Evolution Layer:**
- Purpose: Evolve diverse population of retrieval strategies using MAP-Elites with optional LLM mutations
- Location: `swarm_rag_module/swarm_rag/evolution/`
- Contains: EvolutionEngine, Genome types, MAP-Elites archive/loop, LLM bridges, execution strategies
- Depends on: SwarmRetriever, Evaluator, fitness calculators
- Used by: Application scripts, benchmarks, optimization pipelines

**Integration Layer:**
- Purpose: Provide concrete implementations of storage abstractions
- Location: `swarm_rag_module/swarm_rag/integrations/`
- Contains: STaRK adapter, TorchVectorStore, TorchGraphStore, embedding providers (Cohere, Gemini)
- Depends on: External APIs (Cohere, Gemini), PyTorch tensors
- Used by: Core and Evolution layers via abstract interfaces

**Evaluation Layer:**
- Purpose: Calculate retrieval quality metrics for genomes
- Location: `swarm_rag_module/swarm_rag/eval/`
- Contains: Metrics calculator, metric functions (MRR, Hit@K, Recall@K, diversity)
- Depends on: Retrieved results, ground truth data
- Used by: Fitness calculator, evolution loop for fitness assignment

**Interfaces & Types:**
- Purpose: Define contracts and data structures across all layers
- Location: `swarm_rag_module/swarm_rag/interfaces/`, `swarm_rag_module/swarm_rag/evolution/types/`
- Contains: Abstract base classes (VectorStore, GraphStore, EmbeddingProvider), TypedDicts, enums
- Depends on: PyTorch, typing utilities
- Used by: All other layers

**Utilities:**
- Purpose: Device management, memory utilities, benchmarking
- Location: `swarm_rag_module/swarm_rag/utils/`
- Contains: device detection, LRU caching, memory profiling
- Depends on: PyTorch, system libraries
- Used by: Core and Evolution layers

## Data Flow

**Retrieval (Query → Results):**

1. Query embedding created via `embedding_provider.embed(query)`
2. Initial pool fetched via `vector_store.search(query_embedding, initial_pool_size)`
3. Agents spawn at high-ranked nodes in initial pool
4. Per-step: agents compute movement scores via heuristics, traverse graph neighbors, deposit pheromones
5. Results ranked by pheromone + semantic score combination
6. Top-k results returned as `SingleResult(node_ids, scores)` tensors

**Batch Retrieval (Multiple Queries):**

Queries chunked into batch_size (default 64). Per chunk:
1. Embed all queries at once
2. Initialize TraversalState with batched tensors
3. Loop `steps` iterations, updating state in parallel
4. Combine final results into `BatchResult(n_queries, top_k)` tensors

**Evolution (Genome → Fitness → Archive):**

1. Initialize population: seed_genomes or random genomes (GenomeFactory)
2. Per generation:
   a. Select parents from archive (MAP-Elites selection)
   b. Apply mutation/crossover to create offspring
   c. Evaluate offspring via `evaluator.evaluate_batch(genomes, train_queries, ground_truth)`
   d. Compute fitness via `fitness_calculator.calculate(metrics)` → scalar fitness
   e. Calculate behavioral descriptors via DescriptorCalculator
   f. Attempt to add to archive (if better than cell occupant)
   g. Save checkpoint, log progress

**State Management:**

- **Traversal State:** `TraversalState` dataclass holds current position of all agents across all queries during a batch
  - `agent_positions` (n_queries, n_agents) - current node indices
  - `pheromones` (sparse) - per-query pheromone maps
  - `visited_counts` - track exploration coverage
  - Updated in-place per step to minimize allocations

- **Genome State:** `Genome` dataclass stores complete strategy
  - `params` - SwarmParams (n_agents, steps, decay, etc.)
  - `weights` - WeightTensors for weighted_sum mode or ExpressionNode trees for expression_tree mode
  - Immutable during evaluation, cloned for mutation

- **Archive State:** `MapElitesArchive` maintains elite population
  - Bins indexed by behavioral descriptors
  - Best genome per bin stored with fitness
  - Updated after each generation

## Key Abstractions

**VectorStore:**
- Purpose: Abstract over vector database implementations
- Examples: `TorchVectorStore` (in-memory), `StarkVectorStore` (STaRK backend)
- Pattern: Interface defined in `interfaces/abstract_classes.py`, implementations in `integrations/`
- Methods: `search()`, `search_batch()`, `fetch()`, `fetch_batch()`
- Returns: Tensors on configured device (CPU or GPU)

**GraphStore:**
- Purpose: Abstract over graph structure access
- Examples: `TorchGraphStore` (NetworkX wrapper), `StarkGraphAdapter` (STaRK backend)
- Pattern: Interface defined in `interfaces/abstract_classes.py`
- Methods: `neighbors()`, `neighbors_batch()`, `get_degree()`, `get_avg_degree()`
- Returns: Tensors with neighbor indices and degrees

**EmbeddingProvider:**
- Purpose: Embed text into vectors
- Examples: Cohere, Gemini, local models
- Pattern: Interface in `interfaces/abstract_classes.py`, implementations in `integrations/`
- Method: `embed(text) → tensor`

**Heuristics Registry:**
- Purpose: Pluggable movement/ranking/deposit strategies
- Examples: semantic_similarity, node_centrality, pheromone_repulsion
- Pattern: Three registries (movement, ranking, deposit) in `core/heuristics.py`
- Usage: Strategies selected and composed into strategies dict in genome

**Genome:**
- Purpose: Complete retrieval strategy encapsulation
- Pattern: Dataclass in `evolution/types/genome.py`
- Contains: params (SwarmParams), weights (WeightTensors), mode (weighted_sum vs expression_tree)
- Immutable during evaluation; mutations create new copies

**Evolutionary Strategies:**
- **Expression Trees:** `ExpressionNode` for symbolic computation (flexible, evolves structure)
- **Weighted Sum:** `WeightTensors` with GPU-optimized linear combinations (fast, fixed structure)
- **Fitness:** Lexicographic or Pareto-based assignment
- **Descriptors:** Behavioral features for MAP-Elites binning (fragmentation, coverage, etc.)

## Entry Points

**SwarmRetriever Retrieval:**
- Location: `swarm_rag_module/swarm_rag/core/swarm_retriever.py:SwarmRetriever.retrieve()` and `.retrieve_batch()`
- Triggers: Called by application code or evaluator during evolution
- Responsibilities: Orchestrate multi-agent traversal, combine heuristics, return ranked results

**EvolutionEngine Optimization:**
- Location: `swarm_rag_module/swarm_rag/evolution/engine.py:EvolutionEngine.optimize()`
- Triggers: Application initializes engine, calls optimize()
- Responsibilities: Setup MAP-Elites loop, track progress, save checkpoints, return best genome

**Evaluator Metrics Calculation:**
- Location: `swarm_rag_module/swarm_rag/eval/metrics.py:Evaluator.calculate_metrics()`
- Triggers: PopulationEvaluator calls per genome per query
- Responsibilities: Compute MRR, Hit@K, Recall@K, diversity metrics for single query

**MAP-Elites Orchestrator:**
- Location: `swarm_rag_module/swarm_rag/evolution/orchestrators/map_elites.py:MAPElitesOrchestrator.run()`
- Triggers: EvolutionEngine creates and runs orchestrator
- Responsibilities: Parent selection, offspring creation, fitness assignment, archive updates

## Error Handling

**Strategy:** Try-except wrapping with fallback to safe defaults; validation at type boundaries.

**Patterns:**

- **Tensor Validation:** torch.isfinite() checks after score computation; NaN neighbors masked out
- **Graph Traversal:** Invalid neighbor IDs caught and skipped (pheromone_buffer_size headroom prevents out-of-bounds)
- **Evolution:** Mutation failures logged but not fatal; genome reverts to valid parent
- **Device Errors:** GPU failures caught, fallback to CPU with warning
- **Cache Misses:** LRU cache returns None, caller handles with direct fetch
- **Type Conversion:** TypedDict fields validated at input; Genome creation uses factory pattern to enforce constraints

## Cross-Cutting Concerns

**Logging:** `logging` module throughout; SwarmRetriever logs device choice and compilation; EvolutionEngine logs generation progress, checkpoints, best genomes.

**Validation:**
- Tensor shape checks in retrieval hot paths (assertions disabled in production)
- Genome constraint validation in GenomeFactory (param ranges, mutation bounds)
- TypedDict strict typing for configs

**Authentication:** Integration layer handles API keys via environment variables (e.g., COHERE_API_KEY, GEMINI_API_KEY)

**Caching:**
- Vector embeddings cached in LRUCache (doc_cache, query_cache)
- Graph neighbors cached (neighbor_cache) with degree_cache
- Embedding precompute shared across evaluations (SharedPrecompute)
- Query embedding result cached per unique embedding

**Performance Profiling:**
- StepProfiler tracks hot sections (embedding, neighbor lookup, scoring)
- Enabled via SWARM_PROFILE=1 environment variable
- torch.compile() available for GPU scoring functions
- Batching instrumentation shows throughput gains

---

*Architecture analysis: 2026-01-29*
