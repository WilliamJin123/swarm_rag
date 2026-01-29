# Codebase Structure

**Analysis Date:** 2026-01-29

## Directory Layout

```
swarm_rag_module/
├── swarm_rag/                          # Main package
│   ├── core/                           # Multi-agent retrieval engine
│   │   ├── __init__.py
│   │   ├── swarm_retriever.py          # Core SwarmRetriever class (1000+ LOC)
│   │   └── heuristics.py               # Movement/ranking/deposit registries
│   ├── eval/                           # Metrics & evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py                  # Evaluator class + Metrics TypedDict
│   │   ├── metric_functions.py         # Hit@K, Recall@K, MRR functions
│   │   └── report.py                   # Result aggregation & reporting
│   ├── evolution/                      # Evolutionary optimization
│   │   ├── engine.py                   # EvolutionEngine entry point
│   │   ├── engine.py
│   │   ├── types/                      # Type definitions
│   │   │   ├── genome.py               # Genome dataclass
│   │   │   ├── config.py               # EvolutionConfig, MapElitesConfig
│   │   │   ├── expressions.py          # ExpressionNode for symbolic trees
│   │   │   └── fitness_results.py      # FitnessResult TypedDict
│   │   ├── execution/                  # Evolutionary operations
│   │   │   ├── evaluator.py            # PopulationEvaluator
│   │   │   ├── factory.py              # GenomeFactory (creation, mutation)
│   │   │   ├── fitness.py              # FitnessCalculator
│   │   │   ├── fitness_strategies.py   # Lexicographic, Pareto strategies
│   │   │   ├── strategies.py           # Mutation/crossover implementations
│   │   │   ├── llm_strategies.py       # LLM-guided mutations
│   │   │   ├── tracker.py              # ProgressTracker
│   │   │   ├── profiler.py             # GenerationProfiler
│   │   │   └── weighted_sum.py         # Weighted sum genome execution
│   │   ├── map_elites/                 # Quality-Diversity algorithm
│   │   │   ├── archive.py              # MapElitesArchive (elite storage)
│   │   │   ├── loop.py                 # MapElitesLoop (evolution loop)
│   │   │   └── descriptors/            # Behavioral feature calculators
│   │   │       ├── base.py
│   │   │       ├── builtin.py          # Fragmentation, coverage, etc.
│   │   │       └── registry.py
│   │   ├── llm/                        # LLM-guided mutations
│   │   │   ├── client.py               # LLM API client wrapper
│   │   │   ├── bridge.py               # Genome ↔ LLM serialization
│   │   │   ├── expression_builder.py   # Symbolic tree generation
│   │   │   ├── decision_tracker.py     # Mutation outcome tracking
│   │   │   ├── evolution_journal.py    # Mutation history log
│   │   │   ├── strategic_oracle.py     # High-level strategy advice
│   │   │   ├── tactical_advisor.py     # Mutation-level suggestions
│   │   │   └── intents.py              # Intent parsing from LLM
│   │   ├── orchestrators/              # Evolution orchestrators
│   │   │   ├── base.py                 # BaseOrchestrator interface
│   │   │   └── map_elites.py           # MAPElitesOrchestrator
│   │   ├── adapters/                   # Adapter layer for SwarmRetriever
│   │   │   └── swarm_adapter.py        # SwarmRetriever wrapper
│   │   ├── storage/                    # Checkpointing & logging
│   │   │   └── run_manager.py          # RunManager (results, checkpoints)
│   │   ├── seed_genomes.py             # Initial population seeding
│   │   └── focused_mutation.py         # Focused mutation strategies
│   ├── integrations/                   # Backend implementations
│   │   ├── __init__.py
│   │   ├── stark.py                    # STaRK graph/vector integration
│   │   ├── torch_vector_store.py       # In-memory tensor vector store
│   │   ├── torch_graph_store.py        # NetworkX-backed graph store
│   │   ├── cohere_embed.py             # Cohere embedding API
│   │   └── gemini_embed.py             # Gemini embedding API
│   ├── interfaces/                     # Abstract contracts
│   │   ├── __init__.py
│   │   ├── abstract_classes.py         # VectorStore, GraphStore, EmbeddingProvider
│   │   ├── retriever_types.py          # SingleResult, BatchResult, RetrievalConfig
│   │   ├── enums.py                    # HeuristicKey, GeneticKey enums
│   │   ├── types.py                    # AgentGroupConfig
│   │   ├── evaluable.py                # Evaluable protocol
│   │   ├── protocols.py                # Other protocols
│   │   └── registry.py                 # _MovementRegistry, _RankingRegistry, _DepositRegistry
│   ├── utils/                          # Utilities
│   │   ├── __init__.py
│   │   ├── device.py                   # get_device(), move_to_device()
│   │   ├── memory.py                   # Memory profiling utilities
│   │   └── benchmark.py                # Benchmarking tools
│   └── __init__.py                     # Package exports
├── agent/                              # Agent integration
│   └── swarm_rag_agent.py              # (currently minimal)
├── tests/                              # Test suite
│   ├── conftest.py                     # Pytest fixtures (ToyStochasticRetriever, etc.)
│   ├── core/                           # Core layer tests
│   ├── evolution/                      # Evolution layer tests
│   │   ├── test_genome.py
│   │   ├── test_evaluator.py
│   │   ├── test_integration.py
│   │   └── test_weighted_sum.py
│   ├── unit/                           # Unit tests
│   ├── integration/                    # Integration tests
│   └── bugs/                           # Bug reproduction tests
│       ├── reproduce_compiler_bug.py
│       ├── reproduce_fitness.py
│       └── reproduce_ratio_mutation_bug.py
└── pyproject.toml                      # Package metadata
```

## Directory Purposes

**`core/`:**
- Purpose: Multi-agent swarm retrieval with heuristics-based exploration
- Contains: Core SwarmRetriever, heuristics, profiling
- Key files: `swarm_retriever.py` (main), `heuristics.py` (registries)

**`eval/`:**
- Purpose: Compute retrieval quality metrics
- Contains: Metrics calculator, metric functions, result aggregation
- Key files: `metrics.py` (Evaluator), `metric_functions.py` (MRR, Hit@K, etc.)

**`evolution/`:**
- Purpose: Evolutionary optimization infrastructure
- Contains: Multiple subdirectories for genome, execution, optimization algorithm
- Key patterns: GenomeFactory creates/mutates genomes; PopulationEvaluator batches evaluation; MAPElitesOrchestrator runs optimization loop

**`evolution/types/`:**
- Purpose: Type definitions for evolution system
- Contains: Genome dataclass, EvolutionConfig nested dataclasses, WeightTensors, ExpressionNode
- Pattern: Strongly-typed configuration system replacing flat TypedDicts

**`evolution/execution/`:**
- Purpose: Evolutionary operations (mutation, crossover, fitness, evaluation)
- Contains: GenomeFactory, Mutation/crossover strategies, PopulationEvaluator, FitnessCalculator
- Key files: `factory.py` (genome creation/mutation), `evaluator.py` (batch evaluation)

**`evolution/map_elites/`:**
- Purpose: MAP-Elites quality-diversity optimization algorithm
- Contains: Archive (elite storage), Loop (breeding logic), Descriptor calculators
- Key files: `archive.py` (elite population), `loop.py` (generation loop)

**`evolution/llm/`:**
- Purpose: LLM-guided mutation and optimization
- Contains: LLM client, serialization bridges, mutation decision tracking
- Key files: `client.py` (LLM API), `bridge.py` (Genome serialization)

**`evolution/orchestrators/`:**
- Purpose: High-level evolution orchestration
- Contains: Base interface, MAP-Elites orchestrator
- Key files: `map_elites.py` (runs full optimization loop)

**`evolution/storage/`:**
- Purpose: Checkpoint, log, and result persistence
- Contains: RunManager for file I/O
- Key files: `run_manager.py` (checkpoint/results management)

**`integrations/`:**
- Purpose: Concrete implementations of VectorStore, GraphStore, EmbeddingProvider
- Contains: STaRK, Torch, embedding provider implementations
- Key files: `stark.py` (STaRK backend), `torch_vector_store.py`, `torch_graph_store.py`

**`interfaces/`:**
- Purpose: Abstract contracts and type definitions
- Contains: ABC classes, TypedDicts, enums, registries
- Key files: `abstract_classes.py` (VectorStore, GraphStore, EmbeddingProvider), `enums.py` (HeuristicKey)

**`utils/`:**
- Purpose: Cross-cutting utilities
- Contains: Device management, memory profiling, benchmarking
- Key files: `device.py` (GPU/CPU detection), `memory.py` (profiling)

**`tests/`:**
- Purpose: Test suite covering all layers
- Contains: Unit, integration, bug reproduction tests
- Key files: `conftest.py` (shared fixtures), subdirectories by layer

## Key File Locations

**Entry Points:**
- `swarm_rag_module/swarm_rag/__init__.py`: Package exports (SwarmRetriever, VectorStore, GraphStore)
- `swarm_rag_module/swarm_rag/core/swarm_retriever.py:SwarmRetriever.retrieve()`: Query execution
- `swarm_rag_module/swarm_rag/evolution/engine.py:EvolutionEngine.optimize()`: Optimization entry point

**Configuration:**
- `swarm_rag_module/swarm_rag/evolution/types/config.py`: EvolutionConfig, MapElitesConfig
- `swarm_rag_module/swarm_rag/interfaces/retriever_types.py`: RetrievalConfig, RunConfig

**Core Logic:**
- `swarm_rag_module/swarm_rag/core/swarm_retriever.py`: Multi-agent traversal
- `swarm_rag_module/swarm_rag/core/heuristics.py`: Strategy registries
- `swarm_rag_module/swarm_rag/evolution/types/genome.py`: Strategy encapsulation

**Testing:**
- `swarm_rag_module/tests/conftest.py`: Shared fixtures (ToyStochasticRetriever, evaluators)
- `swarm_rag_module/tests/evolution/test_integration.py`: End-to-end tests
- `swarm_rag_module/tests/bugs/`: Bug reproduction scripts

## Naming Conventions

**Files:**
- `*.py`: All Python modules
- Test files: `test_*.py` or `reproduce_*.py` for bug reproduction
- Integration modules: Named after backend (e.g., `stark.py`, `torch_vector_store.py`)

**Directories:**
- Lowercase underscore-separated: `core`, `evolution`, `map_elites`
- Layer-based organization: `core`, `eval`, `evolution`, `integrations`, `interfaces`, `utils`

**Classes:**
- PascalCase: `SwarmRetriever`, `Genome`, `GenomeFactory`, `MapElitesArchive`
- Abstract bases prefix with virtual nature: `VectorStore`, `GraphStore`, `EmbeddingProvider`

**Functions:**
- snake_case: `calculate_metrics()`, `get_device()`, `search_batch()`
- Private functions prefixed with `_`: `_compute_agent_scores()`, `_create_default_weight_tensors()`

**Type Definitions:**
- TypedDict: PascalCase: `RetrievalConfig`, `RunConfig`, `Metrics`
- Enum: PascalCase: `HeuristicKey`, `GeneticKey`
- Dataclass: PascalCase: `Genome`, `TraversalState`, `WeightTensors`

## Where to Add New Code

**New Retrieval Heuristic:**
- Add function to appropriate module in `swarm_rag/core/` or new module
- Register with `HeuristicRegistry.register_movement()`, `.register_ranking()`, or `.register_deposit()`
- Example location: New heuristic in `swarm_rag/core/custom_heuristics.py`, registered at module load

**New VectorStore Backend:**
- Create new file in `swarm_rag/integrations/` (e.g., `chroma_store.py`)
- Inherit from `VectorStore` abstract class
- Implement: `search()`, `search_batch()`, `fetch()`, `fetch_batch()`, `device` property
- Add to `swarm_rag/__init__.py` exports if needed

**New Graph Store Backend:**
- Create new file in `swarm_rag/integrations/`
- Inherit from `GraphStore` abstract class
- Implement: `neighbors()`, `neighbors_batch()`, `get_degree()`, `get_avg_degree()`, `n_nodes` property
- Add to `swarm_rag/__init__.py` exports if needed

**New Metric:**
- Add calculation function to `swarm_rag/eval/metric_functions.py`
- Add field to `Metrics` TypedDict in `swarm_rag/eval/metrics.py`
- Add computation logic in `Evaluator.calculate_metrics()`
- Use in fitness calculator weights via `FitnessCalculator(weights={...})`

**New Evolutionary Strategy:**
- For mutation/crossover: Add class to `swarm_rag/evolution/execution/strategies.py`
- For fitness assignment: Create class inheriting from `FitnessStrategy` in `fitness_strategies.py`
- For descriptors: Add to `swarm_rag/evolution/map_elites/descriptors/builtin.py`
- Register in appropriate factory (GenomeFactory for mutations, DescriptorRegistry for descriptors)

**New Test:**
- Unit test: `swarm_rag_module/tests/unit/test_*.py`
- Integration test: `swarm_rag_module/tests/integration/test_*.py`
- Bug reproduction: `swarm_rag_module/tests/bugs/reproduce_*.py`
- Use fixtures from `conftest.py` (ToyStochasticRetriever, test configs)

**New Utility Function:**
- Device-related: `swarm_rag/utils/device.py`
- Memory-related: `swarm_rag/utils/memory.py`
- Benchmarking: `swarm_rag/utils/benchmark.py`

## Special Directories

**`evolution/types/`:**
- Purpose: Centralized type definitions for evolution system
- Generated: No
- Committed: Yes
- Imports: Used throughout evolution layer, imported at module boundaries

**`tests/`:**
- Purpose: Test suite
- Generated: No (test data files like evo_results, evo_results_map are generated outputs)
- Committed: Yes (test files), No (test output directories)
- Structure: Mirrors source layer structure (core, evolution, etc.)

**`.pytest_cache/`:**
- Purpose: pytest cache
- Generated: Yes
- Committed: No
- Created by: pytest automatically

---

*Structure analysis: 2026-01-29*
