# Coding Conventions

**Analysis Date:** 2026-01-29

## Naming Patterns

**Files:**
- Snake_case for module files: `engine.py`, `genome.py`, `test_genome.py`
- Test files prefixed with `test_`: `test_core_evo.py`, `test_tracker.py`, `test_fitness_strategies.py`
- Organized in hierarchical directories by domain: `evolution/`, `integrations/`, `interfaces/`

**Functions:**
- Snake_case: `calculate_metrics()`, `normalize_ratios()`, `create_random_genome()`
- Private functions prefixed with underscore: `_get_sort_key()`, `_config_from_weights()`
- Methods handling core logic named descriptively: `evaluate()`, `compile()`, `retrieve_batch()`

**Variables:**
- Snake_case throughout: `max_workers`, `ground_truth_ids`, `mutation_rate`
- Constants in UPPERCASE: `FIXED_PARAMS`, `DEFAULT_PARAMS`, `DEFAULT_EARLY_EXIT_THRESHOLD`
- Short meaningful names for loop variables: `g` for genome, `f` for fitness, `q` for query

**Types & Classes:**
- PascalCase for classes: `Genome`, `ExpressionNode`, `FitnessResult`, `PopulationEvaluator`
- TypedDict for type contracts: `RetrievedNode`, `Metrics`, `SwarmParams`, `CompiledStrategies`
- Protocol classes for interfaces: `RetrievalBackend`, `Evaluable` (marked with `@runtime_checkable`)
- Dataclass extensively used: `@dataclass` decorator for `Genome`, `FitnessResult`, `ExpressionNode`

## Code Style

**Formatting:**
- No enforced formatter (no `.black`, `.flake8`, or `.ruff` config detected)
- Style inferred from codebase: 4-space indentation, 80-120 character lines
- Multiline imports grouped logically

**Linting:**
- No enforced linting config detected (no `.pylintrc` or `.flake8` present)
- Type hints present but not universally enforced: `from typing import Dict, List, Optional, TypedDict`

## Import Organization

**Order:**
1. Python standard library: `os`, `random`, `logging`, `json`, `tempfile`, `shutil`
2. Third-party libraries: `torch`, `pandas`, `numpy`, `pytest`
3. Relative imports within package: `from ..interfaces.enums import GeneticKey`
4. Local module imports: `from .types.genome import Genome`

**Path Aliases:**
- No alias paths configured (no `@` paths or `jsconfig/tsconfig` equivalents)
- Uses relative imports exclusively: `from ..core.swarm_retriever import SwarmRetriever`
- Package structure allows direct access: `from swarm_rag.evolution.engine import EvolutionEngine`

## Error Handling

**Patterns:**
- Silent defaults with fallback values used in retriever: `kwargs.get('n_agents', 1)`, `features.get(self.value, 0.0)`
- Direct assertion for testing: `assert f1 > f2, "High quality should beat low quality"` in `test_core_evo.py`
- Exception handling not heavily featured; prevention via type contracts preferred
- Division by zero prevented with clipping: `torch.where(right_tensor == 0, 1e-8, right_tensor)`

**Type Checking:**
- Runtime type checking in critical paths: `isinstance(arg, torch.Tensor)`, `isinstance(other, Genome)`
- Protocol-based duck typing for pluggable components (`RetrievalBackend` protocol)

## Logging

**Framework:** Python standard `logging` module

**Patterns:**
- Module-level logger initialized: `logger = logging.getLogger(__name__)` in `engine.py`
- Info-level logging for major events: `logger.info(f"Using Fitness Strategy: {self.fitness_strategy.__class__.__name__}")`
- Logging kept minimal; focus on progress tracking via `ProgressTracker` class
- `ProgressTracker` writes JSONL for machine-readable logging: `tracker.log(generation=1, train_stats=train_stats)`

## Comments

**When to Comment:**
- Docstrings required for all public classes and methods
- Complex logic receives inline comments explaining intent
- Examples: "Prevent overflow - use scalar to avoid device mismatch" in expression evaluation
- Algorithm explanations above loops: "Wrap right if it has lower precedence OR equal precedence..."

**JSDoc/TSDoc:**
- Google-style docstrings used: Parameter descriptions with `Args:`, return descriptions with `Returns:`
- Example from `engine.py`:
  ```python
  """
  MAP-Elites based evolutionary optimization engine.

  Uses quality-diversity optimization to evolve a diverse population of
  specialized retrieval strategies, with optional LLM-guided mutations.

  Example:
      config = EvolutionConfig(...)
      engine = EvolutionEngine(...)
      best_genome = engine.optimize()
  """
  ```
- Complex types documented inline: `def evaluate(self, features: Dict[str, torch.Tensor]) -> Union[float, torch.Tensor]`

## Function Design

**Size:**
- Functions typically 5-30 lines for core logic
- Complex functions up to 50 lines (e.g., `calculate_metrics()`, expression tree evaluation)
- Prefer decomposition into helper methods over inline complexity

**Parameters:**
- Type hints required for all function parameters
- Default arguments for optional parameters: `k_values=[1, 5, 10, 20]` in `Evaluator.__init__`
- Keyword arguments used for flexible configuration: `**kwargs` in retriever batch methods
- No excessive positional arguments; use dataclasses for grouped parameters

**Return Values:**
- Single return type preferred
- Union types used when necessary: `Union[float, torch.Tensor]` for expression evaluation
- Dataclass instances returned for complex results: `FitnessResult`, `Genome`
- None used sparingly; explicit Optional typing when used

## Module Design

**Exports:**
- Public classes and functions exposed at module `__init__.py` level
- Example: `swarm_rag/evolution/__init__.py` exports `EvolutionEngine`, `EvolutionConfig`
- Private modules use underscore prefix: `_compiled_cache` is internal field in `Genome`

**Barrel Files:**
- Used selectively: `swarm_rag/evolution/__init__.py` aggregates key exports
- Each `__init__.py` documents what it exposes via import statements
- Avoids circular imports by separating interfaces from implementations

---

*Convention analysis: 2026-01-29*
