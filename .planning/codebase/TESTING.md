# Testing Patterns

**Analysis Date:** 2026-01-29

## Test Framework

**Runner:**
- pytest 9.0.2
- Config: `C:\Users\jinwi\programming_files_NEW\swarm_rag_experiment\swarm_rag_module\pyproject.toml` specifies test discovery
- No pytest.ini found; uses default discovery patterns

**Assertion Library:**
- Python's built-in `assert` statements
- No external assertion library (pytest built-in assertions sufficient)

**Run Commands:**
```bash
pytest swarm_rag_module/tests/                         # Run all tests
pytest swarm_rag_module/tests/ -v                      # Verbose output
pytest swarm_rag_module/tests/unit/                    # Run unit tests only
pytest swarm_rag_module/tests/ -k "test_fitness"      # Run tests matching pattern
pytest swarm_rag_module/tests/ --tb=short             # Short traceback output
```

## Test File Organization

**Location:**
- Co-located organization: Tests in `swarm_rag_module/tests/` parallel to `swarm_rag_module/swarm_rag/`
- Subdirectories by test type: `unit/`, `integration/`, `evolution/`, `bugs/`
- Shared fixtures in `conftest.py` at test root

**Naming:**
- Test files: `test_*.py` or `*_test.py`
- Test functions: `def test_*` prefix
- Descriptive names: `test_fitness_logic()`, `test_expression_evaluation()`, `test_genome_compiler()`
- Bug reproduction files in `bugs/`: `reproduce_compiler_bug.py`, `reproduce_fitness.py`

**Structure:**
```
swarm_rag_module/tests/
├── conftest.py                          # Shared fixtures
├── unit/                                # Unit tests for individual components
│   ├── test_core_evo.py
│   ├── test_fitness_strategies.py
│   ├── test_norm.py
│   └── test_tracker.py
├── integration/                         # Integration tests (end-to-end)
│   ├── test_full_evolution.py
│   ├── test_map_elites.py
│   └── test_integrations_evo.py
├── evolution/                           # Evolution-specific tests
│   ├── test_genome.py
│   ├── test_evaluator.py
│   ├── test_focused_mutation.py
│   └── test_seed_genomes.py
└── bugs/                                # Bug reproduction and regression
    ├── reproduce_compiler_bug.py
    └── reproduce_fitness.py
```

## Test Structure

**Suite Organization:**
```python
# Typical structure from test_core_evo.py
def test_fitness_logic():
    """Descriptive docstring."""
    print("\n--- Testing FitnessResult (Lexicographic Sorting) ---")

    # 1. Setup
    f1 = FitnessResult(quality_score=0.8, stability_score=0.5)
    f2 = FitnessResult(quality_score=0.6, stability_score=0.9)

    # 2. Action
    # (implicit in comparison)

    # 3. Assert
    assert f1 > f2, "High quality should beat low quality"
    print("  ✓ Quality dominance check passed")
```

**Patterns:**
- Setup-Act-Assert pattern with clear comments
- Print statements for test progress (manual verification-friendly)
- Multiple assertions per test allowed when testing related scenarios
- Docstrings explain what is being tested, not just implementation

## Mocking

**Framework:** Manual mock classes (no external mocking library)

**Patterns:**
```python
# From test_evaluator.py
class MockRetriever:
    pass

class MockEvaluator:
    k_values = [1, 5, 10, 20]

class MockFitnessCalc:
    pass

evaluator = PopulationEvaluator(
    retriever=MockRetriever(),
    evaluator=MockEvaluator(),
    fitness_calc=MockFitnessCalc(),
)
```

**Alternative: Functional Mocks** (from `conftest.py`):
```python
class ToyStochasticRetriever:
    """Fully functional retriever simulation for testing."""
    def retrieve_batch(self, queries: List[str], max_workers: int = 1, **kwargs):
        # Real logic that simulates retriever behavior
        results = []
        n_agents = kwargs.get('n_agents', 1)
        step_prob = kwargs.get('alpha', 0.5)
        for q in queries:
            target_node = int(q)
            found = False
            for _ in range(n_agents):
                current_pos = 0
                # Simulation logic...
            results.append([{'id': target_node, 'score': 1.0}])
        return results
```

**What to Mock:**
- External dependencies (retrieval backends, evaluators)
- Classes with complex initialization (`PopulationEvaluator`)
- Infrastructure components (filesystem, network)

**What NOT to Mock:**
- Core algorithm logic (`Genome`, `FitnessResult`, `ExpressionNode`)
- Business domain objects - prefer real instances or functional substitutes
- Internal data structures

## Fixtures and Factories

**Test Data:**
```python
# From conftest.py
@pytest.fixture
def test_evolution_config(test_storage_config):
    """Returns a standard evolution config for tests."""
    config = EvolutionConfig(storage=test_storage_config)
    config.n_generations = 3
    config.map_elites.batch_size = 10
    config.genetic.selection_k = 3
    config.genetic.param_ranges.n_agents = (1, 5)
    config.genetic.param_ranges.decay = (0.1, 0.99)
    return config

@pytest.fixture
def train_data():
    """Provide simple training data for tests."""
    queries = ["10", "15", "20"]
    ground_truth = [[10], [15], [20]]
    return queries, ground_truth
```

**Location:**
- Fixtures defined in `C:\Users\jinwi\programming_files_NEW\swarm_rag_experiment\swarm_rag_module\tests\conftest.py`
- Shared across all test files via pytest's automatic discovery
- Domain-specific fixtures (e.g., `toy_retriever`, `int_evaluator`) grouped by purpose

**Fixture Patterns:**
- `@pytest.fixture` decorator with descriptive names
- Fixtures accept other fixtures as parameters (dependency injection)
- Temp directories managed via pytest's `tmp_path` fixture:
  ```python
  @pytest.fixture
  def temp_results_dir(tmp_path):
      results_dir = tmp_path / "evo_results"
      results_dir.mkdir(parents=True, exist_ok=True)
      yield str(results_dir)
  ```
- Cleanup handled by pytest or explicit cleanup in fixture

## Coverage

**Requirements:** Not enforced (no coverage config detected)

**View Coverage:**
```bash
pytest swarm_rag_module/tests/ --cov=swarm_rag --cov-report=html
pytest swarm_rag_module/tests/ --cov=swarm_rag --cov-report=term-missing
```

## Test Types

**Unit Tests:**
- Located in `tests/unit/`
- Test individual functions and methods in isolation
- Examples: `test_core_evo.py`, `test_fitness_strategies.py`, `test_genome.py`
- Scope: Single class or function behavior
- Fast execution; no filesystem or network I/O
- Example from `test_core_evo.py`:
  ```python
  def test_fitness_logic():
      f1 = FitnessResult(quality_score=0.8, stability_score=0.5)
      f2 = FitnessResult(quality_score=0.6, stability_score=0.9)
      assert f1 > f2, "High quality should beat low quality"
  ```

**Integration Tests:**
- Located in `tests/integration/`
- Test multiple components working together
- Examples: `test_full_evolution.py`, `test_map_elites.py`, `test_complex_integration.py`
- Scope: End-to-end workflows with real data flow
- Slower; may use temporary filesystem
- Example from `test_full_evolution.py`:
  ```python
  def test_integration_toy_retriever():
      """Full evolution loop with toy retriever."""
      storage = StorageConfig(...)
      config = EvolutionConfig(storage=storage)
      engine = EvolutionEngine(
          retriever=ToyStochasticRetriever(),
          fitness_calculator=FitnessCalculator(weights={"Recall@10": 1.0}),
          evaluator=IntEvaluator(),
          train_query_ids=["10", "15", "20"],
          train_ground_truth=[[10], [15], [20]],
          val_query_ids=["12"],
          val_ground_truth=[[12]],
          config=config
      )
      best = engine.optimize()
      assert best.fitness.quality_score > 0.0
  ```

**E2E Tests:**
- Not explicitly present; integration tests serve as E2E
- No dedicated E2E framework detected (no selenium, playwright, etc.)
- Tests use simulation (`ToyStochasticRetriever`) instead of external services

## Common Patterns

**Async Testing:**
- No async tests detected (codebase is synchronous)
- No pytest-asyncio or similar plugins in use

**Error Testing:**
```python
# Implicit via assertion failure
assert param not in evolvable_ranges, f"{param} should not be evolvable"

# From test_genome.py
def test_fixed_params_have_default_values():
    """Fixed parameters should have default values."""
    assert "drop_zone_inc" in FIXED_PARAMS
    assert FIXED_PARAMS["drop_zone_inc"] == 0.05
```

**Parametrized Tests:**
- Not heavily used (not detected in current codebase)
- Could be applied to test multiple scenarios with `@pytest.mark.parametrize`
- Example pattern (not currently in code):
  ```python
  @pytest.mark.parametrize("quality,stability,expected", [
      (0.8, 0.5, True),
      (0.6, 0.9, False),
  ])
  def test_fitness_comparison(quality, stability, expected):
      f1 = FitnessResult(quality_score=quality, stability_score=stability)
      f2 = FitnessResult(quality_score=0.7, stability_score=0.7)
      assert (f1 > f2) == expected
  ```

**Test Organization:**
- Test files co-located with related integration tests
- Conftest provides shared infrastructure and test doubles
- Bug files in `tests/bugs/` for regression prevention
- Evolution-specific tests in `tests/evolution/` for domain clustering

---

*Testing analysis: 2026-01-29*
