# Evolution Speedup & Convergence Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Achieve 3-5x faster evolution and improved convergence toward target metrics (Hit@1 >50%, Hit@5 >75%, Recall@20 >80%, MRR >75%)

**Architecture:** Implement highest-impact improvements from brainstorming document: parallel LLM mutations, aggressive early stopping, warm-start populations, focused metric-aware mutations, and streamlined hyperparameter search space.

**Tech Stack:** Python, PyTorch, concurrent.futures, existing swarm_rag evolution infrastructure

---

## Task 1: Streamline Hyperparameter Search Space

**Files:**
- Modify: `swarm_rag_module/swarm_rag/evolution/types/genome.py:15-50`
- Modify: `swarm_rag_module/swarm_rag/evolution/execution/strategies.py:100-200`
- Test: `tests/evolution/test_genome.py` (create if not exists)

### Step 1: Write failing test for fixed parameters

```python
# tests/evolution/test_genome.py
import pytest
from swarm_rag.evolution.types.genome import Genome, FIXED_PARAMS, EVOLVABLE_PARAM_RANGES

def test_fixed_params_not_in_evolvable_ranges():
    """Fixed parameters should not be in evolvable ranges."""
    for param in FIXED_PARAMS:
        assert param not in EVOLVABLE_PARAM_RANGES, f"{param} should not be evolvable"

def test_fixed_params_have_default_values():
    """Fixed parameters should have default values."""
    assert "drop_zone_inc" in FIXED_PARAMS
    assert FIXED_PARAMS["drop_zone_inc"] == 0.05
    assert "start_subset" in FIXED_PARAMS
    assert FIXED_PARAMS["start_subset"] == 10

def test_genome_uses_fixed_params():
    """New genomes should use fixed parameter values."""
    genome = Genome.create_random()
    assert genome.params["drop_zone_inc"] == 0.05
    assert genome.params["start_subset"] == 10

def test_evolvable_ranges_are_tightened():
    """Evolvable parameter ranges should be tightened per brainstorm."""
    assert EVOLVABLE_PARAM_RANGES["n_agents"] == (15, 50)
    assert EVOLVABLE_PARAM_RANGES["steps"] == (3, 7)
    assert EVOLVABLE_PARAM_RANGES["decay"] == (0.3, 0.8)
    assert EVOLVABLE_PARAM_RANGES["initial_pool_size"] == (20, 60)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/evolution/test_genome.py -v`
Expected: FAIL with "FIXED_PARAMS" not defined or wrong values

### Step 3: Add fixed and evolvable parameter definitions to genome.py

```python
# Add after imports in swarm_rag_module/swarm_rag/evolution/types/genome.py

# Fixed parameters - removed from evolution to reduce search space
FIXED_PARAMS: Dict[str, Any] = {
    "drop_zone_inc": 0.05,  # Rarely impacts results significantly
    "start_subset": 10,     # 10 starting nodes is usually sufficient
}

# Tightened ranges for evolvable parameters
EVOLVABLE_PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "n_agents": (15, 50),        # Fewer agents = faster, too many = redundant
    "steps": (3, 7),             # Most signal captured in 3-6 steps
    "decay": (0.3, 0.8),         # Keep dynamic, tighten range
    "initial_pool_size": (20, 60),  # Tight range around optimal
}
```

### Step 4: Update Genome class to use fixed params

```python
# Modify Genome.__init__ or create_random() to use FIXED_PARAMS
def create_random(cls) -> "Genome":
    """Create a new genome with random evolvable params and fixed params."""
    params = dict(FIXED_PARAMS)  # Start with fixed values
    for name, (low, high) in EVOLVABLE_PARAM_RANGES.items():
        if isinstance(low, int):
            params[name] = random.randint(low, high)
        else:
            params[name] = random.uniform(low, high)
    # ... rest of initialization
```

### Step 5: Run test to verify it passes

Run: `pytest tests/evolution/test_genome.py -v`
Expected: PASS

### Step 6: Update mutation strategies to respect fixed params

```python
# In swarm_rag_module/swarm_rag/evolution/execution/strategies.py
# Modify parameter mutation logic to skip FIXED_PARAMS

def _mutate_params(self, genome: Genome) -> None:
    """Mutate only evolvable parameters."""
    from swarm_rag.evolution.types.genome import FIXED_PARAMS, EVOLVABLE_PARAM_RANGES

    for name, (low, high) in EVOLVABLE_PARAM_RANGES.items():
        if name in FIXED_PARAMS:
            continue  # Skip fixed parameters
        if random.random() < self.mutation_rate:
            # ... existing mutation logic
```

### Step 7: Run full test suite to verify no regressions

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

### Step 8: Commit

```bash
git add swarm_rag_module/swarm_rag/evolution/types/genome.py swarm_rag_module/swarm_rag/evolution/execution/strategies.py tests/evolution/test_genome.py
git commit -m "feat: streamline hyperparameter search space

- Fix drop_zone_inc=0.05 and start_subset=10 (remove from evolution)
- Tighten evolvable ranges: n_agents 15-50, steps 3-7, decay 0.3-0.8
- Reduces search space for faster convergence

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Implement Aggressive Early Stopping Tiers

**Files:**
- Modify: `swarm_rag_module/swarm_rag/evolution/execution/evaluator.py:30-80`
- Test: `tests/evolution/test_evaluator.py` (create if not exists)

### Step 1: Write failing test for new tier configuration

```python
# tests/evolution/test_evaluator.py
import pytest
from swarm_rag.evolution.execution.evaluator import DEFAULT_TIERS, EvaluationTier

def test_aggressive_early_stopping_tiers():
    """Verify aggressive early stopping tier configuration."""
    assert len(DEFAULT_TIERS) == 4

    # Tier 1: Quick filter (3 queries, 0.15 threshold)
    assert DEFAULT_TIERS[0].queries == 3
    assert DEFAULT_TIERS[0].threshold == 0.15

    # Tier 2: Filter poor performers (8 queries, 0.30 threshold)
    assert DEFAULT_TIERS[1].queries == 8
    assert DEFAULT_TIERS[1].threshold == 0.30

    # Tier 3: Filter mediocre (20 queries, 0.45 threshold)
    assert DEFAULT_TIERS[2].queries == 20
    assert DEFAULT_TIERS[2].threshold == 0.45

    # Tier 4: Full evaluation (no threshold)
    assert DEFAULT_TIERS[3].threshold is None

def test_tier_thresholds_are_progressive():
    """Thresholds should increase with each tier."""
    thresholds = [t.threshold for t in DEFAULT_TIERS if t.threshold is not None]
    for i in range(1, len(thresholds)):
        assert thresholds[i] > thresholds[i-1], "Thresholds must be progressive"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/evolution/test_evaluator.py::test_aggressive_early_stopping_tiers -v`
Expected: FAIL with wrong tier values

### Step 3: Update DEFAULT_TIERS in evaluator.py

```python
# swarm_rag_module/swarm_rag/evolution/execution/evaluator.py

DEFAULT_TIERS = [
    EvaluationTier(queries=3, threshold=0.15, name="quick_filter"),   # Filter broken
    EvaluationTier(queries=8, threshold=0.30, name="poor_filter"),    # Filter poor
    EvaluationTier(queries=20, threshold=0.45, name="mediocre_filter"),  # Filter mediocre
    EvaluationTier(queries=100_000, threshold=None, name="full"),     # Full eval
]
```

### Step 4: Run test to verify it passes

Run: `pytest tests/evolution/test_evaluator.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add swarm_rag_module/swarm_rag/evolution/execution/evaluator.py tests/evolution/test_evaluator.py
git commit -m "feat: implement aggressive early stopping tiers

- Tier 1: 3 queries @ 0.15 threshold (filter completely broken)
- Tier 2: 8 queries @ 0.30 threshold (filter poor performers)
- Tier 3: 20 queries @ 0.45 threshold (filter mediocre)
- Expected: ~20% of genomes reach full evaluation

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Add Warm-Start Seed Genomes

**Files:**
- Create: `swarm_rag_module/swarm_rag/evolution/seed_genomes.py`
- Modify: `swarm_rag_module/swarm_rag/evolution/execution/strategies.py` (SEEDED_INITIALIZATION)
- Test: `tests/evolution/test_seed_genomes.py`

### Step 1: Write failing test for seed genomes

```python
# tests/evolution/test_seed_genomes.py
import pytest
from swarm_rag.evolution.seed_genomes import SEED_GENOMES, create_seed_genome

def test_seed_genomes_exist():
    """At least 3 seed genomes should be defined."""
    assert len(SEED_GENOMES) >= 3

def test_seed_genome_has_required_fields():
    """Each seed genome should have required fields."""
    required_fields = ["n_agents", "steps", "decay", "initial_pool_size",
                       "movement_tree", "deposit_tree"]
    for seed in SEED_GENOMES:
        for field in required_fields:
            assert field in seed, f"Seed missing {field}"

def test_create_seed_genome_returns_valid_genome():
    """create_seed_genome should return a valid Genome object."""
    genome = create_seed_genome(SEED_GENOMES[0])
    assert genome is not None
    assert genome.params["n_agents"] == SEED_GENOMES[0]["n_agents"]
    assert genome.evaluated is False

def test_seed_genomes_use_fixed_params():
    """Seed genomes should use fixed parameter values."""
    from swarm_rag.evolution.types.genome import FIXED_PARAMS
    genome = create_seed_genome(SEED_GENOMES[0])
    for key, value in FIXED_PARAMS.items():
        assert genome.params[key] == value
```

### Step 2: Run test to verify it fails

Run: `pytest tests/evolution/test_seed_genomes.py -v`
Expected: FAIL with "seed_genomes module not found"

### Step 3: Create seed_genomes.py with known good configurations

```python
# swarm_rag_module/swarm_rag/evolution/seed_genomes.py
"""
Known good genome configurations for warm-starting evolution.
These configurations provide a strong baseline and reduce wasted
generations discovering basic effective strategies.
"""
from typing import Dict, Any, List
from swarm_rag.evolution.types.genome import Genome, FIXED_PARAMS

SEED_GENOMES: List[Dict[str, Any]] = [
    # High-semantic config with balanced exploration
    {
        "name": "semantic_balanced",
        "n_agents": 25,
        "steps": 4,
        "decay": 0.5,
        "initial_pool_size": 30,
        "movement_tree": "ADD(MUL(semantic_similarity, 0.7), MUL(node_centrality, 0.3))",
        "deposit_tree": "semantic_similarity",
        "ranking_tree": "ADD(MUL(visit_count, 0.4), MUL(semantic_similarity, 0.6))",
    },
    # Hub-explorer config - emphasizes graph structure
    {
        "name": "hub_explorer",
        "n_agents": 30,
        "steps": 5,
        "decay": 0.6,
        "initial_pool_size": 40,
        "movement_tree": "ADD(MUL(node_centrality, 0.5), MUL(semantic_similarity, 0.3), MUL(pheromone_repulsion, 0.2))",
        "deposit_tree": "node_centrality",
        "ranking_tree": "ADD(MUL(pheromone_level, 0.3), MUL(semantic_similarity, 0.7))",
    },
    # Diversity-focused config - avoids clustering
    {
        "name": "diversity_focused",
        "n_agents": 20,
        "steps": 4,
        "decay": 0.4,
        "initial_pool_size": 35,
        "movement_tree": "ADD(MUL(pheromone_repulsion, 0.4), MUL(semantic_similarity, 0.6))",
        "deposit_tree": "MUL(semantic_similarity, pheromone_repulsion)",
        "ranking_tree": "ADD(MUL(visit_count, 0.5), MUL(pheromone_level, 0.5))",
    },
    # Conservative config - fewer agents, more steps
    {
        "name": "conservative_deep",
        "n_agents": 18,
        "steps": 6,
        "decay": 0.7,
        "initial_pool_size": 25,
        "movement_tree": "ADD(MUL(semantic_similarity, 0.8), MUL(node_centrality, 0.2))",
        "deposit_tree": "semantic_similarity",
        "ranking_tree": "semantic_similarity",
    },
    # Aggressive exploration config
    {
        "name": "aggressive_explorer",
        "n_agents": 45,
        "steps": 3,
        "decay": 0.35,
        "initial_pool_size": 50,
        "movement_tree": "ADD(MUL(pheromone_repulsion, 0.5), MUL(semantic_similarity, 0.5))",
        "deposit_tree": "ADD(semantic_similarity, node_centrality)",
        "ranking_tree": "ADD(MUL(visit_count, 0.6), MUL(semantic_similarity, 0.4))",
    },
]


def create_seed_genome(seed_config: Dict[str, Any]) -> Genome:
    """
    Create a Genome from a seed configuration.

    Args:
        seed_config: Dictionary with seed parameters and tree strings

    Returns:
        Initialized Genome with fixed params and seed configuration
    """
    from swarm_rag.evolution.types.genome import Genome, FIXED_PARAMS
    from swarm_rag.evolution.types.expression_tree import parse_expression_tree

    # Start with fixed params
    params = dict(FIXED_PARAMS)

    # Add evolvable params from seed
    params["n_agents"] = seed_config["n_agents"]
    params["steps"] = seed_config["steps"]
    params["decay"] = seed_config["decay"]
    params["initial_pool_size"] = seed_config["initial_pool_size"]

    # Parse expression trees
    strategies = {
        "ranking": parse_expression_tree(seed_config["ranking_tree"]),
        "g0_movement": parse_expression_tree(seed_config["movement_tree"]),
        "g0_deposit": parse_expression_tree(seed_config["deposit_tree"]),
    }

    return Genome(
        params=params,
        strategies=strategies,
        group_ratios={"g0": 1.0},
        evaluated=False,
    )
```

### Step 4: Run test to verify it passes

Run: `pytest tests/evolution/test_seed_genomes.py -v`
Expected: PASS

### Step 5: Update SEEDED_INITIALIZATION strategy to use seed genomes

```python
# In swarm_rag_module/swarm_rag/evolution/execution/strategies.py
# Update the SEEDED_INITIALIZATION strategy

from swarm_rag.evolution.seed_genomes import SEED_GENOMES, create_seed_genome

class SeededInitializationStrategy(InitializationStrategy):
    """Initialize population with known good seed genomes plus random fill."""

    def create_population(self, size: int) -> List[Genome]:
        population = []

        # Add seed genomes first
        for seed_config in SEED_GENOMES:
            if len(population) >= size:
                break
            genome = create_seed_genome(seed_config)
            population.append(genome)

        # Fill remaining with random genomes
        while len(population) < size:
            population.append(Genome.create_random())

        return population
```

### Step 6: Run full test suite

Run: `pytest tests/ -v --tb=short`
Expected: PASS

### Step 7: Commit

```bash
git add swarm_rag_module/swarm_rag/evolution/seed_genomes.py swarm_rag_module/swarm_rag/evolution/execution/strategies.py tests/evolution/test_seed_genomes.py
git commit -m "feat: add warm-start seed genomes for faster convergence

- Add 5 known good genome configurations
- Semantic balanced, hub explorer, diversity focused, conservative, aggressive
- SEEDED_INITIALIZATION uses seeds + random fill
- Reduces wasted generations discovering basics

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Implement Focused Metric-Aware Mutation

**Files:**
- Create: `swarm_rag_module/swarm_rag/evolution/execution/focused_mutation.py`
- Modify: `swarm_rag_module/swarm_rag/evolution/execution/strategies.py`
- Test: `tests/evolution/test_focused_mutation.py`

### Step 1: Write failing test for metric-aware mutation

```python
# tests/evolution/test_focused_mutation.py
import pytest
from swarm_rag.evolution.execution.focused_mutation import (
    identify_weakest_metric,
    get_mutation_focus,
    MutationFocus,
)

def test_identify_weakest_metric():
    """Should identify the metric furthest from target."""
    metrics = {
        "Hit@1": 0.45,      # Gap: 0.50 - 0.45 = 0.05
        "Hit@5": 0.60,      # Gap: 0.75 - 0.60 = 0.15  <-- WORST
        "Recall@20": 0.75,  # Gap: 0.80 - 0.75 = 0.05
        "MRR": 0.70,        # Gap: 0.75 - 0.70 = 0.05
    }
    assert identify_weakest_metric(metrics) == "Hit@5"

def test_identify_weakest_metric_hit1():
    """Should identify Hit@1 as weakest when appropriate."""
    metrics = {
        "Hit@1": 0.20,      # Gap: 0.30  <-- WORST
        "Hit@5": 0.70,      # Gap: 0.05
        "Recall@20": 0.78,  # Gap: 0.02
        "MRR": 0.72,        # Gap: 0.03
    }
    assert identify_weakest_metric(metrics) == "Hit@1"

def test_get_mutation_focus_hit1():
    """Hit@1 weakness should focus on ranking."""
    focus = get_mutation_focus("Hit@1")
    assert focus.ranking_intensity == "high"
    assert focus.movement_intensity == "low"

def test_get_mutation_focus_recall():
    """Recall@20 weakness should focus on movement and exploration."""
    focus = get_mutation_focus("Recall@20")
    assert focus.movement_intensity == "high"
    assert focus.bias_more_agents is True

def test_mutation_focus_dataclass():
    """MutationFocus should have required fields."""
    focus = MutationFocus(
        ranking_intensity="high",
        movement_intensity="low",
        deposit_intensity="medium",
        bias_more_agents=False,
        bias_more_steps=False,
    )
    assert focus is not None
```

### Step 2: Run test to verify it fails

Run: `pytest tests/evolution/test_focused_mutation.py -v`
Expected: FAIL with module not found

### Step 3: Create focused_mutation.py

```python
# swarm_rag_module/swarm_rag/evolution/execution/focused_mutation.py
"""
Focused mutation that targets genome components based on weakest metrics.
Instead of random mutation, this guides mutations toward components most
likely to improve the bottleneck metric.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import random

# Target metrics from brainstorm document
METRIC_TARGETS = {
    "Hit@1": 0.50,
    "Hit@5": 0.75,
    "Recall@20": 0.80,
    "MRR": 0.75,
}


@dataclass
class MutationFocus:
    """Specifies mutation intensity for different genome components."""
    ranking_intensity: str  # "high", "medium", "low"
    movement_intensity: str
    deposit_intensity: str
    bias_more_agents: bool  # Increase n_agents
    bias_more_steps: bool   # Increase steps


def identify_weakest_metric(metrics: Dict[str, float]) -> str:
    """
    Identify which metric has the largest gap from its target.

    Args:
        metrics: Current metric values

    Returns:
        Name of the weakest metric
    """
    gaps = {}
    for metric_name, target in METRIC_TARGETS.items():
        current = metrics.get(metric_name, 0.0)
        gaps[metric_name] = target - current

    return max(gaps, key=gaps.get)


def get_mutation_focus(weakest_metric: str) -> MutationFocus:
    """
    Determine mutation focus based on weakest metric.

    Args:
        weakest_metric: Name of the metric furthest from target

    Returns:
        MutationFocus specifying component intensities
    """
    if weakest_metric == "Hit@1":
        # Hit@1 low -> need better precision at top
        # Focus on ranking strategy
        return MutationFocus(
            ranking_intensity="high",
            movement_intensity="low",
            deposit_intensity="low",
            bias_more_agents=False,
            bias_more_steps=False,
        )

    elif weakest_metric == "Hit@5":
        # Hit@5 low -> need better early ranking
        # Balance between ranking and movement
        return MutationFocus(
            ranking_intensity="medium",
            movement_intensity="medium",
            deposit_intensity="low",
            bias_more_agents=False,
            bias_more_steps=False,
        )

    elif weakest_metric == "Recall@20":
        # Recall@20 low -> need better coverage/exploration
        # Focus on movement and increase exploration params
        return MutationFocus(
            ranking_intensity="low",
            movement_intensity="high",
            deposit_intensity="medium",
            bias_more_agents=True,
            bias_more_steps=True,
        )

    elif weakest_metric == "MRR":
        # MRR low -> first hit is too deep
        # Need both better movement and ranking
        return MutationFocus(
            ranking_intensity="medium",
            movement_intensity="medium",
            deposit_intensity="medium",
            bias_more_agents=False,
            bias_more_steps=False,
        )

    # Default: balanced mutation
    return MutationFocus(
        ranking_intensity="medium",
        movement_intensity="medium",
        deposit_intensity="medium",
        bias_more_agents=False,
        bias_more_steps=False,
    )


def apply_focused_mutation(genome: "Genome", focus: MutationFocus) -> "Genome":
    """
    Apply mutation to genome based on focus.

    Args:
        genome: Genome to mutate
        focus: MutationFocus specifying intensities

    Returns:
        Mutated genome
    """
    from swarm_rag.evolution.types.genome import EVOLVABLE_PARAM_RANGES

    # Map intensity to mutation probability
    intensity_probs = {"high": 0.8, "medium": 0.4, "low": 0.1}

    # Mutate ranking strategy
    if random.random() < intensity_probs[focus.ranking_intensity]:
        _mutate_strategy(genome, "ranking")

    # Mutate movement strategies
    if random.random() < intensity_probs[focus.movement_intensity]:
        for key in genome.strategies:
            if "movement" in key:
                _mutate_strategy(genome, key)

    # Mutate deposit strategies
    if random.random() < intensity_probs[focus.deposit_intensity]:
        for key in genome.strategies:
            if "deposit" in key:
                _mutate_strategy(genome, key)

    # Bias toward more agents if indicated
    if focus.bias_more_agents and random.random() < 0.5:
        low, high = EVOLVABLE_PARAM_RANGES["n_agents"]
        genome.params["n_agents"] = min(
            int(genome.params["n_agents"] * 1.1),
            high
        )

    # Bias toward more steps if indicated
    if focus.bias_more_steps and random.random() < 0.5:
        low, high = EVOLVABLE_PARAM_RANGES["steps"]
        genome.params["steps"] = min(
            genome.params["steps"] + 1,
            high
        )

    return genome


def _mutate_strategy(genome: "Genome", strategy_key: str) -> None:
    """Mutate a single strategy using standard tree mutation."""
    from swarm_rag.evolution.types.expression_tree import mutate_tree

    if strategy_key in genome.strategies:
        genome.strategies[strategy_key] = mutate_tree(
            genome.strategies[strategy_key]
        )
```

### Step 4: Run test to verify it passes

Run: `pytest tests/evolution/test_focused_mutation.py -v`
Expected: PASS

### Step 5: Register FOCUSED_MUTATION strategy

```python
# In swarm_rag_module/swarm_rag/evolution/execution/strategies.py
# Add new mutation strategy

from swarm_rag.evolution.execution.focused_mutation import (
    identify_weakest_metric,
    get_mutation_focus,
    apply_focused_mutation,
)

class FocusedMutationStrategy(MutationStrategy):
    """
    Metric-aware mutation that targets genome components based on
    which metric is furthest from target.
    """
    name = "FOCUSED_MUTATION"

    def mutate(self, genome: Genome, context: Optional[Dict] = None) -> Genome:
        """
        Apply focused mutation based on genome's current metrics.

        Args:
            genome: Genome to mutate
            context: Optional context with metrics

        Returns:
            Mutated genome
        """
        # Get current metrics (from context or genome)
        metrics = {}
        if context and "metrics" in context:
            metrics = context["metrics"]
        elif genome.metrics:
            metrics = genome.metrics

        # If no metrics available, fall back to standard mutation
        if not metrics:
            return self._standard_mutate(genome)

        # Identify weakest metric and get focus
        weakest = identify_weakest_metric(metrics)
        focus = get_mutation_focus(weakest)

        # Apply focused mutation
        return apply_focused_mutation(genome.copy(), focus)

# Register the strategy
GeneticRegistry.register_mutation("FOCUSED_MUTATION", FocusedMutationStrategy)
```

### Step 6: Run full test suite

Run: `pytest tests/ -v --tb=short`
Expected: PASS

### Step 7: Commit

```bash
git add swarm_rag_module/swarm_rag/evolution/execution/focused_mutation.py swarm_rag_module/swarm_rag/evolution/execution/strategies.py tests/evolution/test_focused_mutation.py
git commit -m "feat: implement focused metric-aware mutation

- Identify weakest metric relative to targets
- Focus mutations on components that affect weak metric
- Hit@1 weak -> focus ranking; Recall@20 weak -> focus movement
- Bias exploration params when recall is low

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Implement Parallel LLM Mutations

**Files:**
- Modify: `swarm_rag_module/swarm_rag/evolution/map_elites/loop.py`
- Modify: `swarm_rag_module/swarm_rag/evolution/types/config.py`
- Test: `tests/evolution/test_parallel_mutation.py`

### Step 1: Write failing test for parallel mutation config

```python
# tests/evolution/test_parallel_mutation.py
import pytest
from swarm_rag.evolution.types.config import EvolutionConfig

def test_config_has_parallel_mutation_workers():
    """Config should have parallel_mutation_workers setting."""
    config = EvolutionConfig()
    assert hasattr(config, "parallel_mutation_workers")
    assert config.parallel_mutation_workers >= 1

def test_default_parallel_workers():
    """Default should be 4 workers for parallel mutations."""
    config = EvolutionConfig()
    assert config.parallel_mutation_workers == 4
```

### Step 2: Run test to verify it fails

Run: `pytest tests/evolution/test_parallel_mutation.py -v`
Expected: FAIL with "parallel_mutation_workers" not found

### Step 3: Add parallel_mutation_workers to EvolutionConfig

```python
# In swarm_rag_module/swarm_rag/evolution/types/config.py

@dataclass
class EvolutionConfig:
    # ... existing fields ...

    # Parallel mutation settings
    parallel_mutation_workers: int = 4  # Number of workers for parallel mutations
```

### Step 4: Run test to verify it passes

Run: `pytest tests/evolution/test_parallel_mutation.py -v`
Expected: PASS

### Step 5: Write test for parallel mutation execution

```python
# Add to tests/evolution/test_parallel_mutation.py

def test_parallel_mutation_produces_multiple_offspring():
    """Parallel mutation should produce batch_size offspring."""
    from swarm_rag.evolution.map_elites.loop import MapElitesLoop
    from unittest.mock import MagicMock

    # Setup mock archive with a genome
    mock_archive = MagicMock()
    mock_genome = MagicMock()
    mock_genome.copy.return_value = MagicMock()
    mock_archive.select_parent.return_value = mock_genome

    loop = MapElitesLoop(
        archive=mock_archive,
        mutation_strategy=MagicMock(),
        crossover_strategy=MagicMock(),
        batch_size=8,
        parallel_workers=4,
    )

    offspring = loop._generate_offspring_parallel()
    assert len(offspring) == 8
```

### Step 6: Implement parallel mutation in MapElitesLoop

```python
# In swarm_rag_module/swarm_rag/evolution/map_elites/loop.py

from concurrent.futures import ThreadPoolExecutor, as_completed

class MapElitesLoop:
    def __init__(
        self,
        archive: "MAPElitesArchive",
        mutation_strategy: "MutationStrategy",
        crossover_strategy: "CrossoverStrategy",
        batch_size: int = 10,
        parallel_workers: int = 4,
    ):
        self.archive = archive
        self.mutation_strategy = mutation_strategy
        self.crossover_strategy = crossover_strategy
        self.batch_size = batch_size
        self.parallel_workers = parallel_workers

    def _generate_single_offspring(self, idx: int) -> "Genome":
        """Generate a single offspring (for parallel execution)."""
        parent = self.archive.select_parent()
        child = parent.copy()

        # Apply crossover with probability
        if random.random() < 0.3:
            other_parent = self.archive.select_parent()
            child = self.crossover_strategy.crossover(child, other_parent)

        # Apply mutation
        child = self.mutation_strategy.mutate(child)

        return child

    def _generate_offspring_parallel(self) -> List["Genome"]:
        """Generate batch of offspring using parallel workers."""
        offspring = []

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = [
                executor.submit(self._generate_single_offspring, i)
                for i in range(self.batch_size)
            ]

            for future in as_completed(futures):
                try:
                    child = future.result()
                    offspring.append(child)
                except Exception as e:
                    logger.warning(f"Parallel mutation failed: {e}")

        return offspring

    def step(self) -> List["Genome"]:
        """
        Execute one step of MAP-Elites loop.

        Returns:
            List of offspring genomes
        """
        if self.parallel_workers > 1:
            return self._generate_offspring_parallel()
        else:
            return [self._generate_single_offspring(i) for i in range(self.batch_size)]
```

### Step 7: Run full test suite

Run: `pytest tests/ -v --tb=short`
Expected: PASS

### Step 8: Commit

```bash
git add swarm_rag_module/swarm_rag/evolution/map_elites/loop.py swarm_rag_module/swarm_rag/evolution/types/config.py tests/evolution/test_parallel_mutation.py
git commit -m "feat: implement parallel LLM mutations

- Add parallel_mutation_workers config (default: 4)
- ThreadPoolExecutor for concurrent offspring generation
- Expected 3-4x speedup on mutation phase

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Add Adaptive Step Count (Early Convergence Detection)

**Files:**
- Modify: `swarm_rag_module/swarm_rag/core/swarm_retriever.py`
- Test: `tests/core/test_adaptive_steps.py`

### Step 1: Write failing test for convergence detection

```python
# tests/core/test_adaptive_steps.py
import pytest
import torch
from swarm_rag.core.swarm_retriever import should_continue_stepping

def test_should_continue_early_steps():
    """Always continue for first min_steps."""
    positions = torch.tensor([1, 2, 3, 4, 5])
    prev_positions = torch.tensor([1, 2, 3, 4, 5])  # All same (converged)

    # Step 0 should continue even if converged
    assert should_continue_stepping(positions, prev_positions, step_idx=0, min_steps=2) is True
    # Step 1 should continue even if converged
    assert should_continue_stepping(positions, prev_positions, step_idx=1, min_steps=2) is True

def test_should_stop_when_converged():
    """Stop if 80%+ agents haven't moved after min_steps."""
    # 10 agents, 8 stuck (80%)
    positions = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    prev_positions = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 99, 100])  # 8 same

    assert should_continue_stepping(positions, prev_positions, step_idx=3, min_steps=2) is False

def test_should_continue_when_not_converged():
    """Continue if agents are still moving."""
    # 10 agents, only 5 stuck (50%)
    positions = torch.tensor([1, 2, 3, 4, 5, 11, 12, 13, 14, 15])
    prev_positions = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    assert should_continue_stepping(positions, prev_positions, step_idx=3, min_steps=2) is True
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_adaptive_steps.py -v`
Expected: FAIL with "should_continue_stepping" not found

### Step 3: Implement should_continue_stepping

```python
# In swarm_rag_module/swarm_rag/core/swarm_retriever.py

def should_continue_stepping(
    positions: torch.Tensor,
    prev_positions: torch.Tensor,
    step_idx: int,
    min_steps: int = 2,
    convergence_threshold: float = 0.8,
) -> bool:
    """
    Check if agents have converged and further steps are unlikely to help.

    Args:
        positions: Current agent positions (n_agents,)
        prev_positions: Previous agent positions (n_agents,)
        step_idx: Current step index (0-based)
        min_steps: Always run at least this many steps
        convergence_threshold: Stop if this fraction of agents are stuck

    Returns:
        True if stepping should continue, False to stop early
    """
    # Always run minimum steps
    if step_idx < min_steps:
        return True

    # Count agents that haven't moved
    same_position = (positions == prev_positions).sum().item()
    convergence_ratio = same_position / len(positions)

    # Stop if too many agents are stuck
    if convergence_ratio >= convergence_threshold:
        return False

    return True
```

### Step 4: Run test to verify it passes

Run: `pytest tests/core/test_adaptive_steps.py -v`
Expected: PASS

### Step 5: Integrate into retrieve method

```python
# In SwarmRetriever.retrieve() method

def retrieve(self, query: str, k: int = 10, **kwargs) -> List[RetrievalResult]:
    # ... existing setup ...

    prev_positions = None
    for step in range(max_steps):
        # Store previous positions
        prev_positions_tensor = positions.clone() if prev_positions is None else prev_positions

        # Step agents
        positions = self._step_agents(positions, query_vec, pheromones, ...)

        # Check for early convergence
        if not should_continue_stepping(
            positions,
            prev_positions_tensor,
            step,
            min_steps=2,
            convergence_threshold=0.8,
        ):
            logger.debug(f"Early stop at step {step+1}/{max_steps} - agents converged")
            break

        prev_positions = positions.clone()

    # ... ranking and return ...
```

### Step 6: Run full test suite

Run: `pytest tests/ -v --tb=short`
Expected: PASS

### Step 7: Commit

```bash
git add swarm_rag_module/swarm_rag/core/swarm_retriever.py tests/core/test_adaptive_steps.py
git commit -m "feat: add adaptive step count with early convergence detection

- Stop early if 80%+ of agents haven't moved
- Always run minimum 2 steps for initial exploration
- Saves computation on easy queries

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Integration Test and Validation

**Files:**
- Create: `tests/integration/test_evolution_improvements.py`
- Run: Full benchmark

### Step 1: Write integration test

```python
# tests/integration/test_evolution_improvements.py
import pytest
from swarm_rag.evolution.engine import EvolutionEngine
from swarm_rag.evolution.types.config import EvolutionConfig
from swarm_rag.evolution.types.genome import FIXED_PARAMS, EVOLVABLE_PARAM_RANGES
from swarm_rag.evolution.seed_genomes import SEED_GENOMES
from swarm_rag.evolution.execution.evaluator import DEFAULT_TIERS

class TestEvolutionImprovements:
    """Integration tests for evolution speedup improvements."""

    def test_fixed_params_in_use(self):
        """Verify fixed parameters are being used."""
        assert "drop_zone_inc" in FIXED_PARAMS
        assert "start_subset" in FIXED_PARAMS

    def test_tightened_ranges(self):
        """Verify parameter ranges are tightened."""
        assert EVOLVABLE_PARAM_RANGES["n_agents"][1] <= 50
        assert EVOLVABLE_PARAM_RANGES["steps"][1] <= 7

    def test_aggressive_tiers(self):
        """Verify aggressive tier configuration."""
        assert DEFAULT_TIERS[0].queries == 3
        assert DEFAULT_TIERS[0].threshold == 0.15

    def test_seed_genomes_available(self):
        """Verify seed genomes are defined."""
        assert len(SEED_GENOMES) >= 3

    def test_config_has_parallel_workers(self):
        """Verify parallel mutation config exists."""
        config = EvolutionConfig()
        assert config.parallel_mutation_workers >= 1

    @pytest.mark.slow
    def test_evolution_runs_with_improvements(self):
        """Smoke test: evolution runs with all improvements."""
        config = EvolutionConfig(
            generations=2,
            population_size=10,
            parallel_mutation_workers=2,
        )
        engine = EvolutionEngine(config)

        # Should complete without errors
        result = engine.optimize(max_generations=2)
        assert result is not None
```

### Step 2: Run integration test

Run: `pytest tests/integration/test_evolution_improvements.py -v`
Expected: PASS

### Step 3: Commit integration test

```bash
git add tests/integration/test_evolution_improvements.py
git commit -m "test: add integration tests for evolution improvements

- Verify fixed params, tightened ranges, aggressive tiers
- Smoke test for evolution with all improvements

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

### Step 4: Run benchmark comparison (optional)

```bash
# Before improvements (baseline)
python -m swarm_rag.benchmarks.evolution_speed --generations 5 --output baseline.json

# After improvements
python -m swarm_rag.benchmarks.evolution_speed --generations 5 --output improved.json

# Compare
python -m swarm_rag.benchmarks.compare baseline.json improved.json
```

### Step 5: Final commit with all changes

```bash
git add -A
git commit -m "feat: complete evolution speedup and convergence improvements

Summary of changes:
- Streamlined hyperparameter search space (4 vs 6 params)
- Aggressive early stopping tiers (3q@0.15, 8q@0.30, 20q@0.45)
- Warm-start seed genomes (5 known good configurations)
- Focused metric-aware mutation
- Parallel LLM mutations (4 workers)
- Adaptive step count (early convergence detection)

Expected improvements:
- 3-5x faster evolution
- Better convergence toward target metrics

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

| Task | Description | Expected Impact |
|------|-------------|-----------------|
| 1 | Streamline hyperparameter search space | Faster convergence (smaller search space) |
| 2 | Aggressive early stopping tiers | 2-3x speedup (filter bad genomes faster) |
| 3 | Warm-start seed genomes | Fewer wasted generations |
| 4 | Focused metric-aware mutation | More directed search |
| 5 | Parallel LLM mutations | 3-4x speedup on mutation phase |
| 6 | Adaptive step count | Faster evaluation on easy queries |
| 7 | Integration test | Validate all improvements work together |

**Total estimated speedup:** 3-5x faster evolution with improved convergence
