# Dual-Mode Evolution System Design

## Goal

Achieve the following metrics across STaRK Prime, Amazon, and MAG datasets:
- Hit@1 > 50%
- Hit@5 > 75%
- Recall@20 > 80%
- MRR > 75%

By running GPU-optimized evolution through thousands of generations with two distinct genome representations:
1. **Weighted Sum Mode** - Linear heuristic combinations (fast, interpretable)
2. **Expression Tree Mode** - Nonlinear symbolic expressions (expressive, current system)

---

## Architecture Overview

### Unified Genome Class

A single `Genome` class supports both modes, enabling shared infrastructure (evaluation, checkpointing, MAP-Elites archive).

```python
@dataclass
class Genome:
    id: str
    mode: Literal["weighted_sum", "expression_tree"]

    # === Shared Fields ===
    params: SwarmParams  # n_agents, steps, decay, initial_pool_size
    n_groups: int  # Number of agent groups (1-5, evolved)
    group_ratios: Dict[str, float]  # {"g0": 0.6, "g1": 0.4}

    # === Expression Tree Mode (existing) ===
    strategies: Dict[str, ExpressionNode]  # Empty if weighted_sum mode

    # === Weighted Sum Mode (new) ===
    weight_tensors: Optional[WeightTensors]  # None if expression_tree mode

    # === Self-Adaptive Mutation (ES-style) ===
    mutation_sigmas: MutationSigmas  # Evolved mutation strengths

    # === Evaluation Results ===
    fitness: FitnessResult
    metrics: Dict[str, float]
    evaluated: bool = False
```

### Weight Tensors Structure

For weighted sum mode, all weights stored as contiguous tensors for GPU batch operations:

```python
@dataclass
class WeightTensors:
    """GPU-friendly weight storage for heterogeneous agent groups."""

    # Movement: (n_groups, n_movement_features)
    movement_weights: torch.Tensor
    movement_biases: torch.Tensor  # (n_groups,)

    # Deposit: (n_groups, n_deposit_features)
    deposit_weights: torch.Tensor
    deposit_biases: torch.Tensor  # (n_groups,)

    # Ranking: shared across groups (n_ranking_features,)
    ranking_weights: torch.Tensor
    ranking_bias: float

    def to_device(self, device: str) -> "WeightTensors":
        """Move all tensors to specified device."""
        ...
```

### Self-Adaptive Mutation Sigmas

Each genome carries its own mutation strengths (Evolution Strategy style):

```python
@dataclass
class MutationSigmas:
    """Self-adaptive mutation parameters - these also evolve."""
    weight_sigma: float = 0.10      # Weight perturbation strength
    bias_sigma: float = 0.05        # Bias perturbation strength
    ratio_sigma: float = 0.10       # Group ratio shift strength
    hyperparam_sigma: float = 0.15  # Hyperparameter mutation strength

    # Meta-parameter for sigma evolution
    tau: float = 0.1  # Learning rate for sigma adaptation
```

---

## Heuristic Feature Configuration

User-configurable feature sets per strategy type, specified at run time:

```python
@dataclass
class HeuristicFeatureConfig:
    """Configurable feature sets for evolution runs."""

    movement: List[str] = field(default_factory=lambda: [
        "semantic_similarity_unnormalized",
        "node_centrality",
        "pheromone_repulsion",
    ])

    deposit: List[str] = field(default_factory=lambda: [
        "semantic_unnormalized",
        "exploration_bonus",
        "hub",
    ])

    ranking: List[str] = field(default_factory=lambda: [
        "percentage_visited",
        "semantic_rank",
    ])


# STaRK-specific config (includes stark_centrality)
STARK_FEATURES = HeuristicFeatureConfig(
    movement=[
        "semantic_similarity_unnormalized",
        "stark_centrality",
        "node_centrality",
        "pheromone_repulsion",
    ],
    deposit=[
        "flat",
        "semantic_unnormalized",
        "exploration_bonus",
    ],
    ranking=[
        "percentage_visited",
        "semantic_rank",
    ],
)
```

### Feature Recommendations by Strategy Type

**Movement** (where agents navigate):
| Feature | Default | Notes |
|---------|---------|-------|
| `semantic_similarity_unnormalized` | Yes | Core relevance signal |
| `node_centrality` | Yes | Graph structure awareness |
| `pheromone_repulsion` | Yes | Exploration/exploitation balance |
| `stark_centrality` | STaRK only | Graph-specific centrality |
| `random_jitter` | Optional | Can help escape local minima |

**Deposit** (pheromone leaving):
| Feature | Default | Notes |
|---------|---------|-------|
| `flat` | Yes | Baseline constant deposit |
| `semantic_unnormalized` | Yes | Quality-weighted deposit |
| `exploration_bonus` | Yes | Encourage diversity |
| `hub` | Optional | Reinforce central nodes |

**Ranking** (final scoring):
| Feature | Default | Notes |
|---------|---------|-------|
| `semantic_rank` | Yes | Direct relevance |
| `percentage_visited` | Yes | Swarm consensus |

---

## GPU Optimization Strategy

### Weighted Sum Mode - Batch Computation

All groups computed in single batched matmul:

```python
def compute_movement_scores_batched(
    features: torch.Tensor,      # (N_candidates, F_movement)
    weights: torch.Tensor,       # (G_groups, F_movement)
    biases: torch.Tensor,        # (G_groups,)
) -> torch.Tensor:
    """
    Compute movement scores for all candidates across all groups.
    Returns: (N_candidates, G_groups)
    """
    # Single cuBLAS matmul - extremely fast
    scores = features @ weights.T + biases  # (N, G)
    return scores
```

### Agent Group Assignment

```python
def assign_scores_to_agents(
    scores_all_groups: torch.Tensor,  # (N_candidates, G_groups)
    agent_group_ids: torch.Tensor,    # (A_agents,) values 0..G-1
) -> torch.Tensor:
    """
    Select appropriate group scores for each agent.
    Returns: (N_candidates, A_agents)
    """
    # Advanced indexing - still GPU-friendly
    return scores_all_groups[:, agent_group_ids]
```

### Memory Layout

For maximum GPU efficiency:
- All weight tensors pre-allocated on GPU at evolution start
- Population weights stored as single large tensor: `(pop_size, total_params)`
- Batch mutation via vectorized operations on entire population

---

## Mutation Operators

### Weighted Sum Mode

| Operator | Probability | Description |
|----------|-------------|-------------|
| Weight perturbation | 60% | `w += N(0, genome.mutation_sigmas.weight_sigma)` |
| Bias perturbation | 15% | `b += N(0, genome.mutation_sigmas.bias_sigma)` |
| Group ratio shift | 10% | Rebalance proportions |
| Hyperparam mutation | 10% | Mutate n_agents, steps, decay, pool_size |
| Group add/remove | 5% | Change n_groups by ±1 |

### Self-Adaptive Sigma Update

Before mutating weights, mutate the sigmas themselves:

```python
def adapt_sigmas(genome: Genome) -> None:
    """ES-style sigma adaptation."""
    tau = genome.mutation_sigmas.tau

    # Multiplicative update: sigma' = sigma * exp(tau * N(0,1))
    genome.mutation_sigmas.weight_sigma *= math.exp(tau * random.gauss(0, 1))
    genome.mutation_sigmas.bias_sigma *= math.exp(tau * random.gauss(0, 1))
    # ... etc

    # Clamp to reasonable range
    genome.mutation_sigmas.weight_sigma = max(0.01, min(0.5, genome.mutation_sigmas.weight_sigma))
```

### Expression Tree Mode

Existing operators preserved:
- Subtree mutation/replacement
- Node value mutation
- Subtree crossover
- Tree simplification

---

## Seed Population Strategy

### Baseline Configuration (from test_n_q.py)

```python
BASELINE_CONFIG = {
    "n_agents": 25,
    "steps": 5,
    "decay": 0.5,
    "initial_pool_size": 30,

    "movement_weights": {
        "semantic_similarity_unnormalized": 0.50,
        "stark_centrality": 0.20,
        "pheromone_repulsion": 0.25,
        "random_jitter": 0.05,
    },
    "deposit_weights": {
        "flat": 1.0,
    },
    "ranking_weights": {
        "semantic_rank": 0.90,
        "percentage_visited": 0.10,
    },
}
```

### Seed Variants

| # | Name | Variation |
|---|------|-----------|
| 1 | `baseline_exact` | Exact copy of test_n_q.py config |
| 2 | `high_semantic` | semantic 0.65, stark 0.15, pheromone 0.15, jitter 0.05 |
| 3 | `high_explore` | pheromone 0.40, semantic 0.35, stark 0.20, jitter 0.05 |
| 4 | `hub_focused` | stark 0.35, node_centrality 0.15, semantic 0.40, pheromone 0.10 |
| 5 | `no_jitter` | semantic 0.55, stark 0.20, pheromone 0.25 (no jitter) |
| 6 | `semantic_deposit` | deposit=semantic_unnormalized instead of flat |
| 7 | `consensus_rank` | ranking: percentage_visited 0.30, semantic_rank 0.70 |
| 8 | `more_agents` | n_agents=35, steps=4 |
| 9 | `deeper_search` | n_agents=20, steps=7, decay=0.7 |
| 10 | `fast_shallow` | n_agents=40, steps=3, pool=50 |
| 11-15 | `perturb_1-5` | Gaussian noise ±15% on baseline weights |
| 16-18 | `wildcard_1-3` | Random weights, baseline hyperparams |

---

## Configuration Structure

### Updated EvolutionConfig

```python
@dataclass
class EvolutionConfig:
    # === Mode Selection ===
    genome_mode: Literal["weighted_sum", "expression_tree"] = "expression_tree"

    # === Feature Configuration (weighted_sum mode) ===
    heuristic_features: HeuristicFeatureConfig = field(
        default_factory=HeuristicFeatureConfig
    )

    # === Existing Fields ===
    n_generations: int = 100
    fitness_strategy: str = "lexicographic"

    resources: ResourceConfig = field(default_factory=ResourceConfig)
    map_elites: MapElitesConfig = field(default_factory=MapElitesConfig)
    genetic: GeneticConfig = field(default_factory=GeneticConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
```

### Updated GeneticConfig

```python
@dataclass
class GeneticConfig:
    # === Existing Fields ===
    creation_strategy: str = "baseline_seeded_initialization"
    mutation_strategy: str = "guided_mutation"
    crossover_strategy: str = "uniform_parameter_mix"
    base_mutation_rate: float = 0.20
    crossover_rate: float = 0.6

    # === New: Weighted Sum Specific ===
    n_agent_groups_range: Tuple[int, int] = (1, 5)  # Min/max groups
    self_adaptive_mutation: bool = True  # Enable ES-style sigma evolution

    # === New: Initial Sigma Values ===
    initial_weight_sigma: float = 0.10
    initial_bias_sigma: float = 0.05
    initial_ratio_sigma: float = 0.10
    sigma_tau: float = 0.1  # Sigma learning rate
```

### Example Usage

```python
# Weighted sum mode for STaRK
config = EvolutionConfig(
    genome_mode="weighted_sum",
    heuristic_features=STARK_FEATURES,
    n_generations=1000,
    genetic=GeneticConfig(
        creation_strategy="weighted_sum_seeded",
        mutation_strategy="self_adaptive_es",
        n_agent_groups_range=(1, 5),
        self_adaptive_mutation=True,
    ),
    map_elites=MapElitesConfig(
        bins=[15, 12],
        initial_fill=100,
        batch_size=50,
    ),
)

# Expression tree mode (existing behavior)
config = EvolutionConfig(
    genome_mode="expression_tree",
    n_generations=500,
    genetic=GeneticConfig(
        creation_strategy="baseline_seeded_initialization",
        mutation_strategy="guided_mutation",
    ),
)
```

---

## Implementation Plan

### Phase 1: Core Data Structures
1. Add `WeightTensors` dataclass
2. Add `MutationSigmas` dataclass
3. Add `HeuristicFeatureConfig` dataclass
4. Extend `Genome` class with `mode`, `weight_tensors`, `mutation_sigmas` fields
5. Update `EvolutionConfig` and `GeneticConfig`

### Phase 2: Weighted Sum Compilation
1. Create `WeightedSumCompiler` (parallel to `GenomeCompiler`)
2. Implement `compile_to_strategies()` - converts weight tensors to callable strategies
3. GPU-optimized batch score computation

### Phase 3: Mutation Operators
1. Implement `WeightedSumMutator` class
2. Self-adaptive sigma mutation
3. Weight/bias perturbation
4. Group add/remove operations
5. Register as `"self_adaptive_es"` mutation strategy

### Phase 4: Seeding
1. Create `WeightedSumSeeder` class
2. Implement baseline + variant generation
3. Register as `"weighted_sum_seeded"` creation strategy

### Phase 5: Integration
1. Update `GenomeFactory` to handle both modes
2. Update `PopulationEvaluator` for weighted sum genomes
3. Update checkpoint serialization
4. Update MAP-Elites archive compatibility

### Phase 6: Testing & Validation
1. Unit tests for weighted sum operations
2. GPU vs CPU parity tests
3. Benchmark: generations/minute comparison
4. Validation run on STaRK Prime

---

## Runtime Analysis & Feasibility

### Bottleneck Analysis

The dominant cost is **genome evaluation** (retrieval). Other operations are negligible:

| Operation | Time per genome | Notes |
|-----------|-----------------|-------|
| Retrieval (per query) | ~0.1s raw | Without optimizations |
| Retrieval (per query, shared precompute) | ~0.02-0.03s | Query embeddings + initial pool cached |
| Mutation (weighted sum) | <1ms | Vectorized tensor ops |
| Mutation (expression tree) | ~5ms | Tree traversal |
| Crossover | <1ms | Both modes |
| Archive insertion | <1ms | Hash lookup |

### Key Optimizations Already Implemented

1. **Shared Precompute** - Query embeddings and initial pools computed ONCE per generation
2. **Cross-Genome Metric Batching** - Single GPU call for metrics across all genomes
3. **Adaptive Evaluation Tiers** - Bad genomes exit early with fewer queries:
   - Tier 1 (10 queries): ~20% of genomes exit here
   - Tier 2 (25 queries): ~30% exit
   - Tier 3 (50 queries): ~30% exit
   - Full (100+ queries): ~20% reach full evaluation

### Runtime Scenarios

**Conservative estimate** (with shared precompute, ~0.025s/query):

| Scenario | Generations | Batch Size | Train Queries | Avg Queries/Genome | Time/Gen | Total Time |
|----------|-------------|------------|---------------|-------------------|----------|------------|
| Quick test | 100 | 30 | 100 | 30 | 22s | ~37 min |
| Standard | 500 | 30 | 100 | 30 | 22s | ~3 hours |
| Extended | 1000 | 30 | 100 | 30 | 22s | ~6 hours |
| Intensive | 500 | 50 | 150 | 40 | 50s | ~7 hours |

**Aggressive estimate** (maximally optimized, ~0.015s/query):

| Scenario | Generations | Batch Size | Train Queries | Avg Queries/Genome | Time/Gen | Total Time |
|----------|-------------|------------|---------------|-------------------|----------|------------|
| Quick test | 100 | 30 | 100 | 25 | 11s | ~18 min |
| Standard | 500 | 30 | 100 | 25 | 11s | ~1.5 hours |
| Extended | 1000 | 30 | 100 | 25 | 11s | ~3 hours |
| Marathon | 2000 | 40 | 100 | 25 | 15s | ~8 hours |

### Recommended Default Settings

For a **~2-3 hour run** targeting the goal metrics:

```python
config = EvolutionConfig(
    genome_mode="weighted_sum",
    n_generations=500,
    map_elites=MapElitesConfig(
        bins=[15, 12],          # 180 archive cells
        initial_fill=100,       # Seed archive with 100 genomes
        batch_size=30,          # 30 offspring per generation
    ),
    resources=ResourceConfig(
        concurrent_evaluations=4,
        max_workers_per_retrieval=2,
        enable_shared_precompute=True,       # Critical for speed
        enable_cross_genome_metric_batch=True,
    ),
)

# Training data
train_sample_size = 100   # 100 queries for fitness evaluation
val_sample_size = 50      # 50 queries for validation checkpoints
```

### Scaling Recommendations

**To run faster (1-2 hours):**
- Reduce `train_sample_size` to 75
- Use more aggressive early exit thresholds
- Reduce `batch_size` to 25

**To run longer (overnight, 8+ hours):**
- Increase `n_generations` to 2000
- Increase `train_sample_size` to 150
- Increase `batch_size` to 50

**To maximize exploration:**
- Larger `initial_fill` (150-200)
- More MAP-Elites bins (20x15 = 300 cells)
- Lower early exit thresholds (keep more genomes in evaluation)

---

## Expected Performance

### Weighted Sum Mode
- **Genome size**: ~60 continuous parameters (vs variable tree size)
- **Mutation speed**: O(1) vectorized ops
- **Evaluation**: Same as expression tree (retrieval-dominated)
- **Generations/hour**: 2-3x more than expression trees (faster mutation/crossover)

### Expression Tree Mode
- **Genome size**: 5-50 nodes per strategy
- **Mutation speed**: O(tree_size) recursive ops
- **Expressiveness**: Nonlinear combinations, conditionals

### Comparison Value
Running both modes on same dataset allows:
1. Determine if nonlinearity helps (expression trees beat weighted sums?)
2. Find good linear approximations to complex strategies
3. Use weighted sum results to seed expression tree evolution

---

## Validation Checkpoints

During evolution, validate periodically to catch overfitting:

| Checkpoint | Frequency | Purpose |
|------------|-----------|---------|
| Training metrics | Every generation | Track fitness progress |
| Validation metrics | Every 10 generations | Detect overfitting |
| Best genome snapshot | Every 25 generations | Recovery point |
| Full checkpoint | Every 50 generations | Resume capability |

### Early Stopping Criteria

Consider stopping early if:
1. **Goal reached**: All 4 metrics hit targets on validation set
2. **Plateau detected**: Best fitness unchanged for 50+ generations
3. **Overfitting**: Training metrics improving but validation declining
