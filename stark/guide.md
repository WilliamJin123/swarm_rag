# evolve_stark.py Guide

Run MAP-Elites evolutionary optimization for SwarmRAG on STaRK datasets.

## Quick Start

```bash
# Basic run on prime dataset
python evolve_stark.py --dataset prime

# Run with LLM-guided mutations
python evolve_stark.py --dataset prime --llm

# Start a new run with custom ID
python evolve_stark.py --dataset prime --run-id my_experiment

# Resume an existing run
python evolve_stark.py --dataset prime --resume runs/prime/20240123_143022

# List all existing runs
python evolve_stark.py --list-runs
```

### Prerequisites

1. **Dependencies**: Install the required packages
2. **For LLM mode**: Create a `.env` file with your API keys:
   ```
   CEREBRAS_API_KEY=your_key_here
   # Or for other providers:
   OPENAI_API_KEY=your_key_here
   GROQ_API_KEY=your_key_here
   ```

---

## Command-Line Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `prime` | STaRK dataset: `prime`, `amazon`, `mag` |
| `--gens` | `100` | Number of generations |
| `--train_ss` | `200` | Number of training samples for fitness evaluation |
| `--val_ss` | `100` | Number of validation samples |
| `--llm` | `off` | Enable LLM-guided mutations (flag) |
| `--run-id` | auto | Custom run identifier (auto-generated timestamp if not provided) |
| `--resume` | none | Path to run directory to resume (e.g., `runs/prime/20240123_143022`) |
| `--list-runs` | off | List all existing runs and exit (flag) |

---

## Runs Directory Structure

All outputs are organized in a dataset-first structure:

```
runs/{dataset}/{run_id}/
├── config.json         # Full experiment config snapshot
├── checkpoints/
│   ├── latest.pkl      # Most recent checkpoint
│   └── gen_XXX.pkl     # Per-generation checkpoints (every 5 gens)
├── logs/
│   └── evolution.jsonl # Structured metrics (JSONL format)
├── plots/
│   └── progress.png    # Evolution progress plot
└── results/
    ├── best_genome.json    # Best genome parameters
    └── final_metrics.json  # Final evaluation metrics
```

### Checkpoint Resume

Evolution automatically resumes from the last checkpoint:

```bash
# First run (creates checkpoint)
python evolve_stark.py --dataset prime --gens 50

# Resume from checkpoint (continues from gen 50)
python evolve_stark.py --dataset prime --gens 100 --resume runs/prime/20240123_143022

# List available runs to resume
python evolve_stark.py --list-runs
```

---

## MAP-Elites Deep Dive

MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) is a quality-diversity algorithm that maintains a grid of elite solutions across behavioral dimensions.

### Archive Structure

The archive is a grid-based structure with configurable bins:
- Default configuration: `[15, 12]` bins = **180 cells**
- Each cell stores the best genome found for that behavioral region

### Behavioral Descriptors

Two dimensions define the behavioral space:

| Descriptor | Formula | Range | Purpose |
|------------|---------|-------|---------|
| **Aggressiveness** | `n_agents × steps` | 10-150 | Measures exploration intensity |
| **Complexity** | Expression tree node count | 5-60 | Measures genome structural complexity |

### Bin Calculation

Continuous descriptor values are discretized into grid indices:

```python
# For each dimension:
norm = (value - min) / (max - min)
bin_index = floor(norm * n_bins)
bin_index = clamp(bin_index, 0, n_bins - 1)
```

### Archive Comparison Modes

Controls how genomes compete for archive cells:

| Mode | Description |
|------|-------------|
| `QUALITY_ONLY` | Simple quality_score comparison (default) |
| `WEIGHTED_COMPOSITE` | Weighted: 0.7×quality + 0.2×stability - 0.1×cost |
| `METRIC_THRESHOLD` | Must meet entry thresholds (Hit@1>0.10, etc.) then compare quality |
| `LEXICOGRAPHIC` | Quality > Stability > Cost (with margin) |

### Archive Metrics

- **Coverage**: Fraction of cells occupied (filled_cells / total_cells)
- **QD Score**: Sum of all elite fitness values
- **Max Fitness**: Best fitness in the archive

---

## Fitness Strategies

Three strategies for ranking genomes:

### Lexicographic (Default)

Standard sorting priority: **Quality > Stability > Cost**

```python
sort_key = (quality_score, stability_score, -cost_score)
```

Best for: Final refinement, when you want to maximize quality while considering stability.

### Pareto

NSGA-II inspired non-dominated sorting with crowding distance.

**How it works:**
1. Extract objectives: Quality (max), Stability (max), Cost (min)
2. Perform fast non-dominated sort into fronts
3. Calculate crowding distance within each front
4. Sort by: Rank (ascending), then Crowding Distance (descending)

Best for: Multi-objective exploration, finding diverse trade-offs.

### Phased

Hybrid approach: **Pareto early → Lexicographic late**

- Generations < `phased_switch_gen` (default 25): Use Pareto
- Generations >= `phased_switch_gen`: Use Lexicographic

Best for: Balancing exploration (early) with exploitation (late).

---

## Genetic Operators

### Creation Strategies

| Strategy | Description |
|----------|-------------|
| `standard_initialization` | Ramped half-and-half for diverse expression trees |
| `shallow_growth_initialization` | Max depth 2 for interpretable strategies |
| `seeded_initialization` | Injects Vector, Hybrid, Ant baselines + random fill |
| `baseline_seeded_initialization` | 5 empirically validated variants (recommended) |

**Baseline Seeds** (from `baseline_seeded_initialization`):
1. **Exact Baseline**: semantic(0.5) + centrality(0.2) + pheromone(0.25) + jitter(0.05)
2. **Variant 1**: Uses `semantic_unnormalized` for deposit
3. **No Stark**: Removes `stark_centrality` for graphs without it
4. **Explorer**: Higher pheromone (0.35), more steps (7), slower decay (0.8)
5. **More Agents**: 30 agents, pool_size=40

### Selection Strategies

| Strategy | Description |
|----------|-------------|
| `tournament` | k-tournament selection (default k=3) |
| `roulette` | Fitness-proportional selection |
| `boltzmann` | Adaptive temperature softmax (default) |
| `stochastic_universal_sampling` | Even spacing with random offset |
| `truncation` | Adaptive cutoff: 50%→10% over generations |

**Boltzmann Selection** (default):
- Probability: P(i) ~ exp(Fitness(i) / T)
- Adaptive temperature:
  - Low diversity → Heat up (explore more)
  - High diversity → Cool down (exploit)
- Temperature range: 0.1 to 5.0

### Crossover Strategies

| Strategy | Description |
|----------|-------------|
| `uniform_parameter_mix` | 50/50 trait inheritance (default) |
| `subtree_crossover` | GP-style random subtree swapping |
| `root_mix_crossover` | Top-level operator preservation |

### Mutation Strategies

| Strategy | Description | Rate |
|----------|-------------|------|
| `expression_tree_mutation` | Self-adaptive with smart jitter | Self-adaptive |
| `aggressive_mutation` | High-impact exploration | Fixed 40% |
| `guided_mutation` | Protects critical features (default) | Self-adaptive |
| `llm_mutation` | Three-tier LLM-guided mutations | Context-dependent |

**Guided Mutation** protects critical features:
- **Semantic features**: `semantic_similarity`, `semantic_similarity_unnormalized`
- **Centrality features**: `stark_centrality`, `node_centrality`
- **Diversity features**: `pheromone_repulsion`

If mutation would remove these features, 85% chance to revert.

---

## LLM Three-Tier Architecture

When `--llm` is enabled, mutations use a sophisticated three-tier system:

### Tier 1: Strategic Oracle

**Purpose**: Evolution-wide steering (called periodically every 5 generations or on stagnation)

**Input**: Archive statistics, QD trends, historical success rates

**Output**: `StrategicDirective` containing:
- **Mode**: Overall strategy direction
- **Focus Component**: What to prioritize mutating
- **Exploration Temperature**: How aggressive to be (0.0-1.0)
- **Priority Problems**: Issues to address first

**Available Modes**:
| Mode | When to Use |
|------|-------------|
| `explore_params` | Parameters might be suboptimal |
| `explore_strategies` | Expression/heuristic combinations need work |
| `exploit_top` | Good progress, maximize current solutions |
| `diversify` | Low coverage (<30%), need more diverse solutions |
| `targeted_fix` | Clear pattern in failed diagnoses |
| `balanced` | No strong preference |

### Tier 2: Tactical Advisor

**Purpose**: Per-genome diagnosis, prescribes mutation intents

**Input**: Simplified genome metrics, behavioral signature, strategic directive

**Output**: `MutationPrescription` containing:
- **Diagnosis**: Root cause explanation
- **Primary Intent**: Main mutation goal
- **Secondary Intent**: Optional complementary goal
- **Target Component**: What to change
- **Confidence**: Affects mutation magnitude (0.0-1.0)

**Available Intents** (17 total):

| Intent | Effect |
|--------|--------|
| `INCREASE_EXPLORATION` | More agents, steps, exploration heuristics |
| `REDUCE_EXPLORATION` | Focus on known good areas |
| `REDUCE_LOOPS` | Decrease revisit rate via pheromone |
| `REDUCE_REVISITS` | Similar to reduce_loops |
| `IMPROVE_COVERAGE` | More agents, diverse starting points |
| `INCREASE_DISPERSION` | Stronger repulsion, more spread |
| `REDUCE_COST` | Fewer agents/steps |
| `REDUCE_LATENCY` | Faster execution |
| `IMPROVE_QUALITY` | Better recall/precision |
| `INCREASE_SEMANTIC_FOCUS` | Prioritize semantic similarity |
| `AVOID_DEAD_ENDS` | Prefer well-connected nodes |
| `IMPROVE_CONNECTIVITY` | Favor hub nodes |
| `BALANCE_GROUPS` | Adjust agent group ratios |
| `REBALANCE_HEURISTICS` | Adjust heuristic weights |
| `SLOW_CONVERGENCE` | Agents shouldn't cluster too fast |
| `SPEED_CONVERGENCE` | Cluster faster on good results |
| `NO_CHANGE` | Genome is performing adequately |

### Tier 3: Constrained Executor

**Purpose**: Translates intents into validated mutations (DETERMINISTIC - no LLM calls)

**Responsibilities**:
1. Apply parameter adjustments within bounds
2. Build safe expressions from templates
3. Ensure all changes are valid
4. Track what changes were made for the journal

**Parameter Bounds**:
| Parameter | Range |
|-----------|-------|
| `n_agents` | 5-30 |
| `steps` | 4-12 |
| `decay` | 0.85-0.99 |
| `initial_pool_size` | 10-50 |
| `start_subset` | 5-15 |
| `drop_zone_inc` | 0.05-0.2 |

### Tier 2.5: Creative Synthesizer (Optional)

When creative mode is enabled and triggered:
- LLM generates custom heuristic expressions
- Expressions validated through AST parsing
- Complexity limited to prevent bloat

**Trigger Conditions** (any of):
- Stagnation: No improvement for 5+ generations
- Fill rate: Archive fill rate below 30%
- Top fitness unchanged for 3+ generations
- Periodic: Every 10 generations

---

## Multi-Tier Evaluation

Progressive evaluation reduces wasted computation on poor-performing genomes:

| Tier | Queries | Threshold | Purpose |
|------|---------|-----------|---------|
| 1 - Quick Filter | 5 | 0.10 | Eliminate clearly bad genomes |
| 2 - Promising | 15 | 0.25 | Confirm potential |
| 3 - Competitive | 40 | 0.50 | Validate competitive performance |
| 4 - Full | All | None | Complete evaluation using all `train_ss` queries |

**Early Exit Logic**:
```
For each tier:
  1. Evaluate tier queries
  2. Compute cumulative fitness
  3. If fitness < threshold: Exit early, mark genome
  4. Else: Continue to next tier
```

**Efficiency**: Most bad genomes exit at Tier 1 (5 queries), saving significant evaluation time.

**Note**: The final tier uses all available queries from `train_ss`. If you set `--train_ss 200`, promising genomes will be evaluated on all 200 queries.

---

## Configuration Reference

### EvolutionConfig

Top-level configuration:

```python
EvolutionConfig(
    n_generations=50,           # Total generations
    fitness_strategy="lexicographic",  # lexicographic, pareto, phased
    phased_switch_gen=25,       # Generation to switch (if phased)
    resources=ResourceConfig(...),
    map_elites=MapElitesConfig(...),
    genetic=GeneticConfig(...),
    llm=LLMConfig(...),
    creative_mode=CreativeModeConfig(...),
    storage=StorageConfig(...),
)
```

### ResourceConfig

```python
ResourceConfig(
    concurrent_evaluations=4,   # Parallel genome evaluations (auto-scaled)
    max_workers_per_retrieval=4,
    enable_dynamic_batch_size=True,
    base_batch_size=30,
    min_batch_size=15,
    max_batch_size=50,
)
```

### MapElitesConfig

```python
MapElitesConfig(
    dimensions=["aggressiveness", "complexity"],
    bins=[15, 12],              # 180 total cells
    ranges=[(10.0, 150.0), (5.0, 60.0)],
    initial_fill=100,           # Initial random population
    batch_size=30,              # Offspring per generation
    comparison_mode="quality_only",  # quality_only, weighted_composite, metric_threshold, lexicographic
)
```

### GeneticConfig

```python
GeneticConfig(
    creation_strategy="standard_initialization",
    selection_strategy="boltzmann",
    crossover_strategy="uniform_parameter_mix",
    mutation_strategy="guided_mutation",
    base_mutation_rate=0.25,
    crossover_rate=0.6,
    expr_max_depth=5,
    mutation_max_expr_size=25,
    n_agent_groups=3,
    selection_k=3,              # Tournament size
    boltzmann=BoltzmannConfig(...),
    param_ranges=SwarmParamRanges(...),
)
```

### BoltzmannConfig

```python
BoltzmannConfig(
    temperature=1.0,
    alpha=0.95,                 # Cooling factor
    min_temp=0.1,
    max_temp=5.0,
    adaptive=True,
    diversity_threshold=0.05,
)
```

### SwarmParamRanges

```python
SwarmParamRanges(
    n_agents=(5, 30),
    steps=(4, 12),
    decay=(0.85, 0.99),
    initial_pool_size=(10, 50),
    start_subset=(5, 15),
    drop_zone_inc=(0.05, 0.2),
)
```

### LLMConfig

```python
LLMConfig(
    enabled=False,
    provider="cerebras",        # cerebras, openai, groq, anthropic, together
    model="zai-glm-4.7",
    env_path=".env",
)
```

### CreativeModeConfig

```python
CreativeModeConfig(
    enabled=False,
    trigger_stagnation=5,       # Generations without improvement
    trigger_fill_rate=0.3,      # Archive fill rate threshold
    periodic_interval=10,       # Periodic experimentation interval
    max_creative_per_generation=3,
    complexity_limit=30,        # Max expression nodes
    fallback_on_failure=True,
    track_performance=True,
    max_consecutive_failures=5, # Auto-disable after N failures
)
```

### StorageConfig

```python
StorageConfig(
    base_dir="runs",
    dataset="prime",
    run_id=None,                # Auto-generated if None
    device="auto",              # "auto", "cuda", "mps", "cpu"
    checkpoint_frequency=5,
    validation_frequency=5,
    keep_n_checkpoints=10,      # 0 = keep all
    plot_title="MAP-Elites Evolution",
)
```

---

## Plotting & Metrics

### Evolution Progress Plot

The progress plot (`plots/progress.png`) shows:

**Left Y-Axis** (Quality Metrics):
- `train_best_quality`: Best training fitness over generations
- `train_avg_quality`: Average training fitness over generations
- `val_best_quality`: Best validation fitness (when evaluated)

**Right Y-Axis** (Cost Metrics):
- `train_best_cost`: Cost of best genome
- `train_best_latency`: Latency of best genome

### Archive Metrics (Logged to JSONL)

Each generation logs:
```json
{
    "generation": 42,
    "archive_coverage": 0.456,
    "archive_qd_score": 12.345,
    "archive_filled_cells": 82,
    "train_best_quality": 0.654,
    "train_avg_quality": 0.432,
    "train_best_cost": 125.0,
    "train_best_latency": 0.023,
    "val_best_quality": 0.612,
    "evaluation_time": 45.2
}
```

---

## Recommended Configurations

### Important Caveat

Achieving >90% across all metrics (MRR, Hit@K, Recall@K) is extremely ambitious and may not be achievable depending on dataset characteristics. The metrics that matter most depend on your use case:

- **MRR** (Mean Reciprocal Rank): How high is the first relevant result?
- **Hit@K**: Is there at least one relevant result in top K?
- **Recall@K**: What fraction of all relevant results are in top K?

### Prime Dataset (Smallest, Fastest Iteration)

Best for rapid experimentation and parameter tuning.

```bash
python evolve_stark.py --dataset prime \
  --gens 200 --train_ss 400 --val_ss 200 \
  --llm
```

### Amazon Dataset (Medium Size)

```bash
python evolve_stark.py --dataset amazon \
  --gens 150 --train_ss 300 --val_ss 150 \
  --llm
```

### MAG Dataset (Largest, Slowest)

```bash
python evolve_stark.py --dataset mag \
  --gens 150 --train_ss 300 --val_ss 150 \
  --llm
```

---

## Understanding Fitness Weights

The fitness function uses equal weights for key retrieval metrics:

| Metric | Weight | Description |
|--------|--------|-------------|
| Hit@1 | 0.25 | Is the top result relevant? |
| Hit@5 | 0.25 | Is there a relevant result in top 5? |
| MRR | 0.25 | Mean Reciprocal Rank |
| Recall@20 | 0.25 | Fraction of relevant results in top 20 |
| Complexity | -0.0001 | Small penalty for genome bloat |

---

## Troubleshooting

### Out of Memory
- Reduce concurrent evaluations in `CONFIG` dict
- Use `--gpu never` if GPU memory is limited
- Reduce sample sizes for faster iterations

### Slow Progress
- Ensure GPU is enabled (`device: "auto"` or `"cuda"`)
- Increase `concurrent_evals` if memory allows
- Reduce sample sizes for faster iterations

### LLM Errors
- Verify API keys in `.env`
- Check provider availability
- System falls back to heuristic mutations if LLM fails

### Stagnation
- Try different mutation strategies
- Enable LLM-guided mutations (`--llm`)
- Increase `initial_fill` for more diverse starting population

### Checkpoint Issues
- Use `--list-runs` to see available runs
- Resume with `--resume runs/dataset/run_id`
- Check that the checkpoint files exist in the run directory
