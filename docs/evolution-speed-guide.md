# Evolution Speed Guide

Performance characteristics and configuration recommendations for the swarm RAG evolution system.

## Query Speed Baseline

**Single query on STARK Prime with CUDA: ~0.05 seconds** (after model loaded into VRAM)

This baseline assumes:
- Pre-warmed embedding model in GPU memory
- STARK Prime dataset (~3M nodes)
- Modern NVIDIA GPU (RTX 3080/4080 class or better)

CPU-only execution will be 5-10x slower per query.

## Evaluation Tier Settings

The adaptive evaluation system uses progressive tiers to filter poor-performing genomes early, saving computation on unpromising candidates.

### Current Default Tiers (v2 - doubled for better fitness signal)

| Tier | Queries | Threshold | Time/Genome | Purpose |
|------|---------|-----------|-------------|---------|
| `quick_filter` | 20 | 0.15 | 1.0s | Filter completely broken genomes |
| `poor_filter` | 50 | 0.30 | 2.5s | Filter poor performers |
| `mediocre_filter` | 100 | 0.45 | 5.0s | Filter mediocre genomes |
| `full` | All | None | Variable | Full evaluation for promising genomes |

### Previous Default Tiers (v1)

| Tier | Queries | Threshold | Time/Genome |
|------|---------|-----------|-------------|
| `quick_filter` | 10 | 0.15 | 0.5s |
| `poor_filter` | 25 | 0.30 | 1.25s |
| `mediocre_filter` | 50 | 0.45 | 2.5s |
| `full` | All | None | Variable |

### Trade-offs

**Higher query counts (current v2):**
- More reliable early fitness estimates
- Better convergence quality
- Reduced false positives/negatives in tier filtering
- 2x slower per-genome evaluation

**Lower query counts (previous v1):**
- Faster iteration cycles
- More generations per hour
- Higher variance in early-tier decisions
- May prematurely exit promising genomes

## Time Estimates

### Per-Genome Evaluation Time (0.05s/query)

| Scenario | Queries | Time |
|----------|---------|------|
| Quick filter exit (tier 1) | 20 | 1.0s |
| Poor filter exit (tier 2) | 50 | 2.5s |
| Mediocre filter exit (tier 3) | 100 | 5.0s |
| Full evaluation (200 queries) | 200 | 10.0s |
| Full evaluation (all ~3000) | 3000 | 2.5 min |

### Adaptive Evaluation - Typical Distribution

Assuming typical distribution: 60% tier 1, 20% tier 2, 15% tier 3, 5% full

| Population Size | Configuration | Avg Time/Genome | Total Time |
|-----------------|---------------|-----------------|------------|
| 100 genomes (initial) | v2 tiers | ~2.4s | ~4 min |
| 100 genomes (initial) | v1 tiers | ~1.2s | ~2 min |
| 30 genomes (per gen) | v2 tiers | ~2.4s | ~72s |
| 30 genomes (per gen) | v1 tiers | ~1.2s | ~36s |

### Full Evolution Run Estimates

| Generations | Offspring/Gen | Approx. Total Time (v2) |
|-------------|---------------|-------------------------|
| 50 | 30 | ~1 hour |
| 100 | 30 | ~2 hours |
| 500 | 30 | ~10 hours |
| 1000 | 30 | ~20 hours |

*Note: Initial population evaluation adds ~4 minutes overhead*

### Non-Adaptive Full Evaluation

When `enable_adaptive=False`, all genomes get full evaluation:

| Population Size | Queries/Genome | Time |
|-----------------|----------------|------|
| 100 genomes | 100 | ~8.3 min |
| 100 genomes | 200 | ~16.7 min |
| 30 genomes | 200 | ~5 min |
| 30 genomes | 3000 (all) | ~75 min |

## Optimization Features

The evaluator includes several optimizations that significantly reduce total runtime:

### 1. Shared Pre-computation (`enable_shared_precompute=True`)

**What it does:** Computes query embeddings and initial candidate pools once per generation, reusing them across all genome evaluations.

**Savings:** Eliminates redundant embedding computation. For 30 genomes, this saves ~29x the embedding cost.

### 2. Cross-Genome Metric Batching (`enable_cross_genome_metric_batch=True`)

**What it does:** Batches metric computation across all genomes in a single GPU call instead of computing metrics one genome at a time.

**Savings:** Reduces GPU kernel launch overhead and improves memory throughput. Typically 2-3x faster for metric computation phase.

### 3. Tier-Based Early Exit

**What it does:** Terminates evaluation of poor-performing genomes at early tiers based on quality thresholds.

**Savings:** Assuming 60/20/15/5 distribution across tiers:
- Average queries per genome: ~35 (vs 200+ for full)
- Effective speedup: ~5-6x for typical populations

### 4. Sequential GPU Evaluation

**What it does:** On CUDA devices, evaluates genomes sequentially to avoid GPU context switching overhead.

**Why:** CUDA contexts are thread-local. Parallel GPU evaluation from multiple threads creates contention and may actually slow things down.

## Configuration Recommendations

### Quick Iteration Mode

For rapid prototyping and early exploration:

```python
# Lower tier queries for faster cycles
custom_tiers = [
    EvaluationTier(queries=10, threshold=0.15, name="quick_filter"),
    EvaluationTier(queries=25, threshold=0.30, name="poor_filter"),
    EvaluationTier(queries=50, threshold=0.45, name="mediocre_filter"),
    EvaluationTier(queries=100_000, threshold=None, name="full"),
]

evaluator = PopulationEvaluator(
    tiers=custom_tiers,
    enable_adaptive=True,
    # ...
)
```

**Best for:** Initial experiments, testing new operators, debugging

### Production Quality Mode

For final runs where convergence quality matters:

```python
# Default v2 tiers (higher query counts)
evaluator = PopulationEvaluator(
    tiers=None,  # Uses DEFAULT_TIERS
    enable_adaptive=True,
    # ...
)
```

**Best for:** Final production runs, publishing results

### Maximum Precision Mode

When you need the most accurate fitness values:

```python
evaluator = PopulationEvaluator(
    enable_adaptive=False,  # Full evaluation for all genomes
    # ...
)
```

**Best for:** Final validation, comparing top candidates, research benchmarks

### Hybrid Approach

Run quick iteration for exploration, then re-evaluate top candidates with full precision:

```python
# Phase 1: Quick exploration (100-500 generations)
quick_evaluator = PopulationEvaluator(
    tiers=quick_tiers,
    enable_adaptive=True,
)
# ... run evolution ...

# Phase 2: Re-evaluate top 10 with full queries
top_genomes = select_top_n(population, n=10)
for genome in top_genomes:
    genome.evaluated = False  # Reset

precise_evaluator = PopulationEvaluator(
    enable_adaptive=False,
)
precise_evaluator.evaluate(top_genomes)
```

## Monitoring Progress

During evolution runs, the evaluator logs efficiency statistics:

```
Evaluation complete:
  > Avg queries/genome: 35.2 / 3000
  > Time saved estimate: 98.8%
  > Tier exits: {'quick_filter': 18, 'poor_filter': 6, 'mediocre_filter': 4, 'full': 2}
```

**Key metrics to watch:**
- `Avg queries/genome`: Lower is faster, but too low may indicate overly aggressive filtering
- `Time saved estimate`: Should be >90% for typical populations with adaptive evaluation
- `Tier exits`: Distribution across tiers; heavy concentration in tier 1 suggests thresholds may be too aggressive

## Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1080 (8GB) | RTX 3080+ (10GB+) |
| RAM | 16GB | 32GB+ |
| Storage | SSD | NVMe SSD |

VRAM is the primary bottleneck for embedding models. Larger VRAM allows:
- Larger batch sizes for embedding
- Keeping more of the index in GPU memory
- Reduced CPU-GPU transfer overhead
