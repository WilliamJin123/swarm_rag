# evolve_stark.py Guide

Run MAP-Elites evolutionary optimization for SwarmRAG on STaRK datasets.

## Quick Start

```bash
# Basic run on prime dataset
python evolve_stark.py --dataset prime

# Run with LLM-guided mutations
python evolve_stark.py --dataset prime --llm

# Start from scratch (clear checkpoints)
python evolve_stark.py --dataset prime --scratch
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
| `--pop` | `30` | Offspring per generation (MAP-Elites batch size) |
| `--init_fill` | `100` | Initial random population to seed archive |
| `--train_ss` | `200` | Number of training samples for fitness evaluation |
| `--val_ss` | `100` | Number of validation samples |
| `--concurrent` | `4` | Number of concurrent genome evaluations |
| `--workers` | `4` | Threads per retrieval operation |
| `--llm` | `off` | Enable LLM-guided mutations (flag) |
| `--llm-provider` | `cerebras` | LLM provider: `cerebras`, `openai`, `groq`, `anthropic`, `together` |
| `--llm-model` | `zai-glm-4.7` | Model ID for the LLM provider |
| `--env-path` | `.env` | Path to .env file with API keys |
| `--mutation` | `guided_mutation` | Mutation strategy: `guided_mutation`, `expression_tree_mutation`, `aggressive_mutation` |
| `--gpu` | `auto` | GPU mode: `auto` (detect), `always` (require), `never` (CPU only) |
| `--scratch` | `off` | Clear previous checkpoints/logs (flag) |

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
  --gens 200 --pop 50 --init_fill 200 \
  --train_ss 400 --val_ss 200 \
  --llm --llm-provider cerebras \
  --mutation llm_mutation \
  --gpu always --concurrent 8
```

### Amazon Dataset (Medium Size)

```bash
python evolve_stark.py --dataset amazon \
  --gens 150 --pop 40 --init_fill 150 \
  --train_ss 300 --val_ss 150 \
  --llm --llm-provider cerebras \
  --mutation llm_mutation \
  --gpu always --concurrent 6
```

### MAG Dataset (Largest, Slowest)

```bash
python evolve_stark.py --dataset mag \
  --gens 150 --pop 40 --init_fill 150 \
  --train_ss 300 --val_ss 150 \
  --llm --llm-provider cerebras \
  --mutation llm_mutation \
  --gpu always --concurrent 6
```

---

## Key Parameters for Maximizing Metrics

### Generations & Population Size
- **More generations** = more evolutionary iterations = better exploration of the search space
- **Larger population** (`--pop`) = more offspring per generation = faster archive filling
- **Larger initial fill** (`--init_fill`) = more diverse starting points

### LLM-Guided Mutations
Enable with `--llm` for smarter, context-aware mutations:
- Analyzes behavioral patterns (revisit rates, dead-ends, convergence)
- Prescribes targeted fixes rather than random changes
- Uses three-tier architecture: Strategic Oracle -> Tactical Advisor -> Constrained Executor

### Sample Sizes
- Larger `--train_ss` and `--val_ss` = more robust fitness evaluation
- Trade-off: slower per-generation evaluation
- Recommended: At least 200 training samples for reliable fitness signals

### GPU Acceleration
- Critical for large datasets (Amazon, MAG)
- Use `--gpu always` when GPU is available
- Speeds up vector similarity computations significantly

### Concurrency
- `--concurrent`: Number of genomes evaluated in parallel
- Higher values = faster generations but more memory usage
- Start with 4-6 and increase if resources allow

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

## Output Files

All outputs are saved in the `stark/` directory:

| File | Location | Description |
|------|----------|-------------|
| Generation logs | `logs/evo_{dataset}.jsonl` | Per-generation statistics and metrics |
| Evolution plot | `logs/plot_{dataset}.png` | Visual plot of evolution progress |
| Checkpoints | `checkpoints/ckpt_{dataset}.pkl` | Resumable evolution state |
| Best genome | `best_params/best_params_{dataset}.json` | Final best genome parameters |

### Checkpoint Resume

Evolution automatically resumes from the last checkpoint:

```bash
# First run (creates checkpoint)
python evolve_stark.py --dataset prime --gens 50

# Resume from checkpoint (continues from gen 50)
python evolve_stark.py --dataset prime --gens 100

# Start fresh (ignores checkpoint)
python evolve_stark.py --dataset prime --gens 100 --scratch
```

---

## MAP-Elites Behavioral Dimensions

The archive uses two behavioral dimensions to maintain solution diversity:

1. **Aggressiveness** (range: 10-150)
   - Measures exploration intensity
   - Computed from: `n_agents * steps`

2. **Complexity** (range: 5-60)
   - Measures genome structural complexity
   - Computed from: number of expression nodes, parameters

This ensures the archive maintains diverse solutions across different exploration/complexity trade-offs.

---

## Example: Full Evolution Run

```bash
# 1. Start evolution with LLM guidance
python evolve_stark.py \
  --dataset prime \
  --gens 200 \
  --pop 50 \
  --init_fill 200 \
  --train_ss 400 \
  --val_ss 200 \
  --llm \
  --llm-provider cerebras \
  --llm-model zai-glm-4.7 \
  --gpu always \
  --concurrent 8

# 2. Monitor progress in logs/evo_prime.jsonl

# 3. Best genome saved to best_params/best_params_prime.json
```

---

## Troubleshooting

### Out of Memory
- Reduce `--concurrent` (fewer parallel evaluations)
- Reduce `--pop` (smaller batch size)
- Use `--gpu never` if GPU memory is limited

### Slow Progress
- Ensure GPU is enabled (`--gpu always`)
- Increase `--concurrent` if memory allows
- Reduce sample sizes for faster iterations

### LLM Errors
- Verify API keys in `.env`
- Check provider availability
- System falls back to heuristic mutations if LLM fails

### Stagnation
- Increase `--pop` for more exploration
- Try different mutation strategies
- Use `--scratch` to restart with fresh random population
