---
created: 2026-01-29T00:00
completed: 2026-01-29
title: Make hardcoded magic numbers configurable
area: evolution
priority: medium
status: done
files:
  - swarm_rag_module/swarm_rag/evolution/execution/evaluator.py:38
  - swarm_rag_module/swarm_rag/evolution/execution/weighted_sum.py:261-265
  - swarm_rag_module/swarm_rag/core/swarm_retriever.py:20
---

## Problem

Multiple hardcoded magic numbers throughout the codebase:

1. `DEFAULT_EARLY_EXIT_THRESHOLD: float = 0.30` - module-level constant that should be configurable per-run
2. Mutation probabilities (`PROB_WEIGHT = 0.60`, `PROB_BIAS = 0.15`, etc.) - should be configurable for different evolution strategies
3. `max_samples_per_section: int = 1000` - limits profiling granularity without user control

## Solution

- Move early exit threshold to `ResourceConfig` and pass through constructor
- Move mutation probabilities to `GeneticConfig` dataclass
- Allow profiler max samples override via environment variable or constructor

## Resolution

### Changes Made

1. **Early exit threshold** (already in `ResourceConfig.early_exit_threshold`)
   - Marked `DEFAULT_EARLY_EXIT_THRESHOLD` as deprecated with docstring notice
   - Kept constant for backward compatibility with existing tests/imports

2. **Mutation probabilities** added to `GeneticConfig`:
   - `mutation_prob_weight: float = 0.60`
   - `mutation_prob_bias: float = 0.15`
   - `mutation_prob_ratio: float = 0.10`
   - `mutation_prob_hyperparam: float = 0.10`
   - `mutation_prob_group_change: float = 0.05`

3. **WeightedSumMutator** updated:
   - Constructor now accepts probability overrides
   - Replaced class constants with instance variables
   - `self_adaptive_es_mutation` passes config values to mutator

4. **Profiler max samples** added to `ResourceConfig`:
   - `profiler_max_samples: int = 1000`
   - `SwarmRetriever.__init__` accepts `profiler_max_samples` parameter
   - Priority: constructor param > env var `SWARM_PROFILE_SAMPLES` > default 1000

### Commits

- `db4e0c4`: feat(evolution): add configurable mutation probabilities and profiler settings
- `6f28411`: feat(evolution): make WeightedSumMutator use configurable probabilities
- `d3d617c`: feat(core): make profiler max_samples configurable via constructor
- `83544bb`: docs(evolution): mark DEFAULT_EARLY_EXIT_THRESHOLD as deprecated
