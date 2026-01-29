---
created: 2026-01-29T00:00
title: Make hardcoded magic numbers configurable
area: evolution
priority: medium
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
