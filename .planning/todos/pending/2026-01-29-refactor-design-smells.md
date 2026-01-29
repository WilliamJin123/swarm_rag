---
created: 2026-01-29T00:00
title: Refactor design smells and architecture issues
area: evolution
priority: medium
files:
  - swarm_rag_module/swarm_rag/evolution/types/config.py:603-664
  - swarm_rag_module/swarm_rag/evolution/execution/evaluator.py:78-97
  - swarm_rag_module/swarm_rag/evolution/types/config.py:51-59
  - swarm_rag_module/swarm_rag/evolution/execution/shared_precompute.py:21-48
  - swarm_rag_module/swarm_rag/evolution/execution/fitness.py:80-101
  - swarm_rag_module/swarm_rag/evolution/types/genome.py:77
---

## Problem

Six design smells identified:

1. **EvolutionContext mixes config and state:** Holds both immutable config and mutable runtime state (generation, stagnation_count, etc.)

2. **PopulationEvaluator has 17 constructor parameters:** Too many parameters indicates the class is doing too much

3. **WeightTensors default factory calls get_device():** Triggers CUDA initialization at import time

4. **SharedPrecomputeContext has nullable required fields:** `ground_truth_tensor` and `gt_sizes` are `Optional` but required for GPU-accelerated metrics

5. **FitnessCalculator has two init patterns:** Can init with either `weights` or `config`, creating divergent code paths

6. **Genome mode as string literal:** Uses string literals instead of enum

## Solution

- Split `EvolutionContext` into `EvolutionConfig` (immutable) and `EvolutionState` (mutable)
- Use builder pattern for `PopulationEvaluator`
- Use `"cpu"` as default device in dataclass, move to target device explicitly when used
- Create two classes: `SharedPrecomputeContextCPU` and `SharedPrecomputeContextGPU`
- Single constructor taking `FitnessConfig`, remove legacy weights and usage
- Create `GenomeMode` enum for type safety and IDE support
