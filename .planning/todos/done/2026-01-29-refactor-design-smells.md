---
created: 2026-01-29T00:00
completed: 2026-01-29
title: Refactor design smells and architecture issues
area: evolution
priority: medium
status: done
files:
  - swarm_rag_module/swarm_rag/evolution/types/config.py
  - swarm_rag_module/swarm_rag/evolution/execution/evaluator.py
  - swarm_rag_module/swarm_rag/evolution/execution/shared_precompute.py
  - swarm_rag_module/swarm_rag/evolution/execution/fitness.py
  - swarm_rag_module/swarm_rag/evolution/types/genome.py
---

## Problem

Six design smells identified and resolved:

1. **EvolutionContext mixes config and state** - RESOLVED
2. **PopulationEvaluator has 17 constructor parameters** - RESOLVED
3. **WeightTensors default factory calls get_device()** - RESOLVED
4. **SharedPrecomputeContext has nullable required fields** - RESOLVED
5. **FitnessCalculator has two init patterns** - RESOLVED
6. **Genome mode as string literal** - RESOLVED

## Solution Applied

### 1. GenomeMode enum (commit already in repo)
- Created `GenomeMode` enum in config.py
- Replaced string literals with enum values throughout codebase
- Added backward compatibility in `__setstate__` for legacy checkpoints

### 2. WeightTensors CPU default (f975fe8)
- Removed `_get_default_device()` helper that triggered CUDA init
- Default all tensors to CPU device
- Use `to_device()` method to move tensors to GPU when needed

### 3. FitnessCalculator single constructor (92db6aa)
- Primary constructor now takes `FitnessConfig` as main parameter
- Added `from_weights()` class method for simple weight-based init
- Deprecated `weights` parameter with warning
- Maintained backward compatibility

### 4. SharedPrecomputeContext type-safe accessors (5c63ced)
- Added `has_gpu_ground_truth` property for checking availability
- Added `is_gpu_context` property for device checking
- Added `get_gpu_ground_truth()` method with explicit error handling
- Updated docstrings for clarity

### 5. EvolutionState separation (9f84163)
- Created `EvolutionState` dataclass for mutable runtime state
- Added `reset_for_new_run()` method
- Added `state` property to EvolutionContext
- Maintained backward compatibility with top-level field access

### 6. PopulationEvaluatorBuilder (addbec7)
- Created `EvaluatorConfig` dataclass for parameters
- Created `PopulationEvaluatorBuilder` with fluent interface
- Builder validates required dependencies on build()
- Original constructor unchanged for compatibility

## Commits
- f975fe8: WeightTensors CPU default
- 92db6aa: FitnessCalculator single constructor
- 5c63ced: SharedPrecomputeContext type-safe accessors
- 9f84163: EvolutionState separation
- addbec7: PopulationEvaluatorBuilder
