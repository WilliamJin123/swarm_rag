---
created: 2026-01-29T00:00
title: Eliminate duplicate code definitions
area: core
priority: medium
files:
  - swarm_rag_module/swarm_rag/evolution/seed_genomes.py:22-138
  - swarm_rag_module/swarm_rag/evolution/execution/weighted_sum.py:497-627
  - swarm_rag_module/swarm_rag/core/swarm_retriever.py:81-103
  - swarm_rag_module/swarm_rag/evolution/types/genome.py:54-61
  - swarm_rag_module/swarm_rag/evolution/types/config.py:36-39
  - swarm_rag_module/swarm_rag/utils/device.py
  - swarm_rag_module/swarm_rag/evolution/execution/evaluator.py:129-134
---

## Problem

Three areas of code duplication:

1. **Duplicate seed genome definitions:** Expression tree format in `seed_genomes.py` and weighted sum format in `weighted_sum.py` - two completely separate definitions for the same strategies

2. **Duplicate default parameter definitions:** `_DEFAULT_PARAMS` in `swarm_retriever.py` and `DEFAULT_PARAMS` in `genome.py` - same defaults defined twice with risk of divergence

3. **Duplicate device resolution logic:** Device detection repeated in `config.py`, `device.py`, and `evaluator.py` with subtle differences

## Solution

- Create single source of seed configurations and convert to appropriate format at runtime
- Single authoritative source for defaults in swarm_retriever, import where needed elsewhere
- Use single `get_device()` from utils everywhere
