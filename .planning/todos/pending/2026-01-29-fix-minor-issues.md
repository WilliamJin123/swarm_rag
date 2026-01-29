---
created: 2026-01-29T00:00
title: Fix minor issues - logging, types, error handling
area: core
priority: low
files:
  - swarm_rag_module/swarm_rag/core/swarm_retriever.py:903-904
  - swarm_rag_module/swarm_rag/core/swarm_retriever.py:447-448
  - swarm_rag_module/swarm_rag/evolution/map_elites/loop.py:146
  - swarm_rag_module/swarm_rag/evolution/execution/evaluator.py:1326
  - swarm_rag_module/swarm_rag/evolution/map_elites/loop.py:124
---

## Problem

Five minor issues:

1. **Logger spam in hot path:** `logger.debug(f"Agent {agent_id} at {current_loc}...")` called every other step per agent (12,500 log calls per genome at DEBUG level)

2. **Progress logging uses magic interval:** `if (i + 1) % 10 == 0` hardcodes progress log interval

3. **Type annotation missing self-reference:** Uses string quote for forward reference but imports inside method

4. **Inconsistent error handling:** GPU metric failures silently fall back to CPU without logging the actual error

5. **Thread pool not bounded by available cores:** `ThreadPoolExecutor(max_workers=max_workers)` doesn't check against `os.cpu_count()`

## Solution

- Move hot path logging to trace level or remove
- Use tqdm for progress logging
- Use `from __future__ import annotations` at module level
- GPU failures should fail explicitly without silent CPU fallback (cpu<->gpu overhead too much anyway)
- Use `min(max_workers, os.cpu_count() or 4)`
