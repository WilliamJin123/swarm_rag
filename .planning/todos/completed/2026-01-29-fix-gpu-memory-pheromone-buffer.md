---
created: 2026-01-29T00:00
title: Fix GPU memory leak - pheromone buffer never shrinks
area: core
priority: critical
files:
  - swarm_rag_module/swarm_rag/core/swarm_retriever.py:160
---

## Problem

`_pheromone_buffer_size = max(self._max_node_id + 1024, 150000)` hardcodes minimum 150k floats (~600KB per query) regardless of actual graph size.

This is one of 6 critical GPU memory leaks causing slowdown over generations.

## Solution

Set buffer to `_max_node_id + 1` without the arbitrary minimum.
