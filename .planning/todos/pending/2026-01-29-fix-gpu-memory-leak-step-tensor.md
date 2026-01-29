---
created: 2026-01-29T00:00
title: Fix GPU memory leak - step-level tensor accumulation
area: core
priority: critical
files:
  - swarm_rag_module/swarm_rag/core/swarm_retriever.py:1092-1094
---

## Problem

`id_to_idx_tensor` (size: `_max_node_id + 1`) is created fresh every step inside `_step_agents_batched` but never explicitly deleted, causing memory fragmentation over long traversals.

This is one of 6 critical GPU memory leaks causing slowdown over generations.

## Solution

Pre-allocate `id_to_idx_tensor` once per query and reuse with `.fill_(-1)` each step.
