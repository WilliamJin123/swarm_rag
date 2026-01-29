---
created: 2026-01-29T00:00
title: Fix GPU memory leak - empty_cache called too late
area: evolution
priority: critical
files:
  - swarm_rag_module/swarm_rag/evolution/execution/evaluator.py:341
  - swarm_rag_module/swarm_rag/evolution/execution/evaluator.py:429
---

## Problem

`torch.cuda.empty_cache()` is called after each genome in `_evaluate_all_with_shared`, but the real memory hogs (intermediate metric tensors) accumulate inside `_batch_compute_metrics_all_genomes` before this call.

This is one of 6 critical GPU memory leaks causing slowdown over generations.

## Solution

Move `empty_cache()` inside `_batch_compute_metrics_all_genomes` after each genome's metrics are computed.
