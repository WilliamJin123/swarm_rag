---
created: 2026-01-29T00:00
title: Fix GPU memory leak - shared context tensors not released
area: core
priority: critical
files:
  - swarm_rag_module/swarm_rag/evolution/execution/evaluator.py:369-377
---

## Problem

`ground_truth_tensor` and `gt_sizes` tensors in `SharedPrecomputeContext` are only deleted at the end of `_evaluate_all_with_shared`, but expanded copies (`gt_tensor_expanded`, `gt_sizes_expanded`) created in `_batch_compute_metrics_all_genomes` are never explicitly deleted.

This is one of 6 critical GPU memory leaks causing slowdown over generations.

## Solution

Add `del gt_tensor_expanded; del gt_sizes_expanded` after use and call `torch.cuda.empty_cache()` inside the loop.
