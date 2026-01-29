---
created: 2026-01-29T00:00
title: Fix GPU memory leak - BatchedRetrievalResults.clear()
area: evolution
priority: critical
files:
  - swarm_rag_module/swarm_rag/evolution/execution/shared_precompute.py:334-340
---

## Problem

`BatchedRetrievalResults.clear()` deletes Python reference but doesn't call `torch.cuda.empty_cache()`, leaving memory in CUDA cache.

This is one of 6 critical GPU memory leaks causing slowdown over generations.

## Solution

Add `if torch.cuda.is_available(): torch.cuda.empty_cache()` after deletion.
