---
created: 2026-01-29T00:00
title: Fix GPU memory leak - random jitter tensor every step
area: core
priority: critical
files:
  - swarm_rag_module/swarm_rag/core/swarm_retriever.py:1147
---

## Problem

Feature registry creates `"random_jitter": torch.rand_like(semantic_scores) * 0.1` which creates new GPU tensors every step without cleanup, accumulating memory.

This is one of 6 critical GPU memory leaks causing slowdown over generations.

## Solution

Pre-allocate jitter buffer once per query and use `torch.rand_like(_, out=buffer)`.
