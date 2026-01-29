---
created: 2026-01-29T00:00
title: Fix offspring counter never reset between generations
area: evolution
priority: high
files:
  - swarm_rag_module/swarm_rag/evolution/map_elites/loop.py:35
---

## Problem

`self._offspring_counter` increments indefinitely across generations, causing genome IDs like `g999_c50000` instead of `g999_c1`.

## Solution

Reset counter at the start of each generation's `step()` method.
