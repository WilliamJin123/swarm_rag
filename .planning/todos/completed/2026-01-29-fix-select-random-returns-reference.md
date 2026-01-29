---
created: 2026-01-29T00:00
title: Fix select_random returns stored reference
area: evolution
priority: high
files:
  - swarm_rag_module/swarm_rag/evolution/map_elites/archive.py:286-291
---

## Problem

`select_random()` returns `self.grid[key]` directly, but `add()` stores copies. If caller mutates returned genome, archive integrity is compromised.

## Solution

Return `self.grid[key].copy()` to maintain immutability guarantee.
