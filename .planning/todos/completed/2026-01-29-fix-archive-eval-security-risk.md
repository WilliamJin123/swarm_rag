---
created: 2026-01-29T00:00
title: Fix archive uses eval() for checkpoint restoration
area: evolution
priority: high
files:
  - swarm_rag_module/swarm_rag/evolution/map_elites/archive.py:408
---

## Problem

`key = eval(key_str)` is a security risk and can crash on malformed input.

## Solution

Use `ast.literal_eval(key_str)` instead.
