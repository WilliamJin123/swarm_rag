# Plan: Tensor-Native SwarmRetriever

## Summary

Eliminate all remaining CPU-GPU transfers in SwarmRetriever's hot paths by:
1. Replacing `List[List[int]]` trajectories with a position history tensor
2. Making ranking functions batched/vectorized
3. Using dense pheromone lookup tensors (cached max_node_id)
4. Consolidating fetch methods into a single interface

**Goal:** Zero `.tolist()` or `.item()` calls in the per-step hot loop.

---

## Current State: 26 CPU-GPU Transfer Points

| Category | Count | Impact |
|----------|-------|--------|
| Trajectory updates (`.item()` per agent per step) | 12 | CRITICAL |
| Pheromone dict building (`.tolist()`) | 3 | HIGH |
| Search result processing (`.tolist()`) | 4 | MEDIUM |
| Fetch cache lookups (`.tolist()`) | 2 | MEDIUM |
| Misc (bounds checking, constants) | 5 | LOW |

---

## Design

### 1. Position History Tensor System

**Replace:**
```python
agent_trajectories = [[loc.item()] for loc in agent_locations]  # List[List[int]]
# Per step, per agent:
agent_trajectories[agent_idx].append(new_loc.item())
```

**With:**
```python
# Shape: (n_agents, max_steps + 1), initialized to -1
position_history = torch.full((n_agents, steps + 1), -1, device=device, dtype=torch.long)
position_history[:, 0] = agent_locations

# Per step (batched):
position_history[:, step + 1] = new_locations
agent_locations = new_locations  # Pure tensor assignment
```

**Ranking integration:**
```python
# Compute visit counts at ranking time via torch.unique
all_positions = position_history.flatten()
valid_positions = all_positions[all_positions >= 0]
unique_visited, visit_counts = torch.unique(valid_positions, return_counts=True)
```

**Benefits:**
- Eliminates 12+ `.item()` calls per step
- `torch.unique` is fast and stays on GPU
- Position history is always small: `n_agents * (steps + 1)` ~ 150 entries

---

### 2. Batched Ranking System

**Replace per-node loop:**
```python
for i, node_id in enumerate(valid_ids):
    vec = vectors_matrix[i]
    score = self._calculate_node_score(node_id, votes, query_vec, vec, ranking_func, n_agents)
    results.append({'id': node_id, 'score': score})
```

**With single batched call:**
```python
ctx = HeuristicContext(
    query_vec=query_vec,
    target_vecs=embeddings,        # (n_visited, dim)
    target_ids=unique_visited,     # tensor
    votes=visit_counts,            # tensor
    total_agents=n_agents,
    graph=self.graph_store
)
scores = ranking_func(ctx)  # Returns (n_visited,) tensor

# Top-k on GPU
top_scores, top_indices = torch.topk(scores, k=top_k)
top_ids = unique_visited[top_indices]
```

---

### 3. Heuristics Changes

#### HeuristicContext (heuristics.py:111-133)

```python
@dataclass(slots=True)
class HeuristicContext:
    query_vec: torch.Tensor
    target_vecs: Optional[torch.Tensor] = None
    target_ids: Optional[torch.Tensor] = None  # CHANGE: Always tensor now

    pheromone_values: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    node_degrees: torch.Tensor = field(default_factory=lambda: torch.tensor([]))

    graph: Optional[GraphStore] = None
    max_pheromone: float = 1.0
    avg_degree: float = 1.0
    step_index: int = 0
    agent_index: int = 0
    votes: Optional[torch.Tensor] = None  # CHANGE: Tensor for batched ranking
    total_agents: int = 0

    extra_data: Dict[str, Any] = field(default_factory=dict)
```

#### percentage_visited (heuristics.py:216-220)

**Before:**
```python
def percentage_visited(ctx: HeuristicContext) -> float:
    if ctx.total_agents == 0: return 0.0
    return ctx.votes / ctx.total_agents
```

**After:**
```python
def percentage_visited(ctx: HeuristicContext) -> torch.Tensor:
    if ctx.total_agents == 0:
        return torch.zeros_like(ctx.votes, dtype=torch.float32) if ctx.votes is not None else torch.tensor(0.0)
    return ctx.votes.float() / ctx.total_agents
```

#### semantic_rank (heuristics.py:223-232)

**Before:**
```python
def semantic_rank(ctx: HeuristicContext) -> float:
    val = Heuristics.semantic_similarity_unnormalized(ctx)
    if hasattr(val, 'item'):
        return val.item()
    return float(val)
```

**After:**
```python
def semantic_rank(ctx: HeuristicContext) -> torch.Tensor:
    return Heuristics.semantic_similarity_unnormalized(ctx)
```

#### random_jitter (heuristics.py:204-211)

**Before:**
```python
count = len(ctx.target_ids) if ctx.target_ids is not None else 1
```

**After:**
```python
count = ctx.target_vecs.shape[0] if ctx.target_vecs is not None else 1
```

---

### 4. Dense Pheromone Lookup

**Cache max_node_id at init:**
```python
def __init__(self, ...):
    # ...
    self._max_node_id = self.graph_store.n_nodes
```

**Pheromone lookup helper:**
```python
def _build_pheromone_tensor(
    self,
    query_pheromones: Dict[int, float],
    device: torch.device
) -> torch.Tensor:
    """Build dense pheromone lookup tensor. O(1) indexing, ~500KB for 129K nodes."""
    pheromone_lookup = torch.zeros(self._max_node_id, device=device, dtype=torch.float32)
    if query_pheromones:
        pher_ids = torch.tensor(list(query_pheromones.keys()), device=device, dtype=torch.long)
        pher_vals = torch.tensor(list(query_pheromones.values()), device=device, dtype=torch.float32)
        pheromone_lookup[pher_ids] = pher_vals
    return pheromone_lookup
```

**Usage in hot path:**
```python
# Once per step (not per agent):
pheromone_tensor = self._build_pheromone_tensor(query_pheromones, device)

# For any candidate lookup:
p_vals = pheromone_tensor[candidate_ids]  # Pure tensor indexing
```

---

### 5. Unified Fetch Interface

**Consolidate these methods:**
- `_fetch_vectors_batch()` - cache-aware fetch
- `_fetch_vectors_batch_gpu()` - GPU wrapper

**Into single method:**
```python
def _fetch_embeddings(
    self,
    node_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified embedding fetch.

    Args:
        node_ids: Tensor of node IDs to fetch

    Returns:
        (embeddings, valid_ids) - both tensors on device
        Only returns rows for valid IDs (no NaN rows).
    """
    if not self.cache_vectors:
        # Direct fetch from store
        matrix = self.vector_store.fetch_batch(node_ids)
        valid_mask = ~torch.isnan(matrix).any(dim=1)
        return matrix[valid_mask], node_ids[valid_mask]

    # Cache-aware fetch (existing two-phase logic)
    # But keep node_ids as tensor throughout
    node_ids_list = node_ids.tolist()  # Only for dict key lookup
    # ... cache logic ...
    return embeddings_tensor, valid_ids_tensor
```

---

### 6. SwarmRetriever Method Changes

#### _retrieve() / _retrieve_with_pool_internal()

**Changes:**
1. Initialize `position_history` tensor instead of `agent_trajectories` list
2. Remove per-agent `.item()` loops in step updates
3. Call new `_ranking_from_history()` instead of `_ranking()`

#### New: _ranking_from_history()

```python
def _ranking_from_history(
    self,
    position_history: torch.Tensor,
    query_vec: torch.Tensor,
    ranking_func: Callable,
    top_k: int,
    n_agents: int
) -> List[Dict]:
    # Flatten and get unique visited nodes
    all_positions = position_history.flatten()
    valid_positions = all_positions[all_positions >= 0]
    unique_visited, visit_counts = torch.unique(valid_positions, return_counts=True)

    if unique_visited.numel() == 0:
        return []

    # Fetch embeddings
    embeddings, valid_ids = self._fetch_embeddings(unique_visited)
    visit_counts = visit_counts[torch.isin(unique_visited, valid_ids)]

    # Batched ranking context
    query_vec = query_vec.to(embeddings.device)
    ctx = HeuristicContext(
        query_vec=query_vec,
        target_vecs=embeddings,
        target_ids=valid_ids,
        votes=visit_counts,
        total_agents=n_agents,
        graph=self.graph_store
    )

    scores = ranking_func(ctx)

    # Top-k selection on GPU
    k = min(top_k, scores.numel())
    top_scores, top_indices = torch.topk(scores, k=k)
    top_ids = valid_ids[top_indices]

    # Convert at API boundary only
    return [
        {'id': int(nid), 'score': float(sc)}
        for nid, sc in zip(top_ids.tolist(), top_scores.tolist())
    ]
```

#### _step_agents_batched()

**Changes:**
1. Build pheromone tensor once per step using `_build_pheromone_tensor()`
2. Remove `.tolist()` when building pheromone updates dict (keep at boundary)
3. Return `new_locations` tensor directly

#### _process_agent_step()

**Changes:**
1. Accept pheromone tensor instead of dict
2. Use tensor indexing for pheromone lookup
3. Remove all `.item()` calls

#### combined_ranking() helper

**Changes:**
Handle tensor returns from ranking functions:
```python
def combined_ranking(ctx):
    total = torch.zeros(ctx.target_vecs.shape[0], device=ctx.target_vecs.device)
    for func, w in resolved_strategies:
        val = func(ctx)
        total += val * w
    return total
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `swarm_rag/core/heuristics.py` | HeuristicContext fields, ranking heuristics return tensors |
| `swarm_rag/core/swarm_retriever.py` | Position history, batched ranking, pheromone tensor, unified fetch |

---

## Methods to Update in swarm_retriever.py

| Method | Change |
|--------|--------|
| `__init__` | Cache `self._max_node_id` from graph store |
| `_retrieve` | Position history tensor, remove trajectory list |
| `_retrieve_with_pool_internal` | Same as above |
| `_step_agents_batched` | Pheromone tensor, remove `.tolist()` in updates |
| `_step_agents_sequential_fallback` | Position history compatible |
| `_process_agent_step` | Accept pheromone tensor, tensor indexing |
| `_ranking` | Replace with `_ranking_from_history` |
| `_ranking_vectorized` | Merge into `_ranking_from_history` |
| `_fetch_vectors_batch` | Unify with GPU version |
| `_fetch_vectors_batch_gpu` | Remove (merged) |
| `combined_ranking` (local) | Handle tensor returns |

---

## Methods to Remove

| Method | Reason |
|--------|--------|
| `_fetch_vectors_batch_gpu` | Merged into `_fetch_embeddings` |
| `_ranking` | Replaced by `_ranking_from_history` |
| `_ranking_vectorized` | Merged into `_ranking_from_history` |

---

## Duplicate Code Cleanup

The following methods have near-identical implementations and should be unified:

| Keep | Remove/Merge |
|------|--------------|
| `_retrieve` | `_retrieve_with_pool_internal` (same logic, different entry) |
| `_step_agents_batched` | Remove duplicate in lines 2000+ |
| `_ranking_from_history` | `_ranking`, `_ranking_vectorized` |

---

## Verification

1. **Unit tests:**
   ```bash
   cd swarm_rag_module && python -m pytest tests/unit/test_gpu_utils.py -v
   ```

2. **Integration test (CPU):**
   ```bash
   cd stark && python test_n_q.py -n 5 --device cpu
   ```

3. **Integration test (GPU):**
   ```bash
   cd stark && python test_n_q.py -n 5 --device gpu
   ```

4. **Performance comparison:**
   - Before: ~0.098s/query (GPU)
   - Target: <0.05s/query (50%+ improvement from eliminating transfers)

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Ranking function compatibility | All built-in heuristics updated; custom functions must return tensors |
| Memory for position_history | Tiny: 25 agents * 6 steps * 8 bytes = 1.2KB |
| Memory for pheromone tensor | Acceptable: 129K * 4 bytes = 500KB |
| Breaking API changes | Only internal methods change; public `retrieve()` unchanged |

---

## Implementation Order

1. **heuristics.py** - Update HeuristicContext and ranking heuristics (no dependencies)
2. **swarm_retriever.py** - Add helper methods (`_build_pheromone_tensor`, `_fetch_embeddings`)
3. **swarm_retriever.py** - Add `_ranking_from_history`
4. **swarm_retriever.py** - Update `_retrieve` to use position history
5. **swarm_retriever.py** - Update `_step_agents_batched` for pheromone tensor
6. **swarm_retriever.py** - Remove deprecated methods
7. **Tests** - Verify all pass
8. **Cleanup** - Remove duplicate code blocks
