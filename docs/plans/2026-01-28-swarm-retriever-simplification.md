# SwarmRetriever Simplification Design

## Overview

Refactor SwarmRetriever from 2,357 lines with 10+ overlapping retrieval methods down to a clean builder-pattern API with tensor-first internals. All 4 execution combinations (GPU/CPU × batched/sequential) share the same code path.

## Current Problems

1. **API explosion**: 4 public entry points (`retrieve`, `retrieve_batch`, `retrieve_with_precomputed`, `retrieve_batch_with_precomputed`) with 6 internal execution paths
2. **Code duplication**: ~150 lines of nearly identical traversal loop logic duplicated across methods
3. **Branching complexity**: Separate code paths for GPU vs CPU, batched vs sequential, precomputed vs not
4. **Hard-coded values**: Heuristic weights (0.3/0.4/0.3) scattered across multiple stepping functions

## New API Design

### Initialization

```python
retriever = SwarmRetriever(
    vector_store=vs,
    graph_store=gs,
    embedding_provider=ep,
    # Auto-detects: cuda > mps > cpu
    default_config={
        "n_agents": 25,
        "steps": 5,
        "decay": 0.5,
        "initial_pool_size": 30,
        "start_subset": 10,
        "top_k": 20,
        "movement_strategies": {...},
        "deposit_strategies": {...},
        "ranking_strategies": {...},
    }
)
```

### Builder Pattern

```python
# Simple single query
results = retriever.query("what is X?").run()

# Batch of queries (auto-detected from list input)
results = retriever.query(["q1", "q2", "q3"]).run()

# Query by ID (int input)
results = retriever.query(42).run()

# Precomputed embedding (Tensor input)
results = retriever.query(embedding_tensor).run()

# Precomputed embedding + initial pool (skips embedding AND initial search)
results = retriever.query(embedding_tensor, pool=initial_pool).run()

# Device override
results = retriever.query(queries).on("cpu").run()

# Execution mode override (default: sequential)
results = retriever.query(queries).run(mode="batched", batch_size=32)

# Config overrides
results = retriever.query(queries).run(n_agents=50, steps=10)

# Evolution genome evaluation - compiled dict unpacks directly
compiled = genome_compiler.compile(genome)
results = retriever.query(queries).run(**compiled, mode="batched", batch_size=64)
```

### Return Type

- Single query input → `List[Dict]` (list of retrieved documents)
- List query input → `List[List[Dict]]` (list of results per query)

## Internal Architecture

### Tensor-First Principle

All execution paths use the same tensor-based implementation:

| Mode | Device | Implementation |
|------|--------|----------------|
| batched | cuda | Tensors on CUDA, batch_size=64 (configurable) |
| batched | cpu | Tensors on CPU, batch_size=64 (configurable) |
| sequential | cuda | Tensors on CUDA, batch_size=1 |
| sequential | cpu | Tensors on CPU, batch_size=1 |

**Key insight**: "Sequential" is just `batch_size=1`. No separate code path needed.

### Core Components

```
SwarmRetriever
├── query(input, pool=None) → QueryBuilder
├── _resolve_input(input, pool) → (embeddings: Tensor, pools: Optional[Tensor], is_batch: bool)
├── _traverse(embeddings, pools, config, device, batch_size) → results
└── default_config: Dict

QueryBuilder
├── _retriever: SwarmRetriever
├── _input: Union[str, int, Tensor, List]
├── _pool: Optional[Tensor]
├── _device: Optional[str]  # None = auto-detect
├── on(device) → self
└── run(mode="sequential", batch_size=64, **overrides) → results
```

### Single Traversal Function

```python
def _traverse(
    self,
    query_embeddings: Tensor,  # (n_queries, embed_dim)
    initial_pools: Optional[Tensor],  # (n_queries, pool_size) or None
    config: Dict,
    device: str,
    batch_size: int,  # 1 for sequential, 64+ for batched
) -> List[List[Dict]]:
    """
    Unified traversal for all execution modes.

    Processes queries in chunks of batch_size.
    When batch_size=1, this is "sequential" mode.
    When batch_size>1, this is "batched" mode.
    Device determines where tensors live (cuda/mps/cpu).
    """
    n_queries = query_embeddings.shape[0]
    all_results = []

    for chunk_start in range(0, n_queries, batch_size):
        chunk_end = min(chunk_start + batch_size, n_queries)
        chunk_embeddings = query_embeddings[chunk_start:chunk_end]
        chunk_pools = initial_pools[chunk_start:chunk_end] if initial_pools is not None else None

        # Initialize state for this chunk
        state = self._init_state(chunk_embeddings, chunk_pools, config, device)

        # Run traversal steps
        for step in range(config["steps"]):
            state = self._step(state, config)

        # Rank and collect results
        chunk_results = self._rank_and_collect(state, config)
        all_results.extend(chunk_results)

    return all_results
```

### State Object

```python
@dataclass
class TraversalState:
    """Batched state for all queries in a chunk."""
    query_embeddings: Tensor      # (batch, embed_dim)
    agent_positions: Tensor       # (batch, n_agents) - current node IDs
    visit_history: Tensor         # (batch, n_agents, steps+1) - all visited nodes
    pheromones: Tensor            # (batch, n_nodes) - pheromone levels per query
    step: int                     # current step number
    device: str
```

### Unified Step Function

```python
def _step(self, state: TraversalState, config: Dict) -> TraversalState:
    """
    One traversal step for all queries × all agents.

    Fully batched tensor operations:
    1. Get neighbors for all agent positions
    2. Compute similarities for all neighbors
    3. Score all neighbors (semantic + centrality + repulsion)
    4. Sample next positions via multinomial
    5. Update pheromones
    6. Apply decay
    """
    batch_size, n_agents = state.agent_positions.shape

    # 1. Get neighbors: (batch, n_agents, max_degree)
    neighbors, neighbor_mask = self._get_neighbors_batched(state.agent_positions)

    # 2. Get neighbor embeddings and compute similarities
    neighbor_embeds = self._get_embeddings_batched(neighbors)  # (batch, n_agents, max_degree, embed_dim)
    similarities = torch.einsum('bd,bnad->bna', state.query_embeddings, neighbor_embeds)

    # 3. Score neighbors
    scores = self._compute_scores(similarities, neighbors, state.pheromones, config)
    scores = scores.masked_fill(~neighbor_mask, 0.0)

    # 4. Sample next positions
    probs = F.softmax(scores, dim=-1)
    sampled_idx = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(batch_size, n_agents)
    new_positions = neighbors.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)

    # 5. Compute and apply deposits
    deposits = self._compute_deposits(new_positions, similarities, config)
    state.pheromones.scatter_add_(1, new_positions, deposits)

    # 6. Decay pheromones
    state.pheromones *= config["decay"]

    # 7. Update state
    state.agent_positions = new_positions
    state.visit_history[:, :, state.step + 1] = new_positions
    state.step += 1

    return state
```

## Input Resolution

```python
def _resolve_input(
    self,
    input: Union[str, int, Tensor, List],
    pool: Optional[Tensor]
) -> Tuple[Tensor, Optional[Tensor], bool]:
    """
    Normalize any input type to (embeddings, pools, is_batch).

    Input types:
    - str: Embed the text → (1, embed_dim)
    - int: Look up query ID embedding → (1, embed_dim)
    - Tensor (1D): Single embedding → (1, embed_dim)
    - Tensor (2D): Batch of embeddings → (n, embed_dim)
    - List[str]: Batch embed texts → (n, embed_dim)
    - List[int]: Batch lookup IDs → (n, embed_dim)
    - List[Tensor]: Stack embeddings → (n, embed_dim)

    Pool handling:
    - None: Will compute initial pools via similarity search
    - Tensor: Use provided pools (skips initial search)
    """
    is_batch = isinstance(input, list) or (isinstance(input, Tensor) and input.dim() == 2)

    # Normalize to list
    inputs = input if isinstance(input, list) else [input]

    # Resolve each input to embedding
    embeddings = []
    for inp in inputs:
        if isinstance(inp, str):
            emb = self.embedding_provider.embed(inp)
        elif isinstance(inp, int):
            emb = self.embedding_provider.get_query_embedding(inp)
        elif isinstance(inp, Tensor):
            emb = inp.flatten()
        embeddings.append(emb)

    embeddings = torch.stack(embeddings)  # (n, embed_dim)

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=-1)

    return embeddings, pool, is_batch
```

## Migration Guide

### Old API → New API

```python
# Old: Single query
results = retriever.retrieve(query="what is X?", n_agents=25, steps=5)
# New:
results = retriever.query("what is X?").run(n_agents=25, steps=5)

# Old: Batch queries
results = retriever.retrieve_batch(queries=["q1", "q2"], max_workers=4)
# New:
results = retriever.query(["q1", "q2"]).run(mode="batched")

# Old: Precomputed single
results = retriever.retrieve_with_precomputed(embedding, initial_pool, n_agents=25)
# New:
results = retriever.query(embedding, pool=initial_pool).run(n_agents=25)

# Old: Precomputed batch
results = retriever.retrieve_batch_with_precomputed(embeddings, pools, batch_size=32)
# New:
results = retriever.query(embeddings, pool=pools).run(mode="batched", batch_size=32)
```

### Evolution Integration

```python
# Old (via adapter)
adapter = SwarmRetrieverAdapter(retriever)
compiled = genome_compiler.compile(genome)
results = adapter.retrieve_batch(queries, compiled, max_workers=4)

# New (direct)
compiled = genome_compiler.compile(genome)
results = retriever.query(queries).run(**compiled, mode="batched")
```

The `SwarmRetrieverAdapter` can be simplified to a thin compatibility layer or deprecated entirely.

## Files to Modify

1. **swarm_retriever.py** - Complete rewrite (~2357 → ~800 lines estimated)
2. **swarm_adapter.py** - Simplify or deprecate
3. **test_n_q.py** - Update to new API
4. **evolve_stark.py** - Update to new API (may just work if adapter updated)
5. **benchmark_multi_query.py** - Update to new API
6. **test_multi_query_batch.py** - Update to new API

## Code Reduction Summary

| Component | Current | After | Reduction |
|-----------|---------|-------|-----------|
| Public methods | 4 | 1 (+ QueryBuilder) | 75% |
| Traversal loops | 2 (duplicated) | 1 | 50% |
| Step functions | 4 (`_process_agent_step`, `_step_agents_batched`, `_step_agents_sequential_fallback`, `_step_multi_query`) | 1 | 75% |
| Batch dispatch logic | 3 paths | 1 loop with batch_size | 66% |
| Total lines | ~2357 | ~800 | ~66% |

## Risks and Mitigations

1. **Performance regression** - Tensor ops on CPU may be slower than pure Python for small batches
   - Mitigation: Benchmark before/after, optimize hot paths if needed

2. **Breaking changes** - All downstream code needs updates
   - Mitigation: Clear migration guide, update all known usages

3. **Edge cases** - Current code handles many edge cases (empty pools, invalid neighbors, etc.)
   - Mitigation: Port all edge case handling to new implementation, comprehensive tests

## Implementation Order

1. Create `QueryBuilder` class with new API
2. Implement `_resolve_input()`
3. Implement `TraversalState` dataclass
4. Implement unified `_traverse()` and `_step()`
5. Wire up `QueryBuilder.run()` to `_traverse()`
6. Update tests to new API
7. Update downstream code (test_n_q.py, evolve_stark.py, etc.)
8. Remove old methods
9. Final cleanup and documentation
