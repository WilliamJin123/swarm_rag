# SwarmRAG GPU Optimization Summary

## What Was Done

### 1. Bug Fixes (Phase 1)
- **genome.py:294** - Added `if counts:` guard to prevent IndexError with empty counts
- **genome.py:115** - Added `if n == 0: return` safety check in `normalize_ratios()`
- **evaluator.py:270** - Fixed KeyError by checking `if fallback and f"var_{fallback}" in aggregated`

### 2. Test Improvements (Phase 2)
- **test_swarm.py** - Added assertions and made DummyVectorStore deterministic
- **test_boltzmann.py** - Added `random.seed(42)` for determinism
- **test_eval.py** - Added edge case tests for empty lists
- **test_tracker.py** / **test_migration.py** - Added cleanup fixtures

### 3. GPU Integration (Phase 3)
- Installed CUDA-enabled PyTorch (`torch==2.6.0+cu124`)
- Fixed **gpu_vector_store.py:116-127** - Added `.squeeze()` to handle `[1, 1536]` shaped embeddings
- Created unified **test_n_q.py** with `--device`, `--compare`, `--verbose` flags

---

## Current Performance

| Operation | GPU | CPU | Speedup |
|-----------|-----|-----|---------|
| Pure vector search (100 searches) | 5.19ms | 42.43ms | **8.17x** |
| End-to-end retrieval (per query) | 0.71s | 0.80s | **1.12x** |

**Why the gap?** Graph traversal happens on CPU and dominates total time.

---

## The Bottleneck: CPU Graph Traversal

Each query with 25 agents × 5 steps = **125 graph operations**:

```
Agent Step Pipeline:
  1. Get neighbors from graph  → CPU (scipy sparse)  → ~0.1ms/node
  2. Fetch neighbor embeddings → GPU/CPU             → ~0.05ms
  3. Compute similarity scores → GPU (fast)          → ~0.05ms
  4. Random selection          → CPU                 → ~0.01ms
```

The scipy CSR matrix indexing (`adj_matrix[node_id].indices`) runs on CPU and cannot be parallelized across agents.

---

## Next Steps: GPU Optimization

### Priority 1: GPU Graph Store
Move adjacency matrix to GPU as torch sparse tensor.

```python
# swarm_rag/integrations/gpu_graph_store.py

import torch
from scipy.sparse import csr_matrix

class GPUGraphStore:
    def __init__(self, adj_dict: dict, device: str = "cuda"):
        """Convert adjacency dict to GPU sparse tensor."""
        self.device = device

        # Build CSR components
        n_nodes = len(adj_dict)
        indptr = [0]
        indices = []

        for i in range(n_nodes):
            neighbors = adj_dict.get(i, [])
            indices.extend(neighbors)
            indptr.append(len(indices))

        # Create torch sparse CSR tensor on GPU
        self._adj = torch.sparse_csr_tensor(
            torch.tensor(indptr, dtype=torch.int64),
            torch.tensor(indices, dtype=torch.int64),
            torch.ones(len(indices)),  # edge weights = 1
            size=(n_nodes, n_nodes),
            device=device
        )

        # Dense neighbor lists for fast lookup (if memory allows)
        # For 129K nodes with avg degree 125: ~64MB
        self._max_degree = max(len(v) for v in adj_dict.values())
        self._neighbors = torch.full(
            (n_nodes, self._max_degree), -1,
            dtype=torch.int64, device=device
        )
        self._degrees = torch.zeros(n_nodes, dtype=torch.int64, device=device)

        for i, neighbors in adj_dict.items():
            self._degrees[i] = len(neighbors)
            self._neighbors[i, :len(neighbors)] = torch.tensor(neighbors)

    def get_neighbors(self, node_id: int) -> torch.Tensor:
        """Get neighbors for single node."""
        deg = self._degrees[node_id].item()
        return self._neighbors[node_id, :deg]

    def get_neighbors_batch(self, node_ids: torch.Tensor) -> tuple:
        """
        Batch neighbor lookup for multiple nodes.
        Returns (neighbors, mask) where mask indicates valid neighbors.
        """
        # node_ids: (batch_size,)
        # Returns: (batch_size, max_degree), (batch_size, max_degree)
        neighbors = self._neighbors[node_ids]  # (batch, max_degree)
        mask = neighbors != -1
        return neighbors, mask

    def get_degree(self, node_id: int) -> int:
        return self._degrees[node_id].item()

    def get_degrees_batch(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self._degrees[node_ids]
```

**Expected speedup:** 5-10x for neighbor lookups

### Priority 2: Batch Agent Processing
Process all agents in parallel instead of sequentially.

```python
# In swarm_retriever.py

def _step_agents_batched(self, agents: List[Agent], query_emb: torch.Tensor):
    """Process all agents in one batched operation."""

    # 1. Gather all agent positions
    positions = torch.tensor([a.current_node for a in agents], device=self.device)

    # 2. Batch fetch all neighbors
    all_neighbors, mask = self.graph_store.get_neighbors_batch(positions)
    # all_neighbors: (n_agents, max_degree)

    # 3. Batch fetch all neighbor embeddings
    valid_neighbors = all_neighbors[mask]  # Flatten valid neighbors
    neighbor_embs = self._get_embeddings_batch(valid_neighbors)

    # 4. Batch compute similarities
    # query_emb: (1, dim), neighbor_embs: (n_valid, dim)
    similarities = torch.mm(neighbor_embs, query_emb.t()).squeeze()

    # 5. Reshape back and apply heuristics
    # ... scatter similarities back to (n_agents, max_degree) shape

    # 6. Batched selection (sample from categorical distribution)
    probs = torch.softmax(scores, dim=-1)
    next_nodes = torch.multinomial(probs, num_samples=1).squeeze()

    return next_nodes
```

**Expected speedup:** 2-3x by eliminating Python loop overhead

### Priority 3: Embedding Caching on GPU
Keep frequently accessed embeddings in GPU memory.

```python
class GPUEmbeddingCache:
    def __init__(self, all_embeddings: torch.Tensor, cache_size: int = 10000):
        self.all_embeddings = all_embeddings.to('cuda')  # Full matrix on GPU
        # Or use LRU cache for memory-constrained scenarios

    def get_batch(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Direct indexing on GPU - no CPU transfer."""
        return self.all_embeddings[node_ids]
```

### Priority 4: Reduce Steps, Increase Pool
Trade exploration depth for speed.

```python
# Current (slower, more exploration)
config = {"n_agents": 25, "steps": 5, "initial_pool_size": 30}

# Faster (less exploration, larger initial coverage)
config = {"n_agents": 15, "steps": 2, "initial_pool_size": 100}
```

---

## Implementation Order

1. **GPUGraphStore** - Biggest impact, self-contained change
2. **Batch agent processing** - Requires refactoring `_step_agents()`
3. **GPU embedding cache** - Already partially done in `StarkGPUVectorStore`
4. **Hyperparameter tuning** - Easy win, just config changes

---

## Files to Modify

| File | Change |
|------|--------|
| `swarm_rag/integrations/gpu_graph_store.py` | New file - GPU adjacency matrix |
| `swarm_rag/integrations/stark.py` | Add `StarkGPUGraphAdapter` using new store |
| `swarm_rag/core/swarm_retriever.py` | Add `_step_agents_batched()` method |
| `stark/test_n_q.py` | Add `--gpu-graph` flag to test new implementation |

---

## Verification

```bash
# Test GPU graph store
python test_n_q.py --n 20 --compare --seed 42

# Expected after optimization:
# - GPU speedup: 3-5x (up from 1.1x)
# - Latency: ~0.2s/query (down from 0.7s)
```
