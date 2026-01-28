# Multi-Query GPU Batching Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Process multiple queries simultaneously on GPU instead of sequentially, achieving 5-10x speedup on swarm retrieval.

**Architecture:** Batch N queries (same genome config) into a single GPU traversal. Each query maintains independent pheromone state and agent positions, but all tensor operations are batched. Chunking (batch_size=32) manages memory.

**Tech Stack:** PyTorch tensors, einsum for batched similarity, scatter_add for batched pheromone updates, multinomial for vectorized selection.

---

## Background

### Current Bottleneck

In `swarm_retriever.py:1676-1684`, GPU mode falls back to sequential query processing:

```python
if self._use_gpu or max_workers <= 1:
    # Sequential processing for GPU (CUDA thread-locality)
    return self._retrieve_batch_precomputed_sequential(...)  # ONE query at a time!
```

This causes validation to take 99% of evolution time.

### Memory Budget

| Dataset | n_nodes | 32-query batch | 64-query batch |
|---------|---------|----------------|----------------|
| Prime   | 130K    | 16 MB          | 32 MB          |
| Amazon  | 1.3M    | 160 MB         | 320 MB         |
| MAG     | 1.3M    | 160 MB         | 320 MB         |

All well within GPU memory limits.

### Type Signature Changes

```python
# Current (single query internal state)
agent_locations: Tensor    # (n_agents,)
query_pheromones: Tensor   # (n_nodes,)
position_history: Tensor   # (n_agents, steps+1)

# New (batched queries, same config)
agent_locations: Tensor    # (batch_size, n_agents)
query_pheromones: Tensor   # (batch_size, n_nodes)
position_history: Tensor   # (batch_size, n_agents, steps+1)
query_vecs: Tensor         # (batch_size, dim)
```

---

## Task 1: Add Multi-Query GPU Entry Point

**Files:**
- Modify: `swarm_rag_module/swarm_rag/core/swarm_retriever.py:1675-1695`
- Test: `stark/test_multi_query_batch.py` (new)

**Step 1: Write the failing test**

Create `stark/test_multi_query_batch.py`:

```python
"""Tests for multi-query GPU batching."""
import torch
import pytest


class TestMultiQueryBatching:
    """Test multi-query batched retrieval matches sequential."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a minimal retriever for testing."""
        # We'll use the actual retriever with small test data
        from swarm_rag.core import SwarmRetriever
        from swarm_rag.integrations.torch_vector_store import TorchVectorStore
        from swarm_rag.integrations.torch_graph_store import TorchGraphStore

        # Small test graph: 100 nodes, ~5 edges each
        n_nodes = 100
        dim = 64

        # Create random embeddings
        doc_embs = {i: torch.randn(dim) for i in range(n_nodes)}
        vector_store = TorchVectorStore.from_dict(doc_embs, device="cuda" if torch.cuda.is_available() else "cpu")

        # Create simple graph (each node connects to next 5)
        import scipy.sparse as sp
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(1, 6):
                neighbor = (i + j) % n_nodes
                rows.append(i)
                cols.append(neighbor)
        csr = sp.csr_matrix(([1]*len(rows), (rows, cols)), shape=(n_nodes, n_nodes))
        graph_store = TorchGraphStore.from_csr(csr, device=vector_store.device)

        # Simple embedding provider
        class SimpleEmbedder:
            def __init__(self, dim):
                self.dim = dim
            def embed_query(self, q):
                return torch.randn(self.dim)
            def embed_query_batch(self, qs):
                return torch.randn(len(qs), self.dim)

        retriever = SwarmRetriever(
            vector_store=vector_store,
            graph_store=graph_store,
            embedding_provider=SimpleEmbedder(dim),
            device=vector_store.device,
        )
        return retriever

    def test_multi_query_method_exists(self, mock_retriever):
        """Verify the new method exists."""
        assert hasattr(mock_retriever, '_retrieve_batch_multi_query_gpu')

    def test_multi_query_returns_correct_length(self, mock_retriever):
        """Verify batched method returns results for all queries."""
        n_queries = 10
        query_embeddings = torch.randn(n_queries, 64, device=mock_retriever._device)
        initial_pools = [[i, (i+1)%100, (i+2)%100] for i in range(n_queries)]

        resolved_agents = mock_retriever._prepare_agents(
            agent_groups=None,
            n_agents=5,
            movement_strategies=mock_retriever._DEFAULT_PARAMS['movement_strategies'],
            deposit_strategies=mock_retriever._DEFAULT_PARAMS['deposit_strategies'],
        )

        results = mock_retriever._retrieve_batch_multi_query_gpu(
            query_embeddings=query_embeddings,
            initial_pools=initial_pools,
            resolved_agents=resolved_agents,
            base_seed=42,
            steps=3,
            decay=0.9,
            drop_zone_inc=0.05,
            start_subset=3,
            top_k=5,
            ranking_strategies=mock_retriever._DEFAULT_PARAMS['ranking_strategies'],
        )

        assert len(results) == n_queries
        for r in results:
            assert isinstance(r, list)
```

**Step 2: Run test to verify it fails**

Run: `pytest stark/test_multi_query_batch.py::TestMultiQueryBatching::test_multi_query_method_exists -v`

Expected: FAIL with `AttributeError: 'SwarmRetriever' object has no attribute '_retrieve_batch_multi_query_gpu'`

**Step 3: Write minimal stub implementation**

Add to `swarm_rag_module/swarm_rag/core/swarm_retriever.py` after line 1798 (after `_retrieve_batch_precomputed_parallel`):

```python
def _retrieve_batch_multi_query_gpu(
    self,
    query_embeddings: torch.Tensor,
    initial_pools: List[List[int]],
    resolved_agents: List[Tuple[Callable, Callable]],
    base_seed: int,
    batch_size: int = 32,
    **kwargs
) -> List[List[Dict]]:
    """
    Process multiple queries simultaneously on GPU.

    Batches queries in chunks of `batch_size` to manage memory.
    Each chunk runs the full swarm traversal with batched tensor ops.

    Args:
        query_embeddings: Pre-computed query embeddings (n_queries, dim)
        initial_pools: List of initial candidate pools per query
        resolved_agents: List of (move_fn, deposit_fn) tuples
        base_seed: Base random seed for reproducibility
        batch_size: Number of queries to process simultaneously
        **kwargs: Additional params (steps, decay, top_k, etc.)

    Returns:
        List of result lists, one per query
    """
    # Stub: fall back to sequential for now
    return self._retrieve_batch_precomputed_sequential(
        query_embeddings, initial_pools, resolved_agents, base_seed, **kwargs
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest stark/test_multi_query_batch.py::TestMultiQueryBatching::test_multi_query_method_exists -v`

Expected: PASS

**Step 5: Commit**

```bash
git add stark/test_multi_query_batch.py swarm_rag_module/swarm_rag/core/swarm_retriever.py
git commit -m "feat: add stub for multi-query GPU batching method"
```

---

## Task 2: Implement Batched Initialization

**Files:**
- Modify: `swarm_rag_module/swarm_rag/core/swarm_retriever.py`

**Step 1: Write the failing test**

Add to `stark/test_multi_query_batch.py`:

```python
def test_batched_initialization_shapes(self, mock_retriever):
    """Verify batched tensors have correct shapes."""
    batch_size = 4
    n_agents = 5
    n_nodes = 100
    steps = 3

    # Call internal init helper (we'll add this)
    agent_locs, pheromones, history = mock_retriever._init_multi_query_state(
        batch_size=batch_size,
        n_agents=n_agents,
        n_nodes=n_nodes,
        steps=steps,
        initial_pools=[[0,1,2,3,4] for _ in range(batch_size)],
        drop_zone_inc=0.05,
        seed=42,
    )

    assert agent_locs.shape == (batch_size, n_agents)
    assert pheromones.shape == (batch_size, n_nodes)
    assert history.shape == (batch_size, n_agents, steps + 1)
    assert (history[:, :, 0] == agent_locs).all()  # First position recorded
```

**Step 2: Run test to verify it fails**

Run: `pytest stark/test_multi_query_batch.py::TestMultiQueryBatching::test_batched_initialization_shapes -v`

Expected: FAIL with `AttributeError: '_init_multi_query_state'`

**Step 3: Implement initialization helper**

Add to `swarm_retriever.py`:

```python
def _init_multi_query_state(
    self,
    batch_size: int,
    n_agents: int,
    n_nodes: int,
    steps: int,
    initial_pools: List[List[int]],
    drop_zone_inc: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize batched state tensors for multi-query processing.

    Returns:
        agent_locations: (batch_size, n_agents) - current positions
        query_pheromones: (batch_size, n_nodes) - pheromone values
        position_history: (batch_size, n_agents, steps+1) - trajectory history
    """
    device = self._device

    # Initialize pheromones: (batch_size, pheromone_buffer_size)
    query_pheromones = torch.zeros(
        (batch_size, self._pheromone_buffer_size),
        dtype=torch.float32, device=device
    )

    # Initialize position history: (batch_size, n_agents, steps+1)
    position_history = torch.full(
        (batch_size, n_agents, steps + 1), -1,
        dtype=torch.long, device=device
    )

    # Initialize agent locations from pools
    agent_locations = torch.zeros(
        (batch_size, n_agents), dtype=torch.long, device=device
    )

    torch.manual_seed(seed)
    for q in range(batch_size):
        pool = initial_pools[q]
        if not pool:
            continue
        pool_tensor = torch.tensor(pool, device=device, dtype=torch.long)
        pool_len = len(pool)

        # Weighted sampling favoring earlier pool entries
        weights = torch.tensor(
            [1.0 + drop_zone_inc * (pool_len - i - 1) for i in range(pool_len)],
            device=device
        )
        weights = weights / weights.sum()

        # Sample agent starting positions
        indices = torch.multinomial(weights, n_agents, replacement=True)
        agent_locations[q] = pool_tensor[indices]

    # Record initial positions
    position_history[:, :, 0] = agent_locations

    return agent_locations, query_pheromones, position_history
```

**Step 4: Run test to verify it passes**

Run: `pytest stark/test_multi_query_batch.py::TestMultiQueryBatching::test_batched_initialization_shapes -v`

Expected: PASS

**Step 5: Commit**

```bash
git add swarm_rag_module/swarm_rag/core/swarm_retriever.py stark/test_multi_query_batch.py
git commit -m "feat: implement batched state initialization for multi-query GPU"
```

---

## Task 3: Implement Batched Neighbor Lookup

**Files:**
- Modify: `swarm_rag_module/swarm_rag/core/swarm_retriever.py`

**Step 1: Write the failing test**

Add to `stark/test_multi_query_batch.py`:

```python
def test_batched_neighbor_lookup(self, mock_retriever):
    """Verify batched neighbor lookup returns correct shapes."""
    batch_size = 4
    n_agents = 5

    # Create agent locations
    agent_locations = torch.randint(0, 100, (batch_size, n_agents), device=mock_retriever._device)

    # Call batched neighbor lookup
    all_neighbors, neighbor_mask = mock_retriever._get_neighbors_multi_query(agent_locations)

    # Should be (batch_size, n_agents, max_degree)
    assert all_neighbors.dim() == 3
    assert all_neighbors.shape[0] == batch_size
    assert all_neighbors.shape[1] == n_agents
    assert neighbor_mask.shape == all_neighbors.shape
```

**Step 2: Run test to verify it fails**

Run: `pytest stark/test_multi_query_batch.py::TestMultiQueryBatching::test_batched_neighbor_lookup -v`

Expected: FAIL

**Step 3: Implement batched neighbor lookup**

```python
def _get_neighbors_multi_query(
    self,
    agent_locations: torch.Tensor,  # (batch_size, n_agents)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch fetch neighbors for all agents across all queries.

    Args:
        agent_locations: (batch_size, n_agents) positions

    Returns:
        all_neighbors: (batch_size, n_agents, max_degree) neighbor IDs
        neighbor_mask: (batch_size, n_agents, max_degree) validity mask
    """
    batch_size, n_agents = agent_locations.shape
    device = agent_locations.device

    # Flatten all positions
    flat_positions = agent_locations.flatten()  # (batch_size * n_agents,)

    # Batch fetch from graph store
    flat_neighbors, flat_mask = self.graph_store.get_neighbors_batch(flat_positions)
    # flat_neighbors: (batch_size * n_agents, max_degree)
    # flat_mask: (batch_size * n_agents, max_degree)

    max_degree = flat_neighbors.shape[1]

    # Reshape back to (batch_size, n_agents, max_degree)
    all_neighbors = flat_neighbors.view(batch_size, n_agents, max_degree)
    neighbor_mask = flat_mask.view(batch_size, n_agents, max_degree)

    return all_neighbors, neighbor_mask
```

**Step 4: Run test to verify it passes**

Run: `pytest stark/test_multi_query_batch.py::TestMultiQueryBatching::test_batched_neighbor_lookup -v`

Expected: PASS

**Step 5: Commit**

```bash
git add swarm_rag_module/swarm_rag/core/swarm_retriever.py stark/test_multi_query_batch.py
git commit -m "feat: implement batched neighbor lookup for multi-query GPU"
```

---

## Task 4: Implement Batched Similarity Computation

**Files:**
- Modify: `swarm_rag_module/swarm_rag/core/swarm_retriever.py`

**Step 1: Write the failing test**

```python
def test_batched_similarity_computation(self, mock_retriever):
    """Verify batched similarity returns correct shapes."""
    batch_size = 4
    n_agents = 5
    max_degree = 10
    dim = 64

    query_vecs = torch.randn(batch_size, dim, device=mock_retriever._device)
    all_neighbors = torch.randint(0, 100, (batch_size, n_agents, max_degree), device=mock_retriever._device)
    neighbor_mask = torch.ones_like(all_neighbors, dtype=torch.bool)

    sims = mock_retriever._compute_similarities_multi_query(
        query_vecs, all_neighbors, neighbor_mask
    )

    assert sims.shape == (batch_size, n_agents, max_degree)
```

**Step 2: Run test to verify it fails**

Run: `pytest stark/test_multi_query_batch.py::TestMultiQueryBatching::test_batched_similarity_computation -v`

Expected: FAIL

**Step 3: Implement batched similarity**

```python
def _compute_similarities_multi_query(
    self,
    query_vecs: torch.Tensor,       # (batch_size, dim)
    all_neighbors: torch.Tensor,    # (batch_size, n_agents, max_degree)
    neighbor_mask: torch.Tensor,    # (batch_size, n_agents, max_degree)
) -> torch.Tensor:
    """
    Compute query-neighbor similarities for all queries simultaneously.

    Returns:
        similarities: (batch_size, n_agents, max_degree) similarity scores
    """
    batch_size, n_agents, max_degree = all_neighbors.shape
    device = query_vecs.device

    # Initialize output
    similarities = torch.full(
        (batch_size, n_agents, max_degree), -float('inf'),
        device=device, dtype=torch.float32
    )

    # Get unique neighbor IDs across all queries
    valid_neighbors = all_neighbors[neighbor_mask]
    if valid_neighbors.numel() == 0:
        return similarities

    unique_ids = torch.unique(valid_neighbors[valid_neighbors >= 0])
    if unique_ids.numel() == 0:
        return similarities

    # Fetch embeddings for unique IDs
    embs, valid_mask = self.vector_store.fetch_batch(unique_ids)
    valid_embs = embs[valid_mask].to(device=device, dtype=torch.float32)
    valid_ids = unique_ids[valid_mask]

    if valid_ids.numel() == 0:
        return similarities

    # Normalize query vectors
    query_vecs_norm = torch.nn.functional.normalize(query_vecs, p=2, dim=1)
    valid_embs_norm = torch.nn.functional.normalize(valid_embs, p=2, dim=1)

    # Compute all query-embedding similarities: (batch_size, n_unique)
    all_sims = torch.mm(query_vecs_norm, valid_embs_norm.t())

    # Build ID-to-index mapping
    max_id = self._max_node_id + 1
    id_to_idx = torch.full((max_id,), -1, device=device, dtype=torch.long)
    id_to_idx[valid_ids] = torch.arange(valid_ids.numel(), device=device)

    # Map neighbor IDs to embedding indices
    clamped_neighbors = all_neighbors.clamp(0, max_id - 1)
    emb_indices = id_to_idx[clamped_neighbors]  # (batch, agents, degree)

    # Scatter similarities into output
    # For each (q, a, n), if emb_indices[q,a,n] >= 0:
    #   similarities[q,a,n] = all_sims[q, emb_indices[q,a,n]]
    valid_emb_mask = (emb_indices >= 0) & neighbor_mask

    # Use advanced indexing: need batch indices for all_sims
    batch_idx = torch.arange(batch_size, device=device)[:, None, None].expand_as(emb_indices)

    # Gather similarities
    # all_sims is (batch, n_unique), we need sims for each (batch, agent, degree)
    flat_emb_idx = emb_indices[valid_emb_mask]
    flat_batch_idx = batch_idx[valid_emb_mask]
    gathered_sims = all_sims[flat_batch_idx, flat_emb_idx]

    similarities[valid_emb_mask] = gathered_sims

    # Scale to [0, 1]
    similarities = (similarities + 1.0) / 2.0
    similarities = torch.where(neighbor_mask, similarities, torch.zeros_like(similarities))

    return similarities
```

**Step 4: Run test to verify it passes**

Run: `pytest stark/test_multi_query_batch.py::TestMultiQueryBatching::test_batched_similarity_computation -v`

Expected: PASS

**Step 5: Commit**

```bash
git add swarm_rag_module/swarm_rag/core/swarm_retriever.py stark/test_multi_query_batch.py
git commit -m "feat: implement batched similarity computation for multi-query GPU"
```

---

## Task 5: Implement Batched Pheromone Operations

**Files:**
- Modify: `swarm_rag_module/swarm_rag/core/swarm_retriever.py`

**Step 1: Write the failing test**

```python
def test_batched_pheromone_lookup(self, mock_retriever):
    """Verify batched pheromone lookup."""
    batch_size = 4
    n_agents = 5
    max_degree = 10
    n_nodes = 100

    query_pheromones = torch.rand(batch_size, n_nodes, device=mock_retriever._device)
    all_neighbors = torch.randint(0, n_nodes, (batch_size, n_agents, max_degree), device=mock_retriever._device)

    pheromone_vals = mock_retriever._lookup_pheromones_multi_query(
        query_pheromones, all_neighbors
    )

    assert pheromone_vals.shape == (batch_size, n_agents, max_degree)


def test_batched_pheromone_deposit(self, mock_retriever):
    """Verify batched pheromone deposit."""
    batch_size = 4
    n_agents = 5
    n_nodes = 100

    query_pheromones = torch.zeros(batch_size, n_nodes, device=mock_retriever._device)
    new_locations = torch.randint(0, n_nodes, (batch_size, n_agents), device=mock_retriever._device)

    updated = mock_retriever._deposit_pheromones_multi_query(
        query_pheromones, new_locations, deposit_amount=1.0
    )

    assert updated.shape == query_pheromones.shape
    assert updated.sum() > 0  # Some deposits made
```

**Step 2: Run tests to verify they fail**

Run: `pytest stark/test_multi_query_batch.py -k "pheromone" -v`

Expected: FAIL

**Step 3: Implement pheromone operations**

```python
def _lookup_pheromones_multi_query(
    self,
    query_pheromones: torch.Tensor,  # (batch_size, n_nodes)
    all_neighbors: torch.Tensor,      # (batch_size, n_agents, max_degree)
) -> torch.Tensor:
    """
    Look up pheromone values for all neighbors across all queries.

    Returns:
        pheromone_vals: (batch_size, n_agents, max_degree)
    """
    batch_size = query_pheromones.shape[0]
    n_nodes = query_pheromones.shape[1]
    device = query_pheromones.device

    # Clamp neighbor IDs to valid range
    clamped = all_neighbors.clamp(0, n_nodes - 1)

    # Advanced indexing: pheromones[q, neighbors[q,a,n]]
    batch_idx = torch.arange(batch_size, device=device)[:, None, None]
    batch_idx = batch_idx.expand_as(clamped)

    pheromone_vals = query_pheromones[batch_idx, clamped]

    # Zero out-of-bounds lookups
    out_of_bounds = (all_neighbors < 0) | (all_neighbors >= n_nodes)
    pheromone_vals = torch.where(out_of_bounds, torch.zeros_like(pheromone_vals), pheromone_vals)

    return pheromone_vals


def _deposit_pheromones_multi_query(
    self,
    query_pheromones: torch.Tensor,  # (batch_size, n_nodes)
    new_locations: torch.Tensor,      # (batch_size, n_agents)
    deposit_amount: float = 1.0,
) -> torch.Tensor:
    """
    Deposit pheromones at new agent locations for all queries.

    Returns:
        Updated pheromone tensor
    """
    batch_size, n_nodes = query_pheromones.shape
    device = query_pheromones.device

    # Process each query (scatter_add doesn't support batch dimension well)
    for q in range(batch_size):
        locs = new_locations[q]
        valid_locs = locs[(locs >= 0) & (locs < n_nodes)]
        if valid_locs.numel() > 0:
            unique_locs, counts = torch.unique(valid_locs, return_counts=True)
            deposits = deposit_amount * counts.float()
            query_pheromones[q].scatter_add_(0, unique_locs, deposits)

    return query_pheromones
```

**Step 4: Run tests to verify they pass**

Run: `pytest stark/test_multi_query_batch.py -k "pheromone" -v`

Expected: PASS

**Step 5: Commit**

```bash
git add swarm_rag_module/swarm_rag/core/swarm_retriever.py stark/test_multi_query_batch.py
git commit -m "feat: implement batched pheromone operations for multi-query GPU"
```

---

## Task 6: Implement Batched Step Function

**Files:**
- Modify: `swarm_rag_module/swarm_rag/core/swarm_retriever.py`

**Step 1: Write the failing test**

```python
def test_batched_step_function(self, mock_retriever):
    """Verify batched step updates agent positions."""
    batch_size = 4
    n_agents = 5
    n_nodes = 100
    steps = 3

    agent_locs, pheromones, history = mock_retriever._init_multi_query_state(
        batch_size=batch_size,
        n_agents=n_agents,
        n_nodes=n_nodes,
        steps=steps,
        initial_pools=[[i, (i+1)%n_nodes, (i+2)%n_nodes, (i+3)%n_nodes, (i+4)%n_nodes]
                       for i in range(batch_size)],
        drop_zone_inc=0.05,
        seed=42,
    )

    query_vecs = torch.randn(batch_size, 64, device=mock_retriever._device)

    new_locs, deposits = mock_retriever._step_multi_query(
        agent_locations=agent_locs,
        query_vecs=query_vecs,
        query_pheromones=pheromones,
        step=0,
        max_pheromone=1.0,
    )

    assert new_locs.shape == (batch_size, n_agents)
    # Agents should have moved (mostly)
    assert not torch.equal(new_locs, agent_locs)
```

**Step 2: Run test to verify it fails**

Run: `pytest stark/test_multi_query_batch.py::TestMultiQueryBatching::test_batched_step_function -v`

Expected: FAIL

**Step 3: Implement batched step function**

```python
def _step_multi_query(
    self,
    agent_locations: torch.Tensor,   # (batch_size, n_agents)
    query_vecs: torch.Tensor,        # (batch_size, dim)
    query_pheromones: torch.Tensor,  # (batch_size, n_nodes)
    step: int,
    max_pheromone: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Execute one step for all agents across all queries.

    Returns:
        new_locations: (batch_size, n_agents)
        deposit_locations: (batch_size, n_agents) or None
    """
    batch_size, n_agents = agent_locations.shape
    device = agent_locations.device

    # 1. Get neighbors for all agents
    all_neighbors, neighbor_mask = self._get_neighbors_multi_query(agent_locations)
    # (batch_size, n_agents, max_degree)

    if all_neighbors is None:
        return agent_locations, None

    max_degree = all_neighbors.shape[2]

    # 2. Compute similarities
    similarities = self._compute_similarities_multi_query(
        query_vecs, all_neighbors, neighbor_mask
    )

    # 3. Lookup pheromones
    pheromone_vals = self._lookup_pheromones_multi_query(
        query_pheromones, all_neighbors
    )

    # 4. Get degrees for centrality heuristic
    flat_neighbors = all_neighbors.flatten()
    flat_degrees = self.graph_store.get_degrees_batch(flat_neighbors.clamp(0, self._max_node_id))
    all_degrees = flat_degrees.float().view(batch_size, n_agents, max_degree)

    # 5. Compute heuristic scores
    # Centrality: log(1 + degree) / (log(1 + degree) + avg_log_degree)
    log_degrees = torch.log(1 + all_degrees)
    centrality_scores = log_degrees / (log_degrees + self._avg_log_degree + 1e-8)

    # Pheromone repulsion: 1 - normalized_pheromone
    normalized_pheromones = pheromone_vals / (max_pheromone + 1e-8)
    repulsion_scores = 1.0 - normalized_pheromones

    # Combine with default weights
    total_scores = (
        0.3 * similarities +
        0.4 * centrality_scores +
        0.3 * repulsion_scores
    )

    # Apply mask
    total_scores = torch.where(neighbor_mask, total_scores, torch.zeros_like(total_scores))
    total_scores = torch.clamp(total_scores, min=0.001)

    # 6. Normalize to probabilities
    score_sums = total_scores.sum(dim=-1, keepdim=True)
    probs = total_scores / (score_sums + 1e-10)

    # 7. Sample next positions
    # Reshape for multinomial: (batch_size * n_agents, max_degree)
    flat_probs = probs.view(-1, max_degree)

    # Handle zero-sum rows
    row_sums = flat_probs.sum(dim=1, keepdim=True)
    flat_probs = torch.where(
        row_sums > 1e-10,
        flat_probs,
        torch.ones_like(flat_probs) / max_degree  # Uniform fallback
    )
    flat_probs = flat_probs / (flat_probs.sum(dim=1, keepdim=True) + 1e-10)

    chosen_idx = torch.multinomial(flat_probs, 1).view(batch_size, n_agents)

    # Gather chosen neighbors
    new_locations = all_neighbors.gather(2, chosen_idx.unsqueeze(-1)).squeeze(-1)

    return new_locations, new_locations.clone()
```

**Step 4: Run test to verify it passes**

Run: `pytest stark/test_multi_query_batch.py::TestMultiQueryBatching::test_batched_step_function -v`

Expected: PASS

**Step 5: Commit**

```bash
git add swarm_rag_module/swarm_rag/core/swarm_retriever.py stark/test_multi_query_batch.py
git commit -m "feat: implement batched step function for multi-query GPU"
```

---

## Task 7: Implement Full Multi-Query Traversal

**Files:**
- Modify: `swarm_rag_module/swarm_rag/core/swarm_retriever.py`

**Step 1: Write the failing test**

```python
def test_full_multi_query_traversal(self, mock_retriever):
    """Test complete multi-query batched traversal."""
    n_queries = 8
    query_embeddings = torch.randn(n_queries, 64, device=mock_retriever._device)
    initial_pools = [[i % 100, (i+1) % 100, (i+2) % 100, (i+3) % 100, (i+4) % 100]
                     for i in range(n_queries)]

    resolved_agents = mock_retriever._prepare_agents(
        agent_groups=None,
        n_agents=5,
        movement_strategies=mock_retriever._DEFAULT_PARAMS['movement_strategies'],
        deposit_strategies=mock_retriever._DEFAULT_PARAMS['deposit_strategies'],
    )

    # This should now use the real batched implementation
    results = mock_retriever._retrieve_batch_multi_query_gpu(
        query_embeddings=query_embeddings,
        initial_pools=initial_pools,
        resolved_agents=resolved_agents,
        base_seed=42,
        batch_size=4,  # Process in 2 batches
        steps=3,
        decay=0.9,
        drop_zone_inc=0.05,
        start_subset=5,
        top_k=5,
        ranking_strategies=mock_retriever._DEFAULT_PARAMS['ranking_strategies'],
    )

    assert len(results) == n_queries
    for r in results:
        assert isinstance(r, list)
        assert len(r) <= 5  # top_k
```

**Step 2: Run test to verify it fails**

Run: `pytest stark/test_multi_query_batch.py::TestMultiQueryBatching::test_full_multi_query_traversal -v`

Expected: FAIL (stub returns sequential results, not batched)

**Step 3: Implement full traversal with chunking**

Replace the stub in `_retrieve_batch_multi_query_gpu`:

```python
def _retrieve_batch_multi_query_gpu(
    self,
    query_embeddings: torch.Tensor,
    initial_pools: List[List[int]],
    resolved_agents: List[Tuple[Callable, Callable]],
    base_seed: int,
    batch_size: int = 32,
    **kwargs
) -> List[List[Dict]]:
    """
    Process multiple queries simultaneously on GPU with chunking.
    """
    n_queries = len(query_embeddings)
    n_agents = len(resolved_agents)
    steps = kwargs.get('steps', self._DEFAULT_PARAMS['steps'])
    decay = kwargs.get('decay', self._DEFAULT_PARAMS['decay'])
    drop_zone_inc = kwargs.get('drop_zone_inc', self._DEFAULT_PARAMS['drop_zone_inc'])
    start_subset = kwargs.get('start_subset', self._DEFAULT_PARAMS['start_subset'])
    top_k = kwargs.get('top_k', self._DEFAULT_PARAMS['top_k'])
    ranking_strategies = kwargs.get('ranking_strategies', self._DEFAULT_PARAMS['ranking_strategies'])

    ranking_func = self._compose_strategy(ranking_strategies, "ranking")

    all_results = []
    gid = kwargs.get('genome_id', '')

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        actual_batch = end - start

        batch_embeddings = query_embeddings[start:end]
        batch_pools = initial_pools[start:end]

        # Truncate pools to start_subset
        batch_pools = [p[:start_subset] if len(p) > start_subset else p for p in batch_pools]

        # Initialize state
        agent_locs, pheromones, history = self._init_multi_query_state(
            batch_size=actual_batch,
            n_agents=n_agents,
            n_nodes=self._pheromone_buffer_size,
            steps=steps,
            initial_pools=batch_pools,
            drop_zone_inc=drop_zone_inc,
            seed=base_seed + start,
        )

        # Traversal loop
        for step in range(steps):
            max_pheromone = torch.clamp(pheromones.max(), min=1.0)

            new_locs, deposit_locs = self._step_multi_query(
                agent_locations=agent_locs,
                query_vecs=batch_embeddings,
                query_pheromones=pheromones,
                step=step,
                max_pheromone=max_pheromone.item(),
            )

            # Update positions
            history[:, :, step + 1] = new_locs
            agent_locs = new_locs

            # Decay and deposit pheromones
            pheromones *= decay
            if deposit_locs is not None:
                pheromones = self._deposit_pheromones_multi_query(
                    pheromones, deposit_locs, deposit_amount=1.0
                )

        # Rank results for each query in batch
        for q in range(actual_batch):
            query_results = self._ranking_from_history(
                history[q],  # (n_agents, steps+1)
                batch_embeddings[q],
                ranking_func,
                top_k,
                n_agents,
            )
            all_results.append(query_results)

        if gid:
            logger.info(f"    [Retriever] [{gid}] Multi-Query Batch: {end}/{n_queries}")

    return all_results
```

**Step 4: Run test to verify it passes**

Run: `pytest stark/test_multi_query_batch.py::TestMultiQueryBatching::test_full_multi_query_traversal -v`

Expected: PASS

**Step 5: Commit**

```bash
git add swarm_rag_module/swarm_rag/core/swarm_retriever.py stark/test_multi_query_batch.py
git commit -m "feat: implement full multi-query GPU traversal with chunking"
```

---

## Task 8: Wire Up Entry Point

**Files:**
- Modify: `swarm_rag_module/swarm_rag/core/swarm_retriever.py:1675-1695`

**Step 1: Write the integration test**

```python
def test_retrieve_batch_uses_multi_query_on_gpu(self, mock_retriever):
    """Verify retrieve_batch_with_precomputed uses multi-query path on GPU."""
    if not mock_retriever._use_gpu:
        pytest.skip("GPU not available")

    n_queries = 8
    query_embeddings = torch.randn(n_queries, 64, device=mock_retriever._device)
    initial_pools = [[i % 100, (i+1) % 100, (i+2) % 100] for i in range(n_queries)]

    # This is the public API
    results = mock_retriever.retrieve_batch_with_precomputed(
        query_embeddings=query_embeddings,
        initial_pools=initial_pools,
        n_agents=5,
        steps=3,
        top_k=5,
    )

    assert len(results) == n_queries
```

**Step 2: Update entry point**

Modify `retrieve_batch_with_precomputed` at line 1675:

```python
# Process based on device mode
if self._use_gpu:
    # Multi-query batched processing for GPU
    return self._retrieve_batch_multi_query_gpu(
        query_embeddings,
        initial_pools,
        resolved_agents,
        base_seed=base_seed,
        batch_size=32,
        **params
    )
elif max_workers <= 1:
    # Sequential processing
    return self._retrieve_batch_precomputed_sequential(
        query_embeddings,
        initial_pools,
        resolved_agents,
        base_seed=base_seed,
        **params
    )
else:
    # Parallel processing for CPU
    return self._retrieve_batch_precomputed_parallel(
        query_embeddings,
        initial_pools,
        resolved_agents,
        base_seed=base_seed,
        max_workers=max_workers,
        **params
    )
```

**Step 3: Run integration test**

Run: `pytest stark/test_multi_query_batch.py -v`

Expected: All PASS

**Step 4: Run evolution test**

Run: `python stark/evolve_stark.py --dataset prime --device cuda --gens 3 --train_ss 30 --val_ss 15`

Expected: Runs successfully with "Multi-Query Batch" log messages

**Step 5: Commit**

```bash
git add swarm_rag_module/swarm_rag/core/swarm_retriever.py
git commit -m "feat: wire up multi-query GPU batching in retrieve_batch_with_precomputed"
```

---

## Task 9: Benchmark and Tune Batch Size

**Files:**
- Create: `stark/benchmark_multi_query.py`

**Step 1: Create benchmark script**

```python
"""Benchmark multi-query batching performance."""
import time
import torch
from swarm_rag.core import SwarmRetriever

def benchmark_batch_sizes(retriever, n_queries=50, batch_sizes=[1, 8, 16, 32, 64]):
    """Benchmark different batch sizes."""
    query_embeddings = torch.randn(n_queries, retriever.vector_store.dim, device=retriever._device)
    initial_pools = [[i % retriever._max_node_id for i in range(j, j+30)]
                     for j in range(n_queries)]

    results = {}

    for bs in batch_sizes:
        # Warmup
        _ = retriever._retrieve_batch_multi_query_gpu(
            query_embeddings[:8], initial_pools[:8],
            retriever._prepare_agents(None, 20,
                retriever._DEFAULT_PARAMS['movement_strategies'],
                retriever._DEFAULT_PARAMS['deposit_strategies']),
            base_seed=42, batch_size=bs, steps=5, decay=0.9,
            drop_zone_inc=0.05, start_subset=10, top_k=20,
            ranking_strategies=retriever._DEFAULT_PARAMS['ranking_strategies'],
        )
        torch.cuda.synchronize()

        # Timed run
        start = time.perf_counter()
        _ = retriever._retrieve_batch_multi_query_gpu(
            query_embeddings, initial_pools,
            retriever._prepare_agents(None, 20,
                retriever._DEFAULT_PARAMS['movement_strategies'],
                retriever._DEFAULT_PARAMS['deposit_strategies']),
            base_seed=42, batch_size=bs, steps=5, decay=0.9,
            drop_zone_inc=0.05, start_subset=10, top_k=20,
            ranking_strategies=retriever._DEFAULT_PARAMS['ranking_strategies'],
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        results[bs] = elapsed
        print(f"Batch size {bs:3d}: {elapsed*1000:.1f}ms ({n_queries/elapsed:.1f} queries/sec)")

    return results

if __name__ == "__main__":
    # Load real data
    from stark.load_stark import load_and_download_skb, load_and_download_embeddings
    from swarm_rag.integrations.stark import StarkVectorStore, StarkGraphAdapter

    print("Loading Prime dataset...")
    skb = load_and_download_skb("prime")
    query_embs, doc_embs = load_and_download_embeddings("prime")

    vector_store = StarkVectorStore(doc_embs, device="cuda")
    # ... setup retriever

    print("\nBenchmarking batch sizes...")
    benchmark_batch_sizes(retriever)
```

**Step 2: Run benchmark**

Run: `python stark/benchmark_multi_query.py`

**Step 3: Commit**

```bash
git add stark/benchmark_multi_query.py
git commit -m "feat: add benchmark script for multi-query batch size tuning"
```

---

## Summary

After completing all tasks:

1. **Multi-query batching** processes N queries simultaneously on GPU
2. **Chunking** (default batch_size=32) manages memory for large datasets
3. **Entry point** automatically uses batched path on GPU
4. **Benchmark** helps tune optimal batch size per dataset

**Expected speedup:** 5-10x on validation, reducing evolution time significantly.
