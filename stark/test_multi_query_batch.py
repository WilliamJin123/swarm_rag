"""Tests for multi-query GPU batching."""
import torch
import pytest

from swarm_rag.evolution.types.config import STARK_FEATURES


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

        # Create weight tensors for testing
        weight_tensors = mock_retriever._create_default_weight_tensors(device=mock_retriever._device)

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
            weight_tensors=weight_tensors,
        )

        assert len(results) == n_queries
        for r in results:
            assert isinstance(r, list)

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

        # Create weight tensors for testing
        weight_tensors = mock_retriever._create_default_weight_tensors(device=mock_retriever._device)
        feature_names = ["semantic_similarity_unnormalized", "node_centrality", "pheromone_repulsion"]

        new_locs, deposits = mock_retriever._step_multi_query(
            agent_locations=agent_locs,
            query_vecs=query_vecs,
            query_pheromones=pheromones,
            step=0,
            max_pheromone=1.0,
            weight_tensors=weight_tensors,
            feature_names=feature_names,
        )

        assert new_locs.shape == (batch_size, n_agents)
        # Agents should have moved (mostly)
        assert not torch.equal(new_locs, agent_locs)

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

        # Create weight tensors for testing
        weight_tensors = mock_retriever._create_default_weight_tensors(device=mock_retriever._device)

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
            weight_tensors=weight_tensors,
        )

        assert len(results) == n_queries
        for r in results:
            assert isinstance(r, list)
            assert len(r) <= 5  # top_k

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
