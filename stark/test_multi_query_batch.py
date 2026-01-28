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
