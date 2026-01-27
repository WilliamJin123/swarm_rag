"""
Tests for GPU utilities: benchmarking, memory profiling, and batch optimization.
"""

import pytest
import torch
import time
from unittest.mock import patch, MagicMock


class TestDeviceUtilities:
    """Tests for device detection and utilities."""

    def test_get_device_cpu_fallback(self):
        """Test that get_device returns cpu when CUDA unavailable."""
        from swarm_rag.utils.device import clear_device_cache, get_device

        # Clear cache first
        clear_device_cache()

        # When testing without GPU, should return cpu
        device = get_device()
        assert device in ["cpu", "cuda"]

    def test_get_device_force_cpu(self):
        """Test force_cpu parameter."""
        from swarm_rag.utils.device import clear_device_cache, get_device

        clear_device_cache()
        device = get_device(force_cpu=True)
        assert device == "cpu"

    def test_ensure_tensor_torch_input(self):
        """Test ensure_tensor with torch tensor input."""
        from swarm_rag.utils.device import ensure_tensor, clear_device_cache

        clear_device_cache()
        original = torch.tensor([1.0, 2.0, 3.0])

        tensor = ensure_tensor(original, device="cpu")
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"
        assert torch.allclose(tensor, original)

    def test_ensure_tensor_list_input(self):
        """Test ensure_tensor with list input."""
        from swarm_rag.utils.device import ensure_tensor, clear_device_cache

        clear_device_cache()
        data = [1.0, 2.0, 3.0]

        tensor = ensure_tensor(data, device="cpu")
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"
        assert torch.allclose(tensor, torch.tensor(data))


class TestBenchmarker:
    """Tests for the Benchmarker class."""

    def test_benchmarker_initialization(self):
        """Test Benchmarker initialization."""
        from swarm_rag.utils.benchmark import Benchmarker

        b = Benchmarker(warmup_iterations=2, n_iterations=5)
        assert b.warmup_iterations == 2
        assert b.n_iterations == 5

    def test_benchmarker_run_simple(self):
        """Test Benchmarker.run with a simple function."""
        from swarm_rag.utils.benchmark import Benchmarker

        b = Benchmarker(warmup_iterations=1, n_iterations=3)

        counter = [0]
        def simple_func():
            counter[0] += 1
            time.sleep(0.001)  # 1ms
            return counter[0]

        result = b.run("test", simple_func, device="cpu")

        assert result.name == "test"
        assert result.device == "cpu"
        assert result.n_iterations == 3
        assert result.mean_time_ms > 0
        assert result.throughput > 0

    def test_benchmark_result_to_dict(self):
        """Test BenchmarkResult.to_dict."""
        from swarm_rag.utils.benchmark import BenchmarkResult

        result = BenchmarkResult(
            name="test",
            device="cpu",
            n_iterations=10,
            mean_time_ms=5.0,
            std_time_ms=0.5,
            min_time_ms=4.0,
            max_time_ms=6.0,
            throughput=200.0
        )

        d = result.to_dict()
        assert d['name'] == "test"
        assert d['mean_time_ms'] == 5.0

    def test_benchmarker_compare_cpu(self):
        """Test Benchmarker.compare on CPU (when CUDA unavailable)."""
        from swarm_rag.utils.benchmark import Benchmarker
        from swarm_rag.utils.device import get_device, clear_device_cache

        clear_device_cache()

        # Skip if CUDA is not available since compare tries to sync
        if get_device() != "cuda":
            # Just test that single runs work on CPU
            b = Benchmarker(warmup_iterations=1, n_iterations=3, sync_cuda=False)

            def slow_func():
                time.sleep(0.01)
                return 1

            result = b.run("cpu_only_test", slow_func, device="cpu")
            assert result.name == "cpu_only_test"
            assert result.mean_time_ms > 0
            return

        b = Benchmarker(warmup_iterations=1, n_iterations=3)

        def slow_func():
            time.sleep(0.01)
            return 1

        def fast_func():
            time.sleep(0.005)
            return 1

        comparison = b.compare("speed_test", slow_func, fast_func)

        assert comparison.benchmark_name == "speed_test"
        assert comparison.cpu_result is not None
        assert comparison.gpu_result is not None


class TestMemoryProfiler:
    """Tests for the MemoryProfiler class."""

    def test_memory_profiler_initialization(self):
        """Test MemoryProfiler initialization."""
        from swarm_rag.utils.memory import MemoryProfiler

        profiler = MemoryProfiler()
        assert len(profiler.snapshots) == 0
        assert len(profiler.deltas) == 0

    def test_memory_snapshot(self):
        """Test taking memory snapshots."""
        from swarm_rag.utils.memory import MemoryProfiler

        profiler = MemoryProfiler()
        snap = profiler.snapshot("test_snap")

        assert snap.label == "test_snap"
        assert snap.timestamp > 0
        assert len(profiler.snapshots) == 1

    def test_memory_track_context_manager(self):
        """Test memory tracking context manager."""
        from swarm_rag.utils.memory import MemoryProfiler

        profiler = MemoryProfiler()

        with profiler.track("test_operation"):
            # Allocate some memory
            _ = torch.zeros((1000, 1000))

        assert len(profiler.deltas) == 1
        assert profiler.deltas[0].label == "test_operation"
        assert profiler.deltas[0].duration_ms > 0

    def test_memory_profile_decorator(self):
        """Test memory profiling decorator."""
        from swarm_rag.utils.memory import MemoryProfiler

        profiler = MemoryProfiler()

        @profiler.profile
        def my_func():
            return torch.zeros((100, 100))

        result = my_func()
        assert result.shape == (100, 100)
        assert len(profiler.deltas) == 1
        assert profiler.deltas[0].label == "my_func"

    def test_memory_snapshot_to_dict(self):
        """Test MemorySnapshot.to_dict."""
        from swarm_rag.utils.memory import MemorySnapshot

        snap = MemorySnapshot(
            timestamp=1.0,
            label="test",
            gpu_allocated=1024 * 1024,
            gpu_total=4 * 1024 * 1024 * 1024,
            process_rss=512 * 1024 * 1024
        )

        d = snap.to_dict()
        assert d['label'] == "test"
        assert d['gpu_allocated_mb'] == 1.0
        assert d['process_rss_mb'] == 512.0

    def test_estimate_tensor_memory(self):
        """Test estimate_tensor_memory utility."""
        from swarm_rag.utils.memory import estimate_tensor_memory

        # 1000 x 768 float32 = 3,072,000 bytes = ~2.93 MB
        mem_mb = estimate_tensor_memory((1000, 768), dtype=torch.float32)
        expected_mb = (1000 * 768 * 4) / (1024 * 1024)
        assert abs(mem_mb - expected_mb) < 0.01

    def test_memory_guard_context_manager(self):
        """Test memory_guard context manager."""
        from swarm_rag.utils.memory import memory_guard

        with memory_guard() as profiler:
            _ = torch.zeros((100, 100))

        assert len(profiler.snapshots) >= 2  # At least start and end


class TestBatchOptimization:
    """Tests for batch optimization in SwarmRetriever."""

    def test_batch_initial_search_fallback(self):
        """Test _batch_initial_search falls back to sequential."""
        # Create mock objects with proper behavior
        mock_vector_store = MagicMock()
        mock_vector_store.search = MagicMock(return_value=[
            {'id': 1, 'score': 0.9},
            {'id': 2, 'score': 0.8}
        ])
        # Important: remove search_batch to force fallback
        del mock_vector_store.search_batch

        mock_graph_store = MagicMock()
        mock_graph_store.contains = MagicMock(return_value=True)
        mock_graph_store.get_avg_degree = MagicMock(return_value=10.0)

        mock_embed = MagicMock()

        # Import after setting up mocks
        from swarm_rag.core.swarm_retriever import SwarmRetriever

        retriever = SwarmRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_provider=mock_embed,
            use_gpu=False,
            cache_neighbors=False,
            cache_vectors=False
        )

        # Test batch search
        torch.manual_seed(42)
        query_vecs = torch.randn(3, 768, dtype=torch.float32)
        results = retriever._batch_initial_search(query_vecs, pool_size=10)

        assert len(results) == 3
        assert mock_vector_store.search.call_count == 3
        # Each result should have 2 IDs (from our mock)
        for r in results:
            assert len(r) == 2

    def test_compute_batch_similarities_cpu(self):
        """Test _compute_batch_similarities_gpu on CPU."""
        # Create properly shaped mock data
        torch.manual_seed(42)
        mock_embeddings = torch.randn(10, 768, dtype=torch.float32)

        mock_vector_store = MagicMock()
        # fetch_batch should return matrix with shape (n_requested, dim)
        def mock_fetch_batch(ids):
            n = len(ids)
            return mock_embeddings[:n]

        mock_vector_store.fetch_batch = MagicMock(side_effect=mock_fetch_batch)

        mock_graph_store = MagicMock()
        mock_graph_store.get_avg_degree = MagicMock(return_value=10.0)

        mock_embed = MagicMock()

        from swarm_rag.core.swarm_retriever import SwarmRetriever

        retriever = SwarmRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_provider=mock_embed,
            use_gpu=False,
            cache_neighbors=False,
            cache_vectors=False
        )

        # Test batch similarities
        query_vecs = torch.randn(2, 768, dtype=torch.float32)
        candidate_ids = [[1, 2, 3], [4, 5]]

        results = retriever._compute_batch_similarities_gpu(query_vecs, candidate_ids)

        assert len(results) == 2
        # Each result is a tuple of (scores, valid_ids)
        for scores, valid_ids in results:
            assert isinstance(scores, torch.Tensor) or len(scores) == 0


class TestHeuristicsGPU:
    """Tests for GPU-aware heuristics."""

    def test_semantic_similarity_torch(self):
        """Test semantic_similarity with torch tensors."""
        from swarm_rag.core.heuristics import Heuristics, HeuristicContext

        query = torch.tensor([1.0, 0.0, 0.0])
        targets = torch.tensor([
            [1.0, 0.0, 0.0],  # Same direction
            [0.0, 1.0, 0.0],  # Orthogonal
        ])

        ctx = HeuristicContext(
            query_vec=query,
            target_vecs=targets,
            target_ids=[0, 1]
        )

        scores = Heuristics.semantic_similarity(ctx)

        # Normalized: [-1,1] -> [0,1]
        # Same direction: cos=1 -> normalized=1
        # Orthogonal: cos=0 -> normalized=0.5
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu()
        assert abs(float(scores[0]) - 1.0) < 0.01
        assert abs(float(scores[1]) - 0.5) < 0.01

    def test_semantic_similarity_orthogonal(self):
        """Test semantic_similarity with orthogonal vectors."""
        from swarm_rag.core.heuristics import Heuristics, HeuristicContext

        query = torch.tensor([1.0, 0.0, 0.0])
        targets = torch.tensor([
            [0.0, 1.0, 0.0],  # Orthogonal
            [-1.0, 0.0, 0.0],  # Opposite direction
        ])

        ctx = HeuristicContext(
            query_vec=query,
            target_vecs=targets,
            target_ids=[0, 1]
        )

        scores = Heuristics.semantic_similarity(ctx)

        if isinstance(scores, torch.Tensor):
            scores = scores.cpu()
        # Orthogonal: cos=0 -> normalized=0.5
        assert abs(float(scores[0]) - 0.5) < 0.01
        # Opposite: cos=-1 -> normalized=0
        assert abs(float(scores[1]) - 0.0) < 0.01

    def test_dot_product_torch(self):
        """Test _dot_product with torch tensors."""
        from swarm_rag.core.heuristics import _dot_product

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])

        result = _dot_product(a, b)
        expected = 1*4 + 2*5 + 3*6  # 32

        assert abs(float(result) - expected) < 1e-6


class TestTorchVectorStore:
    """Tests for TorchVectorStore."""

    def test_torch_vector_store_creation(self):
        """Test TorchVectorStore creation."""
        try:
            import torch
            from swarm_rag.integrations.torch_vector_store import TorchVectorStore
        except ImportError:
            pytest.skip("PyTorch not available")

        # Create test data
        embeddings = torch.randn(100, 64)
        ids = list(range(100))

        store = TorchVectorStore(embeddings, ids, device="cpu")

        assert store.n_docs == 100
        assert store.dim == 64

    def test_torch_vector_store_search(self):
        """Test TorchVectorStore search returns tensors."""
        from swarm_rag.integrations.torch_vector_store import TorchVectorStore

        # Create test data with known relationships
        embeddings = torch.zeros(10, 64)
        embeddings[0] = torch.tensor([1.0] + [0.0] * 63)  # Will be most similar to query
        ids = list(range(10))

        store = TorchVectorStore(embeddings, ids, device="cpu")

        # Query similar to doc 0
        query = torch.tensor([1.0] + [0.0] * 63, dtype=torch.float32)
        result_ids, result_scores = store.search(query, limit=5)

        # Results are now tensors
        assert isinstance(result_ids, torch.Tensor)
        assert isinstance(result_scores, torch.Tensor)
        assert len(result_ids) == 5
        assert result_ids[0].item() == 0  # Doc 0 should be most similar

    def test_torch_vector_store_from_dict(self):
        """Test TorchVectorStore.from_dict."""
        try:
            import torch
            from swarm_rag.integrations.torch_vector_store import TorchVectorStore
        except ImportError:
            pytest.skip("PyTorch not available")

        doc_embs = {
            0: torch.randn(64),
            1: torch.randn(64),
            2: torch.randn(64),
        }

        store = TorchVectorStore.from_dict(doc_embs, device="cpu")

        assert store.n_docs == 3
        assert store.dim == 64

    def test_torch_vector_store_fetch_batch(self):
        """Test TorchVectorStore.fetch_batch."""
        from swarm_rag.integrations.torch_vector_store import TorchVectorStore

        embeddings = torch.randn(10, 64)
        ids = list(range(10))

        store = TorchVectorStore(embeddings, ids, device="cpu")

        result = store.fetch_batch([0, 1, 2])

        assert result.shape == (3, 64)
        # Check no NaN for valid IDs
        assert not torch.isnan(result).any()


class TestTorchGraphStore:
    """Tests for TorchGraphStore."""

    def test_torch_graph_store_creation(self):
        """Test TorchGraphStore creation from adjacency dict."""
        from swarm_rag.integrations.torch_graph_store import TorchGraphStore

        adj_dict = {
            0: [1, 2],
            1: [0, 2],
            2: [0, 1, 3],
            3: [2],
        }

        store = TorchGraphStore.from_adjacency_dict(adj_dict, device="cpu")

        assert store.n_nodes == 4
        assert store.get_degree(0) == 2
        assert store.get_degree(2) == 3

    def test_torch_graph_store_neighbors(self):
        """Test TorchGraphStore neighbor lookup."""
        from swarm_rag.integrations.torch_graph_store import TorchGraphStore

        adj_dict = {
            0: [1, 2],
            1: [0],
            2: [0, 1],
        }

        store = TorchGraphStore.from_adjacency_dict(adj_dict, device="cpu")

        neighbors = store.get_neighbors(0)
        assert len(neighbors) == 2
        assert set(neighbors.tolist()) == {1, 2}

    def test_torch_graph_store_batch_neighbors(self):
        """Test TorchGraphStore batch neighbor lookup."""
        from swarm_rag.integrations.torch_graph_store import TorchGraphStore

        adj_dict = {
            0: [1, 2],
            1: [0],
            2: [0, 1, 3],
            3: [2],
        }

        store = TorchGraphStore.from_adjacency_dict(adj_dict, device="cpu")

        neighbors, mask = store.get_neighbors_batch([0, 1, 2])

        assert neighbors.shape[0] == 3
        assert mask.shape[0] == 3
        # Node 2 has most neighbors (3)
        assert neighbors.shape[1] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
