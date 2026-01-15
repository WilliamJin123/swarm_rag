"""
Tests for GPU utilities: benchmarking, memory profiling, CuPy integration, and batch optimization.
"""

import pytest
import numpy as np
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

    def test_get_array_module_cpu(self):
        """Test get_array_module returns numpy when on CPU."""
        from swarm_rag.utils.device import get_array_module, clear_device_cache

        clear_device_cache()
        with patch.dict('os.environ', {'SWARM_RAG_DEVICE': 'cpu'}):
            clear_device_cache()
            xp = get_array_module()
            assert xp.__name__ == 'numpy'

    def test_ensure_tensor_numpy_input(self):
        """Test ensure_tensor with numpy array input."""
        from swarm_rag.utils.device import ensure_tensor, clear_device_cache

        clear_device_cache()
        arr = np.array([1.0, 2.0, 3.0])

        try:
            import torch
            tensor = ensure_tensor(arr, device="cpu")
            assert isinstance(tensor, torch.Tensor)
            assert tensor.device.type == "cpu"
            np.testing.assert_array_almost_equal(tensor.numpy(), arr)
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_to_numpy_tensor(self):
        """Test to_numpy with torch tensor."""
        from swarm_rag.utils.device import to_numpy

        try:
            import torch
            tensor = torch.tensor([1.0, 2.0, 3.0])
            arr = to_numpy(tensor)
            assert isinstance(arr, np.ndarray)
            np.testing.assert_array_almost_equal(arr, [1.0, 2.0, 3.0])
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_to_numpy_array(self):
        """Test to_numpy with numpy array (passthrough)."""
        from swarm_rag.utils.device import to_numpy

        arr = np.array([1.0, 2.0, 3.0])
        result = to_numpy(arr)
        assert result is arr


class TestCuPyIntegration:
    """Tests for CuPy integration utilities."""

    def test_is_cupy_available(self):
        """Test CuPy availability check."""
        from swarm_rag.utils.device import is_cupy_available

        # Should return boolean without error
        result = is_cupy_available()
        assert isinstance(result, bool)

    def test_cupy_to_numpy_fallback(self):
        """Test cupy_to_numpy falls back gracefully."""
        from swarm_rag.utils.device import cupy_to_numpy

        arr = np.array([1.0, 2.0, 3.0])
        result = cupy_to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_cupy_matmul_cpu(self):
        """Test cupy_matmul on CPU."""
        from swarm_rag.utils.device import cupy_matmul, clear_device_cache

        with patch.dict('os.environ', {'SWARM_RAG_DEVICE': 'cpu'}):
            clear_device_cache()
            a = np.array([[1, 2], [3, 4]])
            b = np.array([[5, 6], [7, 8]])
            result = cupy_matmul(a, b)
            expected = np.matmul(a, b)
            np.testing.assert_array_equal(result, expected)

    def test_cupy_dot_cpu(self):
        """Test cupy_dot on CPU."""
        from swarm_rag.utils.device import cupy_dot, clear_device_cache

        with patch.dict('os.environ', {'SWARM_RAG_DEVICE': 'cpu'}):
            clear_device_cache()
            a = np.array([1, 2, 3])
            b = np.array([4, 5, 6])
            result = cupy_dot(a, b)
            expected = np.dot(a, b)
            assert result == expected

    def test_cupy_norm_cpu(self):
        """Test cupy_norm on CPU."""
        from swarm_rag.utils.device import cupy_norm, clear_device_cache

        with patch.dict('os.environ', {'SWARM_RAG_DEVICE': 'cpu'}):
            clear_device_cache()
            arr = np.array([3, 4])
            result = cupy_norm(arr)
            assert abs(result - 5.0) < 1e-6

    def test_cupy_normalize_cpu(self):
        """Test cupy_normalize on CPU."""
        from swarm_rag.utils.device import cupy_normalize, clear_device_cache

        with patch.dict('os.environ', {'SWARM_RAG_DEVICE': 'cpu'}):
            clear_device_cache()
            arr = np.array([[3, 4], [6, 8]])
            result = cupy_normalize(arr, axis=1)

            # Check each row is unit length
            norms = np.linalg.norm(result, axis=1)
            np.testing.assert_array_almost_equal(norms, [1.0, 1.0])

    def test_cupy_cosine_similarity_cpu(self):
        """Test cupy_cosine_similarity on CPU."""
        from swarm_rag.utils.device import cupy_cosine_similarity, clear_device_cache

        with patch.dict('os.environ', {'SWARM_RAG_DEVICE': 'cpu'}):
            clear_device_cache()
            query = np.array([1.0, 0.0])
            candidates = np.array([
                [1.0, 0.0],  # Same direction
                [0.0, 1.0],  # Orthogonal
                [-1.0, 0.0]  # Opposite
            ])

            scores = cupy_cosine_similarity(query, candidates)

            assert abs(scores[0] - 1.0) < 1e-6  # Same direction
            assert abs(scores[1]) < 1e-6  # Orthogonal
            assert abs(scores[2] + 1.0) < 1e-6  # Opposite

    def test_cupy_topk_cpu(self):
        """Test cupy_topk on CPU."""
        from swarm_rag.utils.device import cupy_topk, clear_device_cache

        with patch.dict('os.environ', {'SWARM_RAG_DEVICE': 'cpu'}):
            clear_device_cache()
            scores = np.array([0.1, 0.5, 0.3, 0.9, 0.2])

            top_scores, top_indices = cupy_topk(scores, k=3)

            # Check top values
            assert len(top_scores) == 3
            assert len(top_indices) == 3
            assert top_scores[0] == 0.9  # Highest
            assert 3 in top_indices  # Index of 0.9 (scores[3] = 0.9)


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
            _ = np.zeros((1000, 1000))

        assert len(profiler.deltas) == 1
        assert profiler.deltas[0].label == "test_operation"
        assert profiler.deltas[0].duration_ms > 0

    def test_memory_profile_decorator(self):
        """Test memory profiling decorator."""
        from swarm_rag.utils.memory import MemoryProfiler

        profiler = MemoryProfiler()

        @profiler.profile
        def my_func():
            return np.zeros((100, 100))

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
        mem_mb = estimate_tensor_memory((1000, 768), dtype=np.float32)
        expected_mb = (1000 * 768 * 4) / (1024 * 1024)
        assert abs(mem_mb - expected_mb) < 0.01

    def test_memory_guard_context_manager(self):
        """Test memory_guard context manager."""
        from swarm_rag.utils.memory import memory_guard

        with memory_guard() as profiler:
            _ = np.zeros((100, 100))

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
        query_vecs = np.random.randn(3, 768).astype(np.float32)
        results = retriever._batch_initial_search(query_vecs, pool_size=10)

        assert len(results) == 3
        assert mock_vector_store.search.call_count == 3
        # Each result should have 2 IDs (from our mock)
        for r in results:
            assert len(r) == 2

    def test_compute_batch_similarities_cpu(self):
        """Test _compute_batch_similarities_gpu on CPU."""
        # Create properly shaped mock data
        mock_embeddings = np.random.randn(10, 768).astype(np.float32)

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
        query_vecs = np.random.randn(2, 768).astype(np.float32)
        candidate_ids = [[1, 2, 3], [4, 5]]

        results = retriever._compute_batch_similarities_gpu(query_vecs, candidate_ids)

        assert len(results) == 2
        # Each result is a tuple of (scores, valid_ids)
        for scores, valid_ids in results:
            assert isinstance(scores, np.ndarray) or len(scores) == 0


class TestHeuristicsGPU:
    """Tests for GPU-aware heuristics."""

    def test_semantic_similarity_numpy(self):
        """Test semantic_similarity with numpy arrays."""
        from swarm_rag.core.heuristics import Heuristics, HeuristicContext

        query = np.array([1.0, 0.0, 0.0])
        targets = np.array([
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
        assert abs(scores[0] - 1.0) < 0.01
        assert abs(scores[1] - 0.5) < 0.01

    def test_semantic_similarity_torch(self):
        """Test semantic_similarity with torch tensors."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        from swarm_rag.core.heuristics import Heuristics, HeuristicContext

        query = torch.tensor([1.0, 0.0, 0.0])
        targets = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        ctx = HeuristicContext(
            query_vec=query,
            target_vecs=targets,
            target_ids=[0, 1]
        )

        scores = Heuristics.semantic_similarity(ctx)

        # Should work with torch tensors
        if isinstance(scores, torch.Tensor):
            scores = scores.numpy()

        assert abs(scores[0] - 1.0) < 0.01
        assert abs(scores[1] - 0.5) < 0.01

    def test_dot_product_mixed_types(self):
        """Test _dot_product with mixed numpy/torch inputs."""
        from swarm_rag.core.heuristics import _dot_product

        a_np = np.array([1.0, 2.0, 3.0])
        b_np = np.array([4.0, 5.0, 6.0])

        result = _dot_product(a_np, b_np)
        expected = np.dot(a_np, b_np)

        assert abs(float(result) - expected) < 1e-6

        try:
            import torch
            a_torch = torch.tensor([1.0, 2.0, 3.0])
            b_torch = torch.tensor([4.0, 5.0, 6.0])

            result = _dot_product(a_torch, b_torch)
            assert abs(float(result) - expected) < 1e-6
        except ImportError:
            pass


class TestGPUVectorStore:
    """Tests for GPUVectorStore."""

    def test_gpu_vector_store_creation(self):
        """Test GPUVectorStore creation."""
        try:
            import torch
            from swarm_rag.integrations.gpu_vector_store import GPUVectorStore
        except ImportError:
            pytest.skip("PyTorch not available")

        # Create test data
        embeddings = torch.randn(100, 64)
        ids = list(range(100))

        store = GPUVectorStore(embeddings, ids, device="cpu")

        assert store.n_docs == 100
        assert store.dim == 64

    def test_gpu_vector_store_search(self):
        """Test GPUVectorStore search."""
        try:
            import torch
            from swarm_rag.integrations.gpu_vector_store import GPUVectorStore
        except ImportError:
            pytest.skip("PyTorch not available")

        # Create test data with known relationships
        embeddings = torch.zeros(10, 64)
        embeddings[0] = torch.tensor([1.0] + [0.0] * 63)  # Will be most similar to query
        ids = list(range(10))

        store = GPUVectorStore(embeddings, ids, device="cpu")

        # Query similar to doc 0
        query = np.array([1.0] + [0.0] * 63, dtype=np.float32)
        results = store.search(query, limit=5)

        assert len(results) == 5
        assert results[0]['id'] == 0  # Doc 0 should be most similar

    def test_gpu_vector_store_from_dict(self):
        """Test GPUVectorStore.from_dict."""
        try:
            import torch
            from swarm_rag.integrations.gpu_vector_store import GPUVectorStore
        except ImportError:
            pytest.skip("PyTorch not available")

        doc_embs = {
            0: torch.randn(64),
            1: torch.randn(64),
            2: torch.randn(64),
        }

        store = GPUVectorStore.from_dict(doc_embs, device="cpu")

        assert store.n_docs == 3
        assert store.dim == 64

    def test_gpu_vector_store_fetch_batch(self):
        """Test GPUVectorStore.fetch_batch."""
        try:
            import torch
            from swarm_rag.integrations.gpu_vector_store import GPUVectorStore
        except ImportError:
            pytest.skip("PyTorch not available")

        embeddings = torch.randn(10, 64)
        ids = list(range(10))

        store = GPUVectorStore(embeddings, ids, device="cpu")

        result = store.fetch_batch([0, 1, 2])

        assert result.shape == (3, 64)
        # Check no NaN for valid IDs
        assert not np.isnan(result).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
