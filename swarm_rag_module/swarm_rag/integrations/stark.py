import math
import os
from typing import Dict, List, Optional, Union
import numpy as np
from numpy.typing import NDArray
from multiprocessing import shared_memory
import scipy.sparse as sp
import atexit

from ..interfaces.abstract_classes import VectorStore, GraphStore, EmbeddingProvider, Matrix
from ..interfaces.enums import HeuristicKey
from ..utils import fail_on_missing_imports, LRUCache, get_device
from ..core import HeuristicContext, HeuristicRegistry

try:
    import torch
    import faiss
    from stark_qa.load_skb import SKB
except ImportError:
    fail_on_missing_imports(
                modules=["torch", "stark_qa", "faiss"],
                extra_name="stark"
            )

# Import GPU vector store (optional, fallback gracefully)
try:
    from .gpu_vector_store import GPUVectorStore
    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False

# Import GPU graph store (optional)
try:
    from .gpu_graph_store import GPUGraphStore
    _GPU_GRAPH_AVAILABLE = True
except ImportError:
    _GPU_GRAPH_AVAILABLE = False

import logging
logger = logging.getLogger(__name__)    

AVG_DEGREE_BY_DATASET = {
    "prime": 125.2,
    "amazon": 18.2,
    "mag": 43.5,
}

AVG_LOG_DEGREE_BY_DATASET = {
    k: math.log(1 + v)
    for k, v in AVG_DEGREE_BY_DATASET.items()
}

# --- 1. Graph Adapter for STaRK SKB ---
class StarkSKBAdapter(GraphStore):
    def __init__(
        self, 
        skb_data: 'SKB', 
        dataset: str, 
        cache_neighbors: bool = True,
        cache_degrees: bool = True,
        neighbor_cache_size: int = 10000,
        degree_cache_size: int = 10000,
        adjacency_dict: Optional[Dict[int, List[int]]] = None,
        cache_path=None
    ):
        """
        Uses CSR (Compressed Sparse Row) representation to ingest a STaRK SKB object. 
        """
        self.skb = skb_data
        if dataset not in AVG_LOG_DEGREE_BY_DATASET:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.dataset = dataset
        self.cache_path = cache_path or f"./adjacency_cache/graph_{dataset}.npz"
        self.avg_log_degree = AVG_LOG_DEGREE_BY_DATASET[dataset]
        
        self.cache_neighbors = cache_neighbors
        self.cache_degrees = cache_degrees

        if os.path.exists(self.cache_path):
            logger.info(f"Loading CSR graph from {self.cache_path}...")
            self.adj_matrix = sp.load_npz(self.cache_path)
            self.use_precomputed = True
        elif adjacency_dict is not None:
            logger.info("Converting dict to CSR and saving for next time...")
            self.adj_matrix = self._dict_to_csr(adjacency_dict)
            sp.save_npz(self.cache_path, self.adj_matrix)
            self.use_precomputed = True
        else:
            logger.warning("No pre-computed adjacency provided. Falling back to LRU cache (Slow).")
            self.use_precomputed = False
            
            if self.cache_neighbors:
                self.neighbor_cache = LRUCache(neighbor_cache_size)
            if self.cache_degrees:
                self.degree_cache = LRUCache(degree_cache_size)

    def _dict_to_csr(self, adj_dict: Dict[int, List[int]]) -> sp.csr_matrix:
        """Converts adjacency dictionary to a memory-efficient CSR matrix."""
        nodes = sorted(adj_dict.keys())
        max_node = nodes[-1] if nodes else 0
        
        indptr = [0]
        indices = []
        # We use a mapping if node IDs are non-contiguous, but STaRK is usually 0-indexed
        for i in range(max_node + 1):
            neighbors = adj_dict.get(i, [])
            indices.extend(neighbors)
            indptr.append(len(indices))
        
        # Use dummy data (1s) as we only care about topology/indices
        data = np.ones(len(indices), dtype=np.int8)
        return sp.csr_matrix((data, indices, indptr), shape=(max_node + 1, max_node + 1))

    def get_neighbors(self, node_id: int) -> np.ndarray:
        if self.use_precomputed:
            if node_id >= self.adj_matrix.shape[0]: return np.array([])
            start = self.adj_matrix.indptr[node_id]
            end = self.adj_matrix.indptr[node_id+1]
            # Treat as READ-ONLY.
            return self.adj_matrix.indices[start:end]
        
        if self.cache_neighbors:
            cached = self.neighbor_cache.get(node_id)
            if cached is not None: return np.array(cached)
        
        neighbors = self.skb.get_neighbor_nodes(node_id)
        np_neighbors = np.array(neighbors)
        
        if self.cache_neighbors:
            self.neighbor_cache.set(node_id, np_neighbors)
            
        return np_neighbors
            
    def get_degree(self, node_id: int) -> int:
        if self.use_precomputed:
            if node_id >= self.adj_matrix.shape[0]: return 0
            return self.adj_matrix.indptr[node_id+1] - self.adj_matrix.indptr[node_id]
        
        if self.cache_degrees:
            cached = self.degree_cache.get(node_id)
            if cached is not None: return cached

        neighbors = self.get_neighbors(node_id)
        deg = len(neighbors)
        
        if self.cache_degrees:
            self.degree_cache.set(node_id, deg)
            
        return deg

    def contains(self, node_id: int) -> bool:
        if self.use_precomputed:
            return node_id < self.adj_matrix.shape[0]
        return self.skb.node_info.get(node_id, "") != ""

    def get_avg_degree(self) -> float:
        """Returns the average degree of the graph"""
        if self.dataset in AVG_DEGREE_BY_DATASET:
            return AVG_DEGREE_BY_DATASET[self.dataset]
        return 10.0 # Default fallback
    
    @staticmethod
    @HeuristicRegistry.register_movement(HeuristicKey.STARK_CENTRALITY)
    def centrality_heuristic(ctx :HeuristicContext) -> np.ndarray:
        """
        Vectorized centrality heuristic.
        Uses pre-fetched ctx.node_degrees for speed.
        """
        graph: StarkSKBAdapter = ctx.graph
        log_degrees = np.log(1 + ctx.node_degrees)

        #Sigmoid normalization
        normalized = log_degrees / (log_degrees + graph.avg_log_degree + 1e-8)

        return normalized


# --- 1b. GPU-Accelerated Graph Adapter for STaRK SKB ---
class StarkGPUGraphAdapter(GraphStore):
    """
    GPU-accelerated graph adapter for STaRK datasets.

    Combines CSR storage with optional GPU acceleration for batch operations.
    Falls back to CPU for single-node queries to maintain interface compatibility.

    Usage:
        # Auto-detect GPU
        adapter = StarkGPUGraphAdapter(skb, "prime", adjacency_dict=adj)

        # Force CPU
        adapter = StarkGPUGraphAdapter(skb, "prime", adjacency_dict=adj, use_gpu=False)

        # Batch operations (GPU-accelerated)
        neighbors, mask = adapter.get_neighbors_batch([1, 2, 3])
    """

    def __init__(
        self,
        skb_data: 'SKB',
        dataset: str,
        adjacency_dict: Optional[Dict[int, List[int]]] = None,
        cache_path: Optional[str] = None,
        use_gpu: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize GPU graph adapter.

        Args:
            skb_data: STaRK SKB object
            dataset: Dataset name (e.g., "prime", "amazon", "mag")
            adjacency_dict: Pre-computed adjacency dictionary
            cache_path: Path to cache CSR matrix
            use_gpu: Whether to enable GPU acceleration
            device: Force specific device ("cuda" or "cpu")
        """
        self.skb = skb_data
        if dataset not in AVG_LOG_DEGREE_BY_DATASET:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.dataset = dataset
        self.cache_path = cache_path or f"./adjacency_cache/graph_{dataset}.npz"
        self.avg_log_degree = AVG_LOG_DEGREE_BY_DATASET[dataset]

        # Load or create CSR matrix (always kept for fallback)
        if os.path.exists(self.cache_path):
            logger.info(f"Loading CSR graph from {self.cache_path}...")
            self.adj_matrix = sp.load_npz(self.cache_path)
        elif adjacency_dict is not None:
            logger.info("Converting dict to CSR and saving...")
            self.adj_matrix = self._dict_to_csr(adjacency_dict)
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            sp.save_npz(self.cache_path, self.adj_matrix)
        else:
            raise ValueError("Must provide adjacency_dict or existing cache_path")

        # Setup GPU store if requested and available
        self._use_gpu = use_gpu and _GPU_GRAPH_AVAILABLE
        self._device = device or (get_device() if use_gpu else "cpu")
        self._gpu_store = None

        if self._use_gpu and self._device == "cuda":
            try:
                logger.info("Initializing GPU graph store...")
                self._gpu_store = GPUGraphStore.from_csr(
                    self.adj_matrix,
                    device=self._device,
                    avg_degree=AVG_DEGREE_BY_DATASET.get(dataset, 10.0)
                )
                logger.info(f"GPU graph store ready on {self._device}")
            except Exception as e:
                logger.warning(f"GPU graph initialization failed: {e}. Falling back to CPU.")
                self._gpu_store = None
                self._use_gpu = False
                self._device = "cpu"
        else:
            self._use_gpu = False
            self._device = "cpu"

    def _dict_to_csr(self, adj_dict: Dict[int, List[int]]) -> sp.csr_matrix:
        """Convert adjacency dictionary to CSR matrix."""
        nodes = sorted(adj_dict.keys())
        max_node = nodes[-1] if nodes else 0

        indptr = [0]
        indices = []

        for i in range(max_node + 1):
            neighbors = adj_dict.get(i, [])
            indices.extend(neighbors)
            indptr.append(len(indices))

        data = np.ones(len(indices), dtype=np.int8)
        return sp.csr_matrix((data, indices, indptr), shape=(max_node + 1, max_node + 1))

    def get_neighbors(self, node_id: int) -> np.ndarray:
        """
        Get neighbors for a single node.

        Uses CPU CSR for single queries (avoids GPU kernel overhead).
        """
        if node_id >= self.adj_matrix.shape[0]:
            return np.array([], dtype=np.int64)

        start = self.adj_matrix.indptr[node_id]
        end = self.adj_matrix.indptr[node_id + 1]
        return self.adj_matrix.indices[start:end]

    def get_neighbors_batch(self, node_ids):
        """
        Batch neighbor lookup (GPU-accelerated when available).

        Args:
            node_ids: List/array of node IDs

        Returns:
            Tuple of (neighbors tensor, mask tensor) if GPU available,
            otherwise (list of neighbor arrays, None)
        """
        if self._gpu_store is not None:
            return self._gpu_store.get_neighbors_batch(node_ids)

        # CPU fallback: return list of arrays
        neighbors_list = [self.get_neighbors(nid) for nid in node_ids]
        return neighbors_list, None

    def get_degrees_batch(self, node_ids):
        """
        Batch degree lookup (GPU-accelerated when available).

        Args:
            node_ids: List/array of node IDs

        Returns:
            Tensor of degrees if GPU available, otherwise numpy array
        """
        if self._gpu_store is not None:
            return self._gpu_store.get_degrees_batch(node_ids)

        # CPU fallback
        return np.array([self.get_degree(nid) for nid in node_ids], dtype=np.int32)

    def get_degree(self, node_id: int) -> int:
        """Get degree for a single node."""
        if node_id >= self.adj_matrix.shape[0]:
            return 0
        return self.adj_matrix.indptr[node_id + 1] - self.adj_matrix.indptr[node_id]

    def contains(self, node_id: int) -> bool:
        """Check if node exists in graph."""
        return node_id < self.adj_matrix.shape[0]

    def get_avg_degree(self) -> float:
        """Return average graph degree."""
        if self.dataset in AVG_DEGREE_BY_DATASET:
            return AVG_DEGREE_BY_DATASET[self.dataset]
        return 10.0

    @property
    def is_gpu(self) -> bool:
        """Check if GPU acceleration is active."""
        return self._gpu_store is not None

    @property
    def device(self) -> str:
        """Return current device."""
        return self._device

    @staticmethod
    def centrality_heuristic(ctx: HeuristicContext) -> np.ndarray:
        """
        Centrality heuristic compatible with GPU adapter.
        """
        graph = ctx.graph
        log_degrees = np.log(1 + ctx.node_degrees)
        avg_log = getattr(graph, 'avg_log_degree', AVG_LOG_DEGREE_BY_DATASET.get('prime', 4.8))
        normalized = log_degrees / (log_degrees + avg_log + 1e-8)
        return normalized


# --- 2. Vector Store Adapter for STaRK Tensors ---
class StarkInMemoryVectorStore(VectorStore):
    def __init__(self, doc_embs: Dict[int, torch.Tensor], shared_name: Optional[str] = None):
        """
        Wraps doc embeddings in SharedMemory to prevent 4x RAM usage in parallel mode.
        """
        self._parent_pid = os.getpid()
        faiss.omp_set_num_threads(1)

        raw_ids = np.array(list(doc_embs.keys()))
        self.id_dtype = raw_ids.dtype
        sorted_keys = np.sort(raw_ids)
        
        first_tensor = doc_embs[sorted_keys[0]].detach().cpu().numpy().squeeze()
        self.dim = first_tensor.shape[0]
        self.vec_dtype = first_tensor.dtype
        self.n_docs = len(sorted_keys)

        matrix_bytes = self.n_docs * self.dim * np.dtype(self.vec_dtype).itemsize
        id_bytes = self.n_docs * np.dtype(self.id_dtype).itemsize

        if shared_name:
            # CHILD PROCESS PATH
            self.shm_matrix = shared_memory.SharedMemory(name=shared_name)
            self.shm_ids = shared_memory.SharedMemory(name=f"{shared_name}_ids")
            self._is_owner = False
        else:
            # PARENT PROCESS PATH
            shared_name = f"stark_vstore_{os.getpid()}"
            self.shm_matrix = shared_memory.SharedMemory(create=True, size=matrix_bytes, name=shared_name)
            self.shm_ids = shared_memory.SharedMemory(create=True, size=id_bytes, name=f"{shared_name}_ids")
            self._is_owner = True

        self.matrix = np.ndarray((self.n_docs, self.dim), dtype=self.vec_dtype, buffer=self.shm_matrix.buf)
        self.ids = np.ndarray((self.n_docs,), dtype=self.id_dtype, buffer=self.shm_ids.buf)

        if self._is_owner:
            logger.info(f"Parent process: Populating shared memory for {self.n_docs} embeddings...")
            tensor_list = [doc_embs[rid] for rid in sorted_keys]
            stacked_matrix = torch.stack(tensor_list).detach().cpu().numpy().astype(self.vec_dtype).squeeze()
            faiss.normalize_L2(stacked_matrix)
            np.copyto(self.matrix, stacked_matrix)
            np.copyto(self.ids, sorted_keys)
            logger.info("Parent process: Shared memory populated.")
        
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.matrix)

        self.real_id_to_idx = {int(real_id): i for i, real_id in enumerate(self.ids)}

        # warmup
        _ = np.sum(self.matrix[0]) 
        _ = np.sum(self.ids[:min(100, len(self.ids))])

        atexit.register(self.close)

    def search(self, query_vec: np.ndarray, limit: int):
        # FAISS requires numpy on CPU
        query_vec_cpu = np.asarray(query_vec)
        q = query_vec_cpu.reshape(1, -1).astype('float32')
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, min(limit, self.n_docs))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            real_id = int(self.ids[idx])
            results.append({'id': real_id, 'score': float(score)})
        return results
    
    def fetch_batch(self, node_ids) -> Matrix:
        indices = [self.real_id_to_idx.get(nid, -1) for nid in node_ids]
        indices_arr = np.array(indices, dtype=np.int64)

        # Use NaN for missing vectors
        result = np.full((len(node_ids), self.dim), np.nan, dtype=self.vec_dtype)

        mask = (indices_arr >= 0)
        
        if np.any(mask):
            valid_internal_indices = indices_arr[mask]
            result[mask] = self.matrix[valid_internal_indices]
            
        return result
    
    def fetch(self, node_id: int) -> Optional[np.ndarray]:
        idx = self.real_id_to_idx.get(node_id)
        if idx is not None:
            return self.matrix[idx]
        logger.warning(
            f"VectorStore: Node ID {node_id} was requested but not found in the embedding map. "
            "This indicates a mismatch between graph nodes and document embeddings."
        )
        return None

    def close(self):
        for shm_attr in ['shm_matrix', 'shm_ids']:
            if hasattr(self, shm_attr):
                shm = getattr(self, shm_attr)
                shm.close()
                # Unlink ONLY in the parent process
                if self._is_owner and os.getpid() == self._parent_pid:
                    try:
                        shm.unlink()
                        logger.info(f"Unlinked {shm_attr}")
                    except:
                        pass


# --- 3. Embedding Adapter (Pre-computed Lookup) ---
class StarkPreComputedEmbeddingHandler(EmbeddingProvider):
    def __init__(self, query_embs: dict[int, torch.Tensor]):
        """Standardized NumPy pre-conversion to avoid GIL/Torch overhead."""
        self.query_embs = {}
        for qid, emb in query_embs.items():
            if hasattr(emb, 'numpy'):
                # Handle torch tensors (move to CPU if needed)
                arr = emb.cpu().detach().numpy() if hasattr(emb, 'is_cuda') and emb.is_cuda else emb.numpy()
            else:
                arr = np.array(emb)
            
            self.query_embs[qid] = arr.squeeze()

    def embed_query(self, query_id: int) -> np.ndarray:
        return self.query_embs[query_id]
        
    def embed_query_batch(self, query_ids: list[int]) -> Matrix:
        """
        Returns a 2D matrix (N_queries, Dimension) of pre-computed embeddings.
        """
        # Fetch individual arrays and stack them into a single (N, D) matrix
        return np.stack([self.query_embs[qid] for qid in query_ids])


# --- 4. GPU-Accelerated Vector Store ---
class StarkGPUVectorStore(VectorStore):
    """
    GPU-accelerated vector store using PyTorch for STaRK datasets.

    This class wraps GPUVectorStore with STaRK-specific optimizations:
    - Automatic device detection with env var override (SWARM_RAG_DEVICE)
    - Fallback to FAISS-based store if GPU unavailable
    - Compatible with existing VectorStore interface

    Usage:
        # Auto-detect GPU
        store = StarkGPUVectorStore(doc_embs)

        # Force CPU
        store = StarkGPUVectorStore(doc_embs, use_gpu=False)
    """

    def __init__(
        self,
        doc_embs: Dict[int, torch.Tensor],
        use_gpu: bool = True,
        device: str = None
    ):
        """
        Initialize GPU vector store.

        Args:
            doc_embs: Dictionary mapping doc_id -> embedding tensor
            use_gpu: Whether to use GPU acceleration (default True)
            device: Force specific device ("cuda" or "cpu")
        """
        self._device = device or (get_device() if use_gpu else "cpu")
        self._use_gpu = use_gpu and self._device == "cuda" and _GPU_AVAILABLE

        if self._use_gpu:
            logger.info(f"Initializing GPU vector store on {self._device}")
            self._store = GPUVectorStore.from_dict(doc_embs, device=self._device)
            self.n_docs = self._store.n_docs
            self.dim = self._store.dim
        else:
            logger.info("GPU not available/requested, using FAISS backend")
            # Fall back to shared-memory FAISS store
            self._store = StarkInMemoryVectorStore(doc_embs)
            self.n_docs = self._store.n_docs
            self.dim = self._store.dim

        # Build ID lookup
        self._ids = np.array(sorted(doc_embs.keys()))
        self.real_id_to_idx = {int(rid): i for i, rid in enumerate(self._ids)}

    def search(self, query_vec: np.ndarray, limit: int) -> List[Dict]:
        """Find top-k most similar documents."""
        return self._store.search(query_vec, limit)

    def fetch(self, node_id: int) -> Optional[np.ndarray]:
        """Fetch embedding for a single document."""
        return self._store.fetch(node_id)

    def fetch_batch(self, node_ids: List[int]) -> Matrix:
        """Fetch embeddings for multiple documents."""
        return self._store.fetch_batch(node_ids)

    def fetch_batch_gpu(self, node_ids: List[int]):
        """
        Fetch embeddings as GPU tensor (only available when using GPU backend).

        Returns:
            Tuple of (embeddings tensor, valid_ids list) if GPU,
            or (numpy array, valid_ids) if CPU
        """
        if self._use_gpu and hasattr(self._store, 'fetch_batch_gpu'):
            return self._store.fetch_batch_gpu(node_ids)

        # Fallback for CPU: return numpy
        matrix = self.fetch_batch(node_ids)
        valid_mask = ~np.isnan(matrix).any(axis=1)
        valid_ids = [nid for i, nid in enumerate(node_ids) if valid_mask[i]]
        return matrix[valid_mask], valid_ids

    def compute_similarities(
        self,
        query_vec: Union[np.ndarray, torch.Tensor],
        candidate_ids: List[int]
    ):
        """
        Compute similarities between query and candidates.

        Optimized for GPU when available.
        """
        if self._use_gpu and hasattr(self._store, 'compute_similarities'):
            return self._store.compute_similarities(query_vec, candidate_ids)

        # CPU fallback
        matrix = self.fetch_batch(candidate_ids)
        valid_mask = ~np.isnan(matrix).any(axis=1)
        valid_ids = [nid for i, nid in enumerate(candidate_ids) if valid_mask[i]]

        if len(valid_ids) == 0:
            return np.array([]), []

        valid_matrix = matrix[valid_mask]
        query_vec = np.asarray(query_vec).flatten()
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        # Cosine similarity
        similarities = np.dot(valid_matrix, query_vec)
        return similarities, valid_ids

    @property
    def device(self) -> str:
        """Return current device."""
        return self._device

    @property
    def is_gpu(self) -> bool:
        """Check if using GPU backend."""
        return self._use_gpu

    def close(self):
        """Clean up resources (delegates to underlying store if needed)."""
        if hasattr(self._store, 'close'):
            self._store.close()


def create_stark_vector_store(
    doc_embs: Dict[int, torch.Tensor],
    use_gpu: str = "auto",
    shared_name: Optional[str] = None
) -> VectorStore:
    """
    Factory function to create the appropriate vector store.

    Args:
        doc_embs: Dictionary mapping doc_id -> embedding tensor
        use_gpu: "auto", "always", "never"
        shared_name: For shared memory (FAISS backend only)

    Returns:
        VectorStore instance (GPU or FAISS-backed)
    """
    if use_gpu == "never":
        return StarkInMemoryVectorStore(doc_embs, shared_name=shared_name)

    if use_gpu == "always":
        if not _GPU_AVAILABLE:
            raise RuntimeError("GPU requested but GPUVectorStore not available")
        return StarkGPUVectorStore(doc_embs, use_gpu=True)

    # Auto mode
    device = get_device()
    if device == "cuda" and _GPU_AVAILABLE:
        return StarkGPUVectorStore(doc_embs, use_gpu=True)

    return StarkInMemoryVectorStore(doc_embs, shared_name=shared_name)