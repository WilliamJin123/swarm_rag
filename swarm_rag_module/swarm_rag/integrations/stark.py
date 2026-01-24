import math
import os
from typing import Dict, List, Optional, Union
import scipy.sparse as sp

import torch

from ..interfaces.abstract_classes import VectorStore, GraphStore, EmbeddingProvider, Matrix
from ..interfaces.enums import HeuristicKey
from ..utils import fail_on_missing_imports, LRUCache, get_device
from ..core import HeuristicContext, HeuristicRegistry

try:
    from stark_qa.load_skb import SKB
except ImportError:
    fail_on_missing_imports(
                modules=["torch", "stark_qa"],
                extra_name="stark"
            )

# Import GPU vector store (optional, fallback gracefully)
# TorchVectorStore is a device-agnostic alias for GPUVectorStore
try:
    from .gpu_vector_store import GPUVectorStore, TorchVectorStore
    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False
    TorchVectorStore = None

# Import GPU graph store (optional)
# TorchGraphStore is a device-agnostic alias for GPUGraphStore
try:
    from .gpu_graph_store import GPUGraphStore, TorchGraphStore
    _GPU_GRAPH_AVAILABLE = True
except ImportError:
    _GPU_GRAPH_AVAILABLE = False
    TorchGraphStore = None

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
        # Keep as list for scipy CSR construction
        data = [1] * len(indices)
        return sp.csr_matrix((data, indices, indptr), shape=(max_node + 1, max_node + 1), dtype='int8')

    def get_neighbors(self, node_id: int) -> torch.Tensor:
        """Get neighbors for a single node as a torch tensor."""
        if self.use_precomputed:
            if node_id >= self.adj_matrix.shape[0]:
                return torch.tensor([], dtype=torch.long)
            start = self.adj_matrix.indptr[node_id]
            end = self.adj_matrix.indptr[node_id+1]
            return torch.tensor(self.adj_matrix.indices[start:end], dtype=torch.long)

        if self.cache_neighbors:
            cached = self.neighbor_cache.get(node_id)
            if cached is not None:
                return torch.as_tensor(cached, dtype=torch.long)

        neighbors = self.skb.get_neighbor_nodes(node_id)
        neighbors_tensor = torch.tensor(neighbors, dtype=torch.long)

        if self.cache_neighbors:
            self.neighbor_cache.set(node_id, neighbors_tensor)

        return neighbors_tensor
            
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
    def centrality_heuristic(ctx :HeuristicContext) -> torch.Tensor:
        """
        Vectorized centrality heuristic.
        Uses pre-fetched ctx.node_degrees for speed.
        """
        graph: StarkSKBAdapter = ctx.graph
        degrees = ctx.node_degrees if isinstance(ctx.node_degrees, torch.Tensor) else torch.as_tensor(ctx.node_degrees)
        log_degrees = torch.log(1 + degrees)

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

        # Use list for scipy CSR construction
        data = [1] * len(indices)
        return sp.csr_matrix((data, indices, indptr), shape=(max_node + 1, max_node + 1), dtype='int8')

    def get_neighbors(self, node_id: int) -> torch.Tensor:
        """
        Get neighbors for a single node.

        Uses CPU CSR for single queries (avoids GPU kernel overhead).
        Returns torch.Tensor for consistency.
        """
        if node_id >= self.adj_matrix.shape[0]:
            return torch.tensor([], dtype=torch.long)

        start = self.adj_matrix.indptr[node_id]
        end = self.adj_matrix.indptr[node_id + 1]
        return torch.tensor(self.adj_matrix.indices[start:end], dtype=torch.long)

    def get_neighbors_batch(self, node_ids):
        """
        Batch neighbor lookup (GPU-accelerated when available).

        Args:
            node_ids: List/array of node IDs

        Returns:
            Tuple of (neighbors tensor, mask tensor) if GPU available,
            otherwise (list of neighbor tensors, None)
        """
        if self._gpu_store is not None:
            return self._gpu_store.get_neighbors_batch(node_ids)

        # CPU fallback: return list of tensors
        neighbors_list = [self.get_neighbors(nid) for nid in node_ids]
        return neighbors_list, None

    def get_degrees_batch(self, node_ids):
        """
        Batch degree lookup (GPU-accelerated when available).

        Args:
            node_ids: List/array of node IDs

        Returns:
            Tensor of degrees
        """
        if self._gpu_store is not None:
            return self._gpu_store.get_degrees_batch(node_ids)

        # CPU fallback - return tensor
        return torch.tensor([self.get_degree(nid) for nid in node_ids], dtype=torch.int32)

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
    def centrality_heuristic(ctx: HeuristicContext) -> torch.Tensor:
        """
        Centrality heuristic compatible with GPU adapter.
        """
        graph = ctx.graph
        degrees = ctx.node_degrees if isinstance(ctx.node_degrees, torch.Tensor) else torch.as_tensor(ctx.node_degrees)
        log_degrees = torch.log(1 + degrees)
        avg_log = getattr(graph, 'avg_log_degree', AVG_LOG_DEGREE_BY_DATASET.get('prime', 4.8))
        normalized = log_degrees / (log_degrees + avg_log + 1e-8)
        return normalized


# --- 2. Vector Store Adapter for STaRK Tensors ---
class StarkInMemoryVectorStore(VectorStore):
    """
    Pure PyTorch in-memory vector store for STaRK datasets.

    Replaces the FAISS-based implementation with torch operations:
    - Uses torch.mm() + torch.topk() for similarity search
    - Uses torch.nn.functional.normalize for L2 normalization
    - No shared memory complexity - torch tensors are copy-on-read in forked processes
    """

    def __init__(self, doc_embs: Dict[int, torch.Tensor], shared_name: Optional[str] = None):
        """
        Initialize vector store from embedding dictionary.

        Args:
            doc_embs: Dictionary mapping doc_id -> embedding tensor
            shared_name: Ignored (kept for API compatibility)
        """
        # Sort keys for deterministic ordering
        sorted_keys = sorted(doc_embs.keys())
        self._ids = torch.tensor(sorted_keys, dtype=torch.long)
        self.n_docs = len(sorted_keys)

        # Get dimensions from first embedding
        first_tensor = doc_embs[sorted_keys[0]].detach().cpu().squeeze()
        self.dim = first_tensor.shape[0]

        # Stack and normalize embeddings
        embeddings = torch.stack([doc_embs[rid].detach().cpu().squeeze() for rid in sorted_keys])
        self._embeddings = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)

        # Build ID lookup
        self._id_to_idx = {int(rid): i for i, rid in enumerate(sorted_keys)}

        logger.info(f"StarkInMemoryVectorStore initialized: {self.n_docs} docs, dim={self.dim}")

    def search(self, query_vec, limit: int) -> List[Dict]:
        """Find top-k most similar documents using cosine similarity."""
        # Ensure query is tensor and normalized
        if not isinstance(query_vec, torch.Tensor):
            query = torch.as_tensor(query_vec, dtype=torch.float32)
        else:
            query = query_vec.to(dtype=torch.float32)

        query = query.view(1, -1)
        query = torch.nn.functional.normalize(query, p=2, dim=1)

        # Compute similarities via matrix multiplication
        similarities = torch.mm(query, self._embeddings.t()).squeeze(0)

        # Get top-k
        k = min(limit, self.n_docs)
        scores, indices = torch.topk(similarities, k=k, largest=True)

        # Convert to list of dicts
        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            real_id = int(self._ids[idx])
            results.append({'id': real_id, 'score': float(score)})

        return results

    def fetch_batch(self, node_ids) -> torch.Tensor:
        """Fetch embeddings for multiple documents."""
        result = torch.full((len(node_ids), self.dim), float('nan'), dtype=torch.float32)

        for i, nid in enumerate(node_ids):
            idx = self._id_to_idx.get(nid)
            if idx is not None:
                result[i] = self._embeddings[idx]

        return result

    def fetch(self, node_id: int) -> Optional[torch.Tensor]:
        """Fetch embedding for a single document."""
        idx = self._id_to_idx.get(node_id)
        if idx is not None:
            return self._embeddings[idx]
        logger.warning(
            f"VectorStore: Node ID {node_id} was requested but not found in the embedding map. "
            "This indicates a mismatch between graph nodes and document embeddings."
        )
        return None

    def close(self):
        """No-op for compatibility. Torch tensors don't need explicit cleanup."""
        pass


# --- 3. Embedding Adapter (Pre-computed Lookup) ---
class StarkPreComputedEmbeddingHandler(EmbeddingProvider):
    def __init__(self, query_embs: dict[int, torch.Tensor]):
        """Standardized pre-conversion to torch tensors."""
        self.query_embs = {}
        for qid, emb in query_embs.items():
            if isinstance(emb, torch.Tensor):
                # Handle torch tensors (move to CPU if needed)
                tensor = emb.cpu().detach().squeeze()
            else:
                tensor = torch.as_tensor(emb).squeeze()

            self.query_embs[qid] = tensor

    def embed_query(self, query_id: int) -> torch.Tensor:
        return self.query_embs[query_id]

    def embed_query_batch(self, query_ids: list[int]) -> Matrix:
        """
        Returns a 2D matrix (N_queries, Dimension) of pre-computed embeddings.
        """
        # Fetch individual tensors and stack them into a single (N, D) tensor
        return torch.stack([self.query_embs[qid] for qid in query_ids])


# --- 4. GPU-Accelerated Vector Store ---
class StarkGPUVectorStore(VectorStore):
    """
    GPU-accelerated vector store using PyTorch for STaRK datasets.

    This class wraps GPUVectorStore with STaRK-specific optimizations:
    - Automatic device detection with env var override (SWARM_RAG_DEVICE)
    - Fallback to PyTorch-based store if GPU unavailable
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
            logger.info("GPU not available/requested, using PyTorch CPU backend")
            # Fall back to PyTorch-based store
            self._store = StarkInMemoryVectorStore(doc_embs)
            self.n_docs = self._store.n_docs
            self.dim = self._store.dim

        # Build ID lookup using torch
        self._ids = torch.tensor(sorted(doc_embs.keys()), dtype=torch.long)
        self.real_id_to_idx = {int(rid): i for i, rid in enumerate(self._ids.tolist())}

    def search(self, query_vec, limit: int) -> List[Dict]:
        """Find top-k most similar documents."""
        return self._store.search(query_vec, limit)

    def fetch(self, node_id: int) -> Optional[torch.Tensor]:
        """Fetch embedding for a single document."""
        return self._store.fetch(node_id)

    def fetch_batch(self, node_ids: List[int]) -> Matrix:
        """Fetch embeddings for multiple documents."""
        return self._store.fetch_batch(node_ids)

    def fetch_batch_gpu(self, node_ids: List[int]):
        """
        Fetch embeddings as GPU tensor (only available when using GPU backend).

        Returns:
            Tuple of (embeddings tensor, valid_ids tensor/list)
        """
        if self._use_gpu and hasattr(self._store, 'fetch_batch_gpu'):
            return self._store.fetch_batch_gpu(node_ids)

        # Fallback for CPU: return tensor
        matrix = self.fetch_batch(node_ids)
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.as_tensor(matrix, dtype=torch.float32)
        valid_mask = ~torch.isnan(matrix).any(dim=1)
        valid_ids = [nid for i, nid in enumerate(node_ids) if valid_mask[i]]
        return matrix[valid_mask], valid_ids

    def compute_similarities(
        self,
        query_vec: Union[torch.Tensor, List],
        candidate_ids: List[int]
    ):
        """
        Compute similarities between query and candidates.

        Optimized for GPU when available.
        """
        if self._use_gpu and hasattr(self._store, 'compute_similarities'):
            return self._store.compute_similarities(query_vec, candidate_ids)

        # CPU fallback using torch
        matrix = self.fetch_batch(candidate_ids)
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.as_tensor(matrix, dtype=torch.float32)

        valid_mask = ~torch.isnan(matrix).any(dim=1)
        valid_ids = [nid for i, nid in enumerate(candidate_ids) if valid_mask[i]]

        if len(valid_ids) == 0:
            return torch.tensor([]), []

        valid_matrix = matrix[valid_mask]

        if not isinstance(query_vec, torch.Tensor):
            query_vec = torch.as_tensor(query_vec, dtype=torch.float32)
        query_vec = query_vec.flatten()
        query_vec = query_vec / (torch.linalg.norm(query_vec) + 1e-8)

        # Cosine similarity
        similarities = torch.matmul(valid_matrix, query_vec)
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
        shared_name: Ignored (kept for API compatibility)

    Returns:
        VectorStore instance (GPU or PyTorch CPU-backed)
    """
    if use_gpu == "never":
        return StarkInMemoryVectorStore(doc_embs)

    if use_gpu == "always":
        if not _GPU_AVAILABLE:
            raise RuntimeError("GPU requested but GPUVectorStore not available")
        return StarkGPUVectorStore(doc_embs, use_gpu=True)

    # Auto mode
    device = get_device()
    if device == "cuda" and _GPU_AVAILABLE:
        return StarkGPUVectorStore(doc_embs, use_gpu=True)

    return StarkInMemoryVectorStore(doc_embs)