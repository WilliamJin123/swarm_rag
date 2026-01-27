"""
STaRK Dataset Integration

Provides STaRK-specific wrappers around TorchVectorStore and TorchGraphStore
with dataset-specific features: CSR caching, centrality heuristics, etc.
"""

import math
import os
from typing import Dict, List, Optional, Tuple, Union

import scipy.sparse as sp
import torch

from ..interfaces.abstract_classes import VectorStore, GraphStore, EmbeddingProvider, Matrix
from ..interfaces.enums import HeuristicKey
from ..utils import fail_on_missing_imports, get_device
from ..core import HeuristicContext, HeuristicRegistry

try:
    from stark_qa.load_skb import SKB
except ImportError:
    fail_on_missing_imports(
                modules=["torch", "stark_qa"],
                extra_name="stark"
            )

# Import torch stores
try:
    from .torch_vector_store import TorchVectorStore
    _TORCH_VECTOR_AVAILABLE = True
except ImportError:
    _TORCH_VECTOR_AVAILABLE = False

try:
    from .torch_graph_store import TorchGraphStore
    _TORCH_GRAPH_AVAILABLE = True
except ImportError:
    _TORCH_GRAPH_AVAILABLE = False

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


class StarkGraphAdapter(GraphStore):
    """
    Graph adapter for STaRK datasets.

    Stores graph in CSR format using torch tensors on the target device.
    GPU mode uses TorchGraphStore for optimized batch operations.
    CPU mode uses torch tensors directly.

    Usage:
        adapter = StarkGraphAdapter(skb, "prime", adjacency_dict=adj)
        adapter = StarkGraphAdapter(skb, "prime", adjacency_dict=adj, use_gpu=False)
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
        Initialize graph adapter.

        Args:
            skb_data: STaRK SKB object
            dataset: Dataset name (e.g., "prime", "amazon", "mag")
            adjacency_dict: Pre-computed adjacency dictionary
            cache_path: Path to cache CSR matrix
            use_gpu: Whether to enable GPU acceleration (default True, auto-fallback)
            device: Force specific device ("cuda" or "cpu")
        """
        self.skb = skb_data
        if dataset not in AVG_LOG_DEGREE_BY_DATASET:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.dataset = dataset
        self.cache_path = cache_path or f"./adjacency_cache/graph_{dataset}.npz"
        self.avg_log_degree = AVG_LOG_DEGREE_BY_DATASET[dataset]

        # Determine target device
        self._device = device or (get_device() if use_gpu else "cpu")
        self._use_gpu = use_gpu and _TORCH_GRAPH_AVAILABLE and self._device == "cuda"

        # Load or create CSR matrix (temporary, for loading/caching)
        csr_matrix = self._load_or_create_csr(adjacency_dict)

        if self._use_gpu:
            # GPU path: Create TorchGraphStore, then discard scipy CSR
            try:
                logger.info("Initializing GPU graph store...")
                self._torch_store = TorchGraphStore.from_csr(
                    csr_matrix,
                    device=self._device,
                    avg_degree=AVG_DEGREE_BY_DATASET.get(dataset, 10.0)
                )
                self._crow_indices = None
                self._col_indices = None
                self._degrees = None
                self._n_nodes = self._torch_store.n_nodes
                logger.info(f"GPU graph store ready on {self._device}")
            except Exception as e:
                logger.warning(f"GPU graph initialization failed: {e}. Falling back to CPU.")
                self._use_gpu = False
                self._device = "cpu"
                self._torch_store = None
                self._init_cpu_tensors(csr_matrix)
        else:
            self._torch_store = None
            self._init_cpu_tensors(csr_matrix)

        del csr_matrix

    def _load_or_create_csr(self, adjacency_dict: Optional[Dict[int, List[int]]]) -> sp.csr_matrix:
        """Load CSR from cache or create from adjacency dict."""
        if os.path.exists(self.cache_path):
            logger.info(f"Loading CSR graph from {self.cache_path}...")
            return sp.load_npz(self.cache_path)
        elif adjacency_dict is not None:
            logger.info("Converting dict to CSR and caching...")
            csr = self._dict_to_csr(adjacency_dict)
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            sp.save_npz(self.cache_path, csr)
            return csr
        else:
            raise ValueError("Must provide adjacency_dict or existing cache_path")

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

        data = [1] * len(indices)
        return sp.csr_matrix((data, indices, indptr), shape=(max_node + 1, max_node + 1), dtype='int8')

    def _init_cpu_tensors(self, csr: sp.csr_matrix):
        """Initialize torch tensors from scipy CSR for CPU operations."""
        self._n_nodes = csr.shape[0]
        self._crow_indices = torch.tensor(csr.indptr, dtype=torch.long, device=self._device)
        self._col_indices = torch.tensor(csr.indices, dtype=torch.long, device=self._device)
        self._degrees = self._crow_indices[1:] - self._crow_indices[:-1]

    def get_neighbors(self, node_id: int) -> torch.Tensor:
        """Get neighbors for a single node. Returns tensor on device."""
        if self._torch_store is not None:
            return self._torch_store.get_neighbors(node_id)

        if node_id < 0 or node_id >= self._n_nodes:
            return torch.tensor([], dtype=torch.long, device=self._device)

        start = self._crow_indices[node_id].item()
        end = self._crow_indices[node_id + 1].item()
        return self._col_indices[start:end]  # Already on device

    def get_neighbors_batch(
        self,
        node_ids: Union[List[int], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch neighbor lookup."""
        if self._torch_store is not None:
            return self._torch_store.get_neighbors_batch(node_ids)

        if isinstance(node_ids, list):
            ids = torch.tensor(node_ids, dtype=torch.long, device=self._device)
        else:
            ids = node_ids.to(device=self._device, dtype=torch.long)

        batch_size = ids.shape[0]
        if batch_size == 0:
            return (
                torch.empty((0, 0), device=self._device, dtype=torch.long),
                torch.empty((0, 0), device=self._device, dtype=torch.bool)
            )

        valid_mask = (ids >= 0) & (ids < self._n_nodes)
        clamped_ids = torch.clamp(ids, 0, self._n_nodes - 1)

        starts = self._crow_indices[clamped_ids]
        ends = self._crow_indices[clamped_ids + 1]
        lengths = torch.where(valid_mask, ends - starts, torch.zeros_like(ends))

        batch_max_degree = int(lengths.max().item()) if batch_size > 0 else 0

        if batch_max_degree == 0:
            return (
                torch.full((batch_size, 1), -1, device=self._device, dtype=torch.long),
                torch.zeros((batch_size, 1), device=self._device, dtype=torch.bool)
            )

        neighbors = torch.full((batch_size, batch_max_degree), -1, device=self._device, dtype=torch.long)
        mask = torch.zeros((batch_size, batch_max_degree), device=self._device, dtype=torch.bool)

        total_neighbors = lengths.sum().item()
        if total_neighbors > 0:
            lengths_int = lengths.int()
            row_indices = torch.repeat_interleave(
                torch.arange(batch_size, device=self._device),
                lengths_int
            )

            offsets = torch.zeros(total_neighbors, device=self._device, dtype=torch.long)
            cumsum = torch.cumsum(lengths, dim=0)
            if batch_size > 1:
                valid_offset_mask = cumsum[:-1] < total_neighbors
                if valid_offset_mask.any():
                    valid_cumsum_indices = cumsum[:-1][valid_offset_mask]
                    valid_lengths = lengths[:-1][valid_offset_mask]
                    offsets[valid_cumsum_indices] = valid_lengths
            col_positions = torch.arange(total_neighbors, device=self._device) - torch.cumsum(offsets, dim=0)

            flat_starts = torch.repeat_interleave(starts, lengths_int)
            flat_indices = torch.clamp(flat_starts + col_positions, 0, self._col_indices.shape[0] - 1)
            flat_neighbors = self._col_indices[flat_indices]

            col_positions = torch.clamp(col_positions, 0, batch_max_degree - 1)
            neighbors[row_indices, col_positions] = flat_neighbors
            mask[row_indices, col_positions] = True

        return neighbors, mask

    def get_degrees_batch(
        self,
        node_ids: Union[List[int], torch.Tensor]
    ) -> torch.Tensor:
        """Batch degree lookup."""
        if self._torch_store is not None:
            return self._torch_store.get_degrees_batch(node_ids)

        if isinstance(node_ids, list):
            ids = torch.tensor(node_ids, dtype=torch.long, device=self._device)
        else:
            ids = node_ids.to(device=self._device, dtype=torch.long)

        valid_mask = (ids >= 0) & (ids < self._n_nodes)
        clamped_ids = torch.clamp(ids, 0, self._n_nodes - 1)

        degrees = self._degrees[clamped_ids]
        return torch.where(valid_mask, degrees, torch.zeros_like(degrees))

    def get_degree(self, node_id: int) -> int:
        """Get degree for a single node."""
        if self._torch_store is not None:
            return self._torch_store.get_degree(node_id)

        if node_id < 0 or node_id >= self._n_nodes:
            return 0
        return int(self._degrees[node_id].item())

    def contains(self, node_id: int) -> bool:
        """Check if node exists in graph."""
        if self._torch_store is not None:
            return self._torch_store.contains(node_id)
        return 0 <= node_id < self._n_nodes

    def get_avg_degree(self) -> float:
        """Return average graph degree."""
        return AVG_DEGREE_BY_DATASET.get(self.dataset, 10.0)

    @property
    def is_gpu(self) -> bool:
        """Check if GPU acceleration is active."""
        return self._torch_store is not None

    @property
    def device(self) -> str:
        """Return current device."""
        return self._device

    @staticmethod
    @HeuristicRegistry.register_movement(HeuristicKey.STARK_CENTRALITY)
    def centrality_heuristic(ctx: HeuristicContext) -> torch.Tensor:
        """Vectorized centrality heuristic."""
        graph: StarkGraphAdapter = ctx.graph
        degrees = ctx.node_degrees if isinstance(ctx.node_degrees, torch.Tensor) else torch.as_tensor(ctx.node_degrees)
        log_degrees = torch.log(1 + degrees)
        avg_log = getattr(graph, 'avg_log_degree', AVG_LOG_DEGREE_BY_DATASET.get('prime', 4.8))
        return log_degrees / (log_degrees + avg_log + 1e-8)


class StarkVectorStore(VectorStore):
    """
    Vector store for STaRK datasets.

    All data stored as torch tensors on the target device.
    GPU mode delegates to TorchVectorStore for optimized operations.
    CPU mode uses the same tensor operations but on CPU.

    Usage:
        store = StarkVectorStore(doc_embs)  # Auto-detect GPU
        store = StarkVectorStore(doc_embs, use_gpu=False)  # Force CPU
        store = StarkVectorStore(doc_embs, device="cuda")  # Force GPU
    """

    def __init__(
        self,
        doc_embs: Dict[int, torch.Tensor],
        use_gpu: bool = True,
        device: str = None
    ):
        """
        Initialize vector store.

        Args:
            doc_embs: Dictionary mapping doc_id -> embedding tensor
            use_gpu: Whether to use GPU acceleration (default True, auto-fallback)
            device: Force specific device ("cuda" or "cpu")
        """
        self._device = device or (get_device() if use_gpu else "cpu")
        self._use_gpu = use_gpu and self._device == "cuda" and _TORCH_VECTOR_AVAILABLE

        if self._use_gpu:
            logger.info(f"Initializing GPU vector store on {self._device}")
            self._torch_store = TorchVectorStore.from_dict(doc_embs, device=self._device)
            self.n_docs = self._torch_store.n_docs
            self.dim = self._torch_store.dim
            self._ids = None
            self._embeddings = None
            self._id_to_idx = None
        else:
            logger.info(f"Initializing torch vector store on {self._device}")
            self._torch_store = None
            self._init_tensors(doc_embs)

    def _init_tensors(self, doc_embs: Dict[int, torch.Tensor]):
        """Initialize torch tensors from embedding dict."""
        sorted_keys = sorted(doc_embs.keys())
        self._ids = torch.tensor(sorted_keys, dtype=torch.long, device=self._device)
        self.n_docs = len(sorted_keys)

        first_tensor = doc_embs[sorted_keys[0]]
        self.dim = first_tensor.squeeze().shape[0]

        embeddings_list = []
        for rid in sorted_keys:
            emb = doc_embs[rid]
            if emb.device != torch.device(self._device):
                emb = emb.to(self._device)
            embeddings_list.append(emb.detach().squeeze())

        embeddings = torch.stack(embeddings_list)
        self._embeddings = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
        self._id_to_idx = {int(rid): i for i, rid in enumerate(sorted_keys)}

        logger.info(f"Vector store initialized: {self.n_docs} docs, dim={self.dim}, device={self._device}")

    def search(self, query_vec, limit: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find top-k most similar documents. Returns (ids, scores) tensors on device."""
        if self._torch_store is not None:
            return self._torch_store.search(query_vec, limit)

        if not isinstance(query_vec, torch.Tensor):
            query = torch.as_tensor(query_vec, dtype=torch.float32, device=self._device)
        else:
            query = query_vec.to(device=self._device, dtype=torch.float32)

        query = torch.nn.functional.normalize(query.view(1, -1), p=2, dim=1)
        similarities = torch.mm(query, self._embeddings.t()).squeeze(0)

        k = min(limit, self.n_docs)
        scores, indices = torch.topk(similarities, k=k, largest=True)

        # Map internal indices to real IDs (stay on device)
        real_ids = self._ids[indices]
        return real_ids, scores

    def search_batch(self, query_vecs: torch.Tensor, limit: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch search for multiple queries. Returns (ids, scores) tensors on device."""
        if self._torch_store is not None:
            return self._torch_store.search_batch(query_vecs, limit)

        if not isinstance(query_vecs, torch.Tensor):
            queries = torch.as_tensor(query_vecs, dtype=torch.float32, device=self._device)
        else:
            queries = query_vecs.to(device=self._device, dtype=torch.float32)

        queries = torch.nn.functional.normalize(queries, p=2, dim=1)
        similarities = torch.mm(queries, self._embeddings.t())

        k = min(limit, self.n_docs)
        scores, indices = torch.topk(similarities, k=k, dim=1, largest=True)

        # Map internal indices to real IDs (stay on device)
        real_ids = self._ids[indices]
        return real_ids, scores

    def fetch(self, node_id: int) -> Optional[torch.Tensor]:
        """Fetch embedding for a single document. Returns tensor on device."""
        if self._torch_store is not None:
            return self._torch_store.fetch(node_id)

        idx = self._id_to_idx.get(node_id)
        if idx is not None:
            return self._embeddings[idx]
        logger.warning(f"VectorStore: Node ID {node_id} not found")
        return None

    def fetch_batch(self, node_ids: Union[List[int], torch.Tensor]) -> Matrix:
        """Fetch embeddings for multiple documents. Returns tensor on device."""
        if self._torch_store is not None:
            return self._torch_store.fetch_batch(node_ids)

        # Convert tensor to list if needed for dict lookup
        if isinstance(node_ids, torch.Tensor):
            node_ids_list = node_ids.tolist()
        else:
            node_ids_list = list(node_ids)

        result = torch.full((len(node_ids_list), self.dim), float('nan'), dtype=torch.float32, device=self._device)

        valid_pairs = []
        for out_idx, nid in enumerate(node_ids_list):
            int_idx = self._id_to_idx.get(nid)
            if int_idx is not None:
                valid_pairs.append((out_idx, int_idx))

        if valid_pairs:
            out_indices, int_indices = zip(*valid_pairs)
            int_indices_t = torch.tensor(int_indices, dtype=torch.long, device=self._device)
            out_indices_t = torch.tensor(out_indices, dtype=torch.long, device=self._device)
            result[out_indices_t] = self._embeddings[int_indices_t]

        return result

    def fetch_batch_gpu(self, node_ids: Union[List[int], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch embeddings as tensor with valid IDs. Returns tensors on device."""
        if self._torch_store is not None and hasattr(self._torch_store, 'fetch_batch_gpu'):
            return self._torch_store.fetch_batch_gpu(node_ids)

        # Convert to list for dict lookup
        if isinstance(node_ids, torch.Tensor):
            node_ids_list = node_ids.tolist()
        else:
            node_ids_list = list(node_ids)

        valid_pairs = []
        for nid in node_ids_list:
            int_idx = self._id_to_idx.get(nid)
            if int_idx is not None:
                valid_pairs.append((nid, int_idx))

        if not valid_pairs:
            return (
                torch.empty((0, self.dim), device=self._device, dtype=torch.float32),
                torch.empty(0, device=self._device, dtype=torch.long)
            )

        valid_nids, int_indices = zip(*valid_pairs)
        int_indices_t = torch.tensor(int_indices, dtype=torch.long, device=self._device)
        embeddings = self._embeddings[int_indices_t]
        valid_ids = torch.tensor(valid_nids, dtype=torch.long, device=self._device)

        return embeddings, valid_ids

    def compute_similarities(
        self,
        query_vec: Union[torch.Tensor, List],
        candidate_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute similarities between query and candidates."""
        if self._torch_store is not None and hasattr(self._torch_store, 'compute_similarities'):
            return self._torch_store.compute_similarities(query_vec, candidate_ids)

        candidate_embs, valid_ids = self.fetch_batch_gpu(candidate_ids)

        if valid_ids.numel() == 0:
            return torch.empty(0, device=self._device), valid_ids

        if not isinstance(query_vec, torch.Tensor):
            query = torch.as_tensor(query_vec, dtype=torch.float32, device=self._device)
        else:
            query = query_vec.to(device=self._device, dtype=torch.float32)

        query = torch.nn.functional.normalize(query.view(1, -1), p=2, dim=1)
        similarities = torch.mm(query, candidate_embs.t()).squeeze(0)

        return similarities, valid_ids

    @property
    def device(self) -> str:
        """Return current device."""
        return self._device

    @property
    def is_gpu(self) -> bool:
        """Check if using GPU backend."""
        return self._torch_store is not None

    def close(self):
        """Clean up resources."""
        if self._torch_store is not None and hasattr(self._torch_store, 'close'):
            self._torch_store.close()

        if self._embeddings is not None:
            del self._embeddings
            self._embeddings = None
        if self._ids is not None:
            del self._ids
            self._ids = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class StarkPreComputedEmbeddingHandler(EmbeddingProvider):
    """
    Pre-computed embedding lookup.

    Stores embeddings as CPU tensors. Caller is responsible for
    moving to GPU if needed.
    """

    def __init__(self, query_embs: dict[int, torch.Tensor]):
        """Initialize from embedding dictionary."""
        self.query_embs = {}
        for qid, emb in query_embs.items():
            if isinstance(emb, torch.Tensor):
                tensor = emb.cpu().detach().squeeze()
            else:
                tensor = torch.as_tensor(emb).squeeze()
            self.query_embs[qid] = tensor

    def embed_query(self, query_id: int) -> torch.Tensor:
        return self.query_embs[query_id]

    def embed_query_batch(self, query_ids: list[int]) -> Matrix:
        """Returns a 2D matrix (N_queries, Dimension) of embeddings."""
        return torch.stack([self.query_embs[qid] for qid in query_ids])
