"""
STaRK Dataset Integration

Thin wrappers around TorchVectorStore and TorchGraphStore
with STaRK-specific features: CSR caching, centrality heuristics, etc.
"""

import math
import os
from typing import Dict, List, Optional, Tuple, Union

import scipy.sparse as sp
import torch

from ..interfaces.types import TorchDeviceStr
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

from .torch_vector_store import TorchVectorStore
from .torch_graph_store import TorchGraphStore

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
    Thin wrapper around TorchGraphStore for STaRK datasets.

    Adds STaRK-specific features:
    - CSR caching to disk
    - Dataset-specific average degree constants
    - Centrality heuristic registration

    Usage:
        adapter = StarkGraphAdapter(skb, "prime", adjacency_dict=adj)
        neighbors, mask = adapter.get_neighbors_batch(torch.tensor([1, 2, 3]))
    """

    def __init__(
        self,
        skb_data: 'SKB',
        dataset: str,
        adjacency_dict: Optional[Dict[int, List[int]]] = None,
        cache_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize graph adapter.

        Args:
            skb_data: STaRK SKB object
            dataset: Dataset name (e.g., "prime", "amazon", "mag")
            adjacency_dict: Pre-computed adjacency dictionary
            cache_path: Path to cache CSR matrix
            device: Target device ("cuda" or "cpu"), auto-detected if None
        """
        self.skb = skb_data
        if dataset not in AVG_LOG_DEGREE_BY_DATASET:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.dataset = dataset
        self.cache_path = cache_path or f"./adjacency_cache/graph_{dataset}.npz"
        self.avg_log_degree = AVG_LOG_DEGREE_BY_DATASET[dataset]
        self._device = device or get_device()

        # Load or create CSR matrix, then create TorchGraphStore
        csr_matrix = self._load_or_create_csr(adjacency_dict)
        self._store = TorchGraphStore.from_csr(
            csr_matrix,
            device=self._device,
            avg_degree=AVG_DEGREE_BY_DATASET.get(dataset, 10.0)
        )
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

    # Delegate all GraphStore methods to underlying TorchGraphStore
    def get_neighbors(self, node_id: int) -> torch.Tensor:
        return self._store.get_neighbors(node_id)

    def get_neighbors_batch(self, node_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._store.get_neighbors_batch(node_ids)

    def get_degrees_batch(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self._store.get_degrees_batch(node_ids)

    def get_degree(self, node_id: int) -> int:
        return self._store.get_degree(node_id)

    def contains(self, node_id: int) -> bool:
        return self._store.contains(node_id)

    def get_avg_degree(self) -> float:
        return AVG_DEGREE_BY_DATASET.get(self.dataset, 10.0)

    @property
    def n_nodes(self) -> int:
        return self._store.n_nodes

    @property
    def is_gpu(self) -> bool:
        return self._store.is_gpu

    @property
    def device(self) -> str:
        return self._device

    def close(self):
        self._store.close()

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
    Thin wrapper around TorchVectorStore for STaRK datasets.

    Usage:
        # Preferred: tensor-based (memory-efficient)
        store = StarkVectorStore(doc_tensor, doc_ids, device="cuda")

        # Backward compatible: dict-based
        store = StarkVectorStore.from_dict(doc_embs, device="cuda")
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        ids: torch.Tensor,
        device: Optional[TorchDeviceStr] = None,
        dense: bool = False
    ):
        """
        Initialize vector store from pre-stacked embeddings.

        Args:
            embeddings: Pre-stacked tensor of shape (N, D)
            ids: Tensor of document IDs of shape (N,)
            device: Target device ("cuda" or "cpu"), auto-detected if None
            dense: If True, use dense O(1) lookup (trades memory for speed)
        """
        self._device = device or get_device()
        self._store = TorchVectorStore.from_tensor(embeddings, ids, device=self._device, dense=dense)

    @classmethod
    def from_dict(
        cls,
        doc_embs: Dict[int, torch.Tensor],
        device: Optional[TorchDeviceStr] = None,
        dense: bool = False
    ) -> "StarkVectorStore":
        """
        Create from dict (backward compatibility). Prefer using tensor constructor directly.

        Args:
            doc_embs: Dictionary mapping doc_id -> embedding tensor
            device: Target device ("cuda" or "cpu"), auto-detected if None
            dense: If True, use dense O(1) lookup (trades memory for speed)

        Returns:
            StarkVectorStore instance
        """
        # Stack embeddings (same logic as TorchVectorStore.from_dict)
        sorted_ids = sorted(doc_embs.keys())
        ids = torch.tensor(sorted_ids, dtype=torch.long)
        emb_list = []
        for i in sorted_ids:
            vec = doc_embs[i]
            if isinstance(vec, torch.Tensor):
                vec = vec.detach().cpu()
            else:
                vec = torch.as_tensor(vec)
            if vec.dim() > 1:
                vec = vec.squeeze()
            emb_list.append(vec)
        embeddings = torch.stack(emb_list)
        return cls(embeddings=embeddings, ids=ids, device=device, dense=dense)

    # Delegate all VectorStore methods to underlying TorchVectorStore
    def search(self, query_vec: torch.Tensor, limit: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._store.search(query_vec, limit)

    def search_batch(self, query_vecs: torch.Tensor, limit: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._store.search_batch(query_vecs, limit)

    def fetch(self, node_id: int) -> Optional[torch.Tensor]:
        return self._store.fetch(node_id)

    def fetch_batch(self, node_ids: Union[List[int], torch.Tensor]) -> Tuple[Matrix, torch.Tensor]:
        return self._store.fetch_batch(node_ids)

    def compute_similarities(
        self,
        query_vec: torch.Tensor,
        candidate_ids: Union[List[int], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._store.compute_similarities(query_vec, candidate_ids)

    def compute_neighbor_similarities(
        self,
        query_vec: torch.Tensor,
        neighbor_ids: torch.Tensor,
        neighbor_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Fused neighbor similarity computation - returns None if not supported."""
        if hasattr(self._store, 'compute_neighbor_similarities'):
            return self._store.compute_neighbor_similarities(query_vec, neighbor_ids, neighbor_mask)
        return None

    @property
    def n_docs(self) -> int:
        return self._store.n_docs

    @property
    def dim(self) -> int:
        return self._store.dim

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_gpu(self) -> bool:
        return self._store.is_gpu

    def close(self):
        self._store.close()


class StarkPreComputedEmbeddingHandler(EmbeddingProvider):
    """
    Pre-computed embedding lookup with lazy GPU transfer.

    Embeddings stay on CPU until first access, then cached on target device.
    """

    def __init__(
        self,
        query_embs: Union[Dict[int, torch.Tensor], torch.Tensor],
        query_ids: Optional[torch.Tensor] = None,
        device: Optional[TorchDeviceStr] = None
    ):
        """
        Initialize from embedding dict or pre-stacked tensor.

        Args:
            query_embs: Either dict mapping query_id -> embedding, or pre-stacked tensor (N, D)
            query_ids: Required if query_embs is a tensor - maps index to query_id
            device: Target device for storage (auto-detected if None)
        """
        self._device = device if device is not None else get_device()
        self._gpu_cache: Dict[int, torch.Tensor] = {}

        if isinstance(query_embs, torch.Tensor):
            # Pre-stacked tensor mode
            if query_ids is None:
                raise ValueError("query_ids required when query_embs is a tensor")
            self._source_embs = query_embs  # Keep on CPU
            self._id_to_idx = {int(qid): idx for idx, qid in enumerate(query_ids.tolist())}
            self._is_tensor_mode = True
        else:
            # Dict mode (backward compatibility)
            self._source_embs = query_embs  # Keep dict on CPU
            self._id_to_idx = None
            self._is_tensor_mode = False

    def embed_query(self, query_id: int) -> torch.Tensor:
        """Get embedding for a query, transferring to GPU on first access."""
        # Check cache first
        if query_id in self._gpu_cache:
            return self._gpu_cache[query_id]

        # Load from source and transfer to GPU
        if self._is_tensor_mode:
            idx = self._id_to_idx[query_id]
            emb = self._source_embs[idx].detach().squeeze().to(self._device)
        else:
            emb = self._source_embs[query_id]
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().squeeze().to(self._device)
            else:
                emb = torch.as_tensor(emb, device=self._device).squeeze()

        self._gpu_cache[query_id] = emb
        return emb

    def embed_query_batch(self, query_ids: list[int]) -> Matrix:
        """Returns a 2D matrix (N_queries, Dimension) of embeddings."""
        return torch.stack([self.embed_query(qid) for qid in query_ids])

    @property
    def device(self) -> str:
        """Return the device embeddings are stored on."""
        return self._device
