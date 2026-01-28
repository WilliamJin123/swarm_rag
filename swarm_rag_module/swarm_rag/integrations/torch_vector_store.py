"""
PyTorch-Native Vector Store

Pure PyTorch operations for vector similarity search.
Works on both CPU and GPU - device is configurable.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Sequence, Any

import torch

from ..utils.device import get_device, ensure_tensor

from ..interfaces.shared_types import TorchDeviceStr
from ..interfaces.abstract_classes import VectorStore, Matrix

import logging
logger = logging.getLogger(__name__)


@dataclass
class TensorSearchResult:
    """
    Search result that stays on device until explicitly converted.
    """
    ids: torch.Tensor      # Shape: (k,), dtype: long
    scores: torch.Tensor   # Shape: (k,), dtype: float32

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Convert to list of dicts for interface compatibility."""
        ids_cpu = self.ids.cpu().tolist()
        scores_cpu = self.scores.cpu().tolist()
        return [
            {'id': int(i), 'score': float(s)}
            for i, s in zip(ids_cpu, scores_cpu)
        ]

    def to_device(self, device: Optional[TorchDeviceStr]) -> "TensorSearchResult":
        """Move result to a different device."""
        return TensorSearchResult(
            ids=self.ids.to(device),
            scores=self.scores.to(device)
        )

    def __len__(self) -> int:
        return self.ids.shape[0]


class TorchVectorStore(VectorStore):
    """
    PyTorch-native vector store with zero CPU-GPU transfers during search.

    Key features:
    - All embeddings stored as normalized tensors on target device
    - Cosine similarity via matrix multiplication
    - Works on both CPU and GPU
    - Compatible with VectorStore interface

    Usage:
        store = TorchVectorStore.from_dict(doc_embs, device="cuda")
        store = TorchVectorStore.from_dict(doc_embs, device="cpu")
        results = store.search(query_vec, limit=10)
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        ids: Union[torch.Tensor, List[int]],
        device: Optional[TorchDeviceStr] = None,
        normalize: bool = True
    ):
        """
        Initialize vector store with pre-loaded embeddings.

        Args:
            embeddings: Tensor of shape (N, D) containing document embeddings
            ids: Tensor/list of document IDs corresponding to each row
            device: Target device ("cuda" or "cpu"), auto-detected if None
            normalize: Whether to L2-normalize embeddings for cosine similarity
        """
        self._device = device or get_device()
        self._dtype = torch.float32

        # Store embeddings on device
        if isinstance(embeddings, torch.Tensor):
            self._embeddings = embeddings.to(device=self._device, dtype=self._dtype)
        else:
            self._embeddings = torch.tensor(
                embeddings, device=self._device, dtype=self._dtype
            )

        # Normalize for cosine similarity
        if normalize:
            self._embeddings = torch.nn.functional.normalize(
                self._embeddings, p=2, dim=1
            )

        # Store IDs as tensor
        if isinstance(ids, torch.Tensor):
            self._ids = ids.to(device='cpu', dtype=torch.long)
        else:
            self._ids = torch.tensor(list(ids), dtype=torch.long)
        self._id_to_idx = {int(real_id): i for i, real_id in enumerate(self._ids.tolist())}

        self.n_docs = len(self._ids)
        self.dim = self._embeddings.shape[1]

        logger.info(
            f"TorchVectorStore initialized: {self.n_docs} docs, dim={self.dim}, "
            f"device={self._device}"
        )

    @classmethod
    def from_dict(
        cls,
        doc_embs: Dict[int, Union[list, torch.Tensor]],
        device: Optional[TorchDeviceStr] = None
    ) -> "TorchVectorStore":
        """
        Create TorchVectorStore from a dictionary of embeddings.

        Args:
            doc_embs: Dictionary mapping doc_id -> embedding vector
            device: Target device (caller should provide; auto-detected if None)

        Returns:
            TorchVectorStore instance
        """
        if not doc_embs:
            raise ValueError("Cannot create store from empty dictionary")

        # Device should be provided by caller; fallback to auto-detect for compatibility
        target_device = device if device is not None else get_device()

        # Sort keys for deterministic ordering
        sorted_ids = sorted(doc_embs.keys())
        ids = torch.tensor(sorted_ids, dtype=torch.long, device=target_device)

        # Stack embeddings
        first_emb = doc_embs[sorted_ids[0]]
        if isinstance(first_emb, torch.Tensor):
            embeddings = torch.stack([
                (doc_embs[i].detach().cpu() if doc_embs[i].is_cuda else doc_embs[i].detach()).squeeze()
                for i in sorted_ids
            ])
        else:
            embeddings = torch.stack([
                torch.as_tensor(doc_embs[i]).squeeze() for i in sorted_ids
            ])

        # Move stacked embeddings to target device
        embeddings = embeddings.to(device=target_device)

        # Ensure 2D: (n_docs, dim)
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        elif embeddings.dim() > 2:
            embeddings = embeddings.squeeze()

        return cls(embeddings=embeddings, ids=ids, device=device)

    def search(
        self,
        query_vec: torch.Tensor,
        limit: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find top-k most similar documents to query vector.

        Args:
            query_vec: Query embedding tensor
            limit: Maximum number of results to return

        Returns:
            Tuple of (ids_tensor, scores_tensor) on device, sorted by score descending
        """
        query = query_vec.to(device=self._device, dtype=self._dtype).view(-1)
        query = torch.nn.functional.normalize(query.unsqueeze(0), p=2, dim=1)

        similarities = torch.mm(query, self._embeddings.t()).squeeze(0)

        k = min(limit, self.n_docs)
        scores, indices = torch.topk(similarities, k=k, largest=True)

        # Ensure _ids_tensor exists on device for tensor indexing
        if not hasattr(self, '_ids_tensor') or self._ids_tensor is None:
            self._ids_tensor = self._ids.to(device=self._device, dtype=torch.long)

        real_ids = self._ids_tensor[indices]
        return real_ids, scores

    def search_batch(
        self,
        query_vecs: torch.Tensor,
        limit: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch search for multiple queries simultaneously.

        Args:
            query_vecs: Query vectors tensor of shape (n_queries, dim)
            limit: Maximum results per query

        Returns:
            Tuple of (ids_tensor, scores_tensor) with shape (n_queries, limit)
        """
        queries = query_vecs.to(device=self._device, dtype=self._dtype)
        queries = torch.nn.functional.normalize(queries, p=2, dim=1)
        similarities = torch.mm(queries, self._embeddings.t())

        k = min(limit, self.n_docs)
        scores, indices = torch.topk(similarities, k=k, dim=1, largest=True)

        # Ensure _ids_tensor exists on device for tensor indexing
        if not hasattr(self, '_ids_tensor') or self._ids_tensor is None:
            self._ids_tensor = self._ids.to(device=self._device, dtype=torch.long)

        # Map internal indices to real IDs using tensor indexing
        real_ids = self._ids_tensor[indices]
        return real_ids, scores

    def fetch(self, node_id: int) -> Optional[torch.Tensor]:
        """Fetch embedding vector for a single document. Returns tensor on device."""
        idx = self._id_to_idx.get(node_id)
        if idx is None:
            logger.warning(f"Document ID {node_id} not found in store")
            return None
        return self._embeddings[idx]

    def fetch_batch(self, node_ids: Union[List[int], torch.Tensor]) -> Tuple[Matrix, torch.Tensor]:
        """
        Fetch embeddings for multiple documents.

        Args:
            node_ids: List or tensor of node IDs

        Returns:
            Tuple of:
                - embeddings: Tensor of shape (N, D) on device. NaN for invalid indices.
                - valid_mask: Boolean tensor of shape (N,) indicating which entries are valid.
        """
        if isinstance(node_ids, torch.Tensor):
            ids_tensor = node_ids.to(device=self._device, dtype=torch.long)
        else:
            ids_tensor = torch.tensor(node_ids, device=self._device, dtype=torch.long)

        n = ids_tensor.shape[0]
        result = torch.full((n, self.dim), float('nan'), dtype=self._dtype, device=self._device)
        valid_mask = torch.zeros(n, dtype=torch.bool, device=self._device)

        if n == 0:
            return result, valid_mask

        # Build reverse mapping tensor for device-native lookup (lazy init)
        if not hasattr(self, '_id_lookup_tensor') or self._id_lookup_tensor is None:
            max_id = int(self._ids.max().item()) + 1
            self._id_lookup_tensor = torch.full((max_id,), -1, device=self._device, dtype=torch.long)
            id_indices = torch.arange(len(self._ids), device=self._device)
            ids_on_device = self._ids.to(device=self._device, dtype=torch.long)
            self._id_lookup_tensor[ids_on_device] = id_indices
            self._max_valid_id = max_id

        # Find which IDs are in valid range
        in_range_mask = (ids_tensor >= 0) & (ids_tensor < self._max_valid_id)
        if not in_range_mask.any():
            return result, valid_mask

        # Lookup internal indices for in-range IDs
        clamped_ids = torch.clamp(ids_tensor, 0, self._max_valid_id - 1)
        internal_indices = self._id_lookup_tensor[clamped_ids]

        # Valid if in range AND found in store
        valid_mask = in_range_mask & (internal_indices >= 0)

        if valid_mask.any():
            valid_internal_indices = internal_indices[valid_mask]
            result[valid_mask] = self._embeddings[valid_internal_indices]

        return result, valid_mask

    def search_tensor_result(
        self,
        query_vec: torch.Tensor,
        limit: int
    ) -> TensorSearchResult:
        """Search returning TensorSearchResult."""
        ids, scores = self.search(query_vec, limit)
        return TensorSearchResult(ids=ids, scores=scores)

    def compute_similarities(
        self,
        query_vec: torch.Tensor,
        candidate_ids: Union[List[int], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute similarities between query and specific candidates."""
        if isinstance(candidate_ids, torch.Tensor):
            ids_tensor = candidate_ids.to(device=self._device, dtype=torch.long)
        else:
            ids_tensor = torch.as_tensor(candidate_ids, device=self._device, dtype=torch.long)

        candidate_embs, valid_mask = self.fetch_batch(ids_tensor)

        if not valid_mask.any():
            return torch.empty(0, device=self._device), torch.empty(0, device=self._device, dtype=torch.long)

        # Extract only valid embeddings and IDs
        valid_embs = candidate_embs[valid_mask]
        valid_ids = ids_tensor[valid_mask]

        query = query_vec.to(device=self._device, dtype=self._dtype).view(1, -1)
        query = torch.nn.functional.normalize(query, p=2, dim=1)
        similarities = torch.mm(query, valid_embs.t()).squeeze(0)

        return similarities, valid_ids

    @property
    def device(self) -> str:
        """Return the device this store is on."""
        return self._device

    @property
    def is_gpu(self) -> bool:
        """Check if using GPU."""
        return self._device == "cuda"

    @property
    def embeddings(self) -> torch.Tensor:
        """Return raw embeddings tensor."""
        return self._embeddings

    def to(self, device: Optional[TorchDeviceStr]) -> "TorchVectorStore":
        """Move store to a different device. Deprecated: set device at construction."""
        logger.warning("TorchVectorStore.to() is deprecated - set device at construction time")
        self._embeddings = self._embeddings.to(device)
        self._device = device
        # Clear cached lookup tensors
        if hasattr(self, '_ids_tensor'):
            self._ids_tensor = None
        if hasattr(self, '_id_lookup_tensor'):
            self._id_lookup_tensor = None
        return self

    def close(self):
        """Release memory held by this store."""
        if hasattr(self, '_embeddings') and self._embeddings is not None:
            del self._embeddings
            self._embeddings = None

        if hasattr(self, '_id_lookup_tensor') and self._id_lookup_tensor is not None:
            del self._id_lookup_tensor
            self._id_lookup_tensor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("TorchVectorStore resources released")

    def __del__(self):
        """Destructor to ensure memory is released."""
        try:
            self.close()
        except Exception:
            pass


__all__ = ['TorchVectorStore', 'TensorSearchResult']
