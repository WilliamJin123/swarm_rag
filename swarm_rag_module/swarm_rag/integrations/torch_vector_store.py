
"""
PyTorch-Native Vector Store

Pure PyTorch operations for vector similarity search.
Works on both CPU and GPU - device is configurable.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Sequence, Any

import torch

from ..utils.device import get_device
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
    - IDs must be provided as a torch.Tensor
    - Efficient lookups via sorting and searchsorted (O(log N))
    - Works on both CPU and GPU
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        ids: torch.Tensor,
        device: Optional[TorchDeviceStr] = None,
        normalize: bool = True,
        dense: bool = False
    ):
        """
        Initialize vector store with pre-loaded embeddings.

        Args:
            embeddings: Tensor of shape (N, D) containing document embeddings
            ids: Tensor of shape (N,) containing document IDs. Must be a torch.Tensor.
            device: Target device ("cuda" or "cpu"), auto-detected if None
            normalize: Whether to L2-normalize embeddings for cosine similarity
            dense: If True, use dense O(1) indexing (trades memory for speed)
        """
        if not isinstance(ids, torch.Tensor):
            raise TypeError(f"ids must be a torch.Tensor, got {type(ids)}")

        self._device = device if device is not None else get_device()
        self._dtype = torch.float32
        self._dense = dense

        # Ensure embeddings are 2D float32
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        embeddings = embeddings.to(dtype=self._dtype)

        # Ensure IDs are 1D long
        if ids.dim() != 1:
            raise ValueError(f"ids must be 1-dimensional, got shape {ids.shape}")
        ids = ids.to(dtype=torch.long)

        if len(embeddings) != len(ids):
            raise ValueError(
                f"Mismatch in length: embeddings ({len(embeddings)}) vs ids ({len(ids)})"
            )

        self.n_docs = len(ids)
        self.dim = embeddings.shape[1]

        # Normalize for cosine similarity
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        if dense:
            # Dense mode: embeddings[node_id] gives embedding directly (O(1) lookup)
            max_id = int(ids.max().item()) + 1
            self._dense_embeddings = torch.zeros(
                (max_id, self.dim), dtype=self._dtype, device=self._device
            )
            self._valid_mask = torch.zeros(max_id, dtype=torch.bool, device=self._device)

            # Scatter embeddings into dense tensor
            ids_device = ids.to(device=self._device)
            self._dense_embeddings[ids_device] = embeddings.to(device=self._device)
            self._valid_mask[ids_device] = True

            # For search(), we still need the sparse representation
            sorted_indices = torch.argsort(ids)
            self._ids = ids[sorted_indices].to(device=self._device)
            self._embeddings = embeddings[sorted_indices].to(device=self._device)

            dense_mb = max_id * self.dim * 4 / 1024 / 1024
            logger.info(
                f"TorchVectorStore (DENSE): {self.n_docs} docs, dim={self.dim}, "
                f"max_id={max_id}, memory={dense_mb:.0f}MB, device={self._device}"
            )
        else:
            # Sparse mode: searchsorted lookup (O(log N))
            sorted_indices = torch.argsort(ids)
            self._ids = ids[sorted_indices].to(device=self._device)
            self._embeddings = embeddings[sorted_indices].to(device=self._device)
            self._dense_embeddings = None
            self._valid_mask = None

            logger.debug(
                f"TorchVectorStore (sparse): {self.n_docs} docs, dim={self.dim}, "
                f"device={self._device}"
            )

    @classmethod
    def from_dict(
        cls,
        doc_embs: Dict[int, Union[list, torch.Tensor]],
        device: Optional[TorchDeviceStr] = None,
        dense: bool = False
    ) -> "TorchVectorStore":
        """
        Create TorchVectorStore from a dictionary of embeddings.

        Args:
            doc_embs: Dictionary mapping doc_id -> embedding vector
            device: Target device
            dense: If True, use dense storage for O(1) lookup (trades memory for speed)

        Returns:
            TorchVectorStore instance
        """
        if not doc_embs:
            raise ValueError("Cannot create store from empty dictionary")

        target_device = device if device is not None else get_device()

        # Sort keys for deterministic ordering
        sorted_ids = sorted(doc_embs.keys())

        # Convert IDs to tensor immediately as per strict requirements
        ids = torch.tensor(sorted_ids, dtype=torch.long)

        # Stack embeddings safely
        # We move to CPU temporarily to stack if they are on mixed devices,
        # then move to target device afterwards.
        emb_list = []
        for i in sorted_ids:
            vec = doc_embs[i]
            if isinstance(vec, torch.Tensor):
                # Detach and move to CPU to ensure stacking works regardless of source device
                vec = vec.detach().cpu()
            else:
                vec = torch.as_tensor(vec)

            # Ensure 1D vector
            if vec.dim() > 1:
                vec = vec.squeeze()
            emb_list.append(vec)

        embeddings = torch.stack(emb_list)

        return cls(embeddings=embeddings, ids=ids, device=device, dense=dense)

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
        query = query_vec.to(device=self._device, dtype=self._dtype)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        query = torch.nn.functional.normalize(query, p=2, dim=1)

        similarities = torch.mm(query, self._embeddings.t()).squeeze(0)

        k = min(limit, self.n_docs)
        scores, indices = torch.topk(similarities, k=k, largest=True)

        # Map internal sorted indices back to real IDs
        real_ids = self._ids[indices]
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
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)
            
        queries = torch.nn.functional.normalize(queries, p=2, dim=1)
        similarities = torch.mm(queries, self._embeddings.t())

        k = min(limit, self.n_docs)
        scores, indices = torch.topk(similarities, k=k, dim=1, largest=True)

        # Map internal indices to real IDs
        real_ids = self._ids[indices]
        return real_ids, scores

    def fetch(self, node_id: int) -> Optional[torch.Tensor]:
        """
        Fetch embedding vector for a single document.
        Returns tensor on device or None if not found.
        """
        # Create a tensor for the ID
        id_tensor = torch.tensor([node_id], device=self._device, dtype=torch.long)
        
        # Searchsorted logic
        idx = torch.searchsorted(self._ids, id_tensor)
        
        # Clamp index to prevent out of bounds if node_id > max_id
        idx = torch.clamp(idx, 0, self.n_docs - 1)
        
        found_id = self._ids[idx]
        
        if found_id == node_id:
            return self._embeddings[idx]
        return None

    def fetch_batch(self, node_ids: Union[List[int], torch.Tensor]) -> Tuple[Matrix, torch.Tensor]:
        """
        Fetch embeddings for multiple documents.

        Uses O(1) direct indexing if dense mode is enabled,
        otherwise falls back to O(log N) binary search.

        Args:
            node_ids: List or tensor of node IDs

        Returns:
            Tuple of:
                - embeddings: Tensor of shape (N, D) on device. NaN for invalid indices.
                - valid_mask: Boolean tensor of shape (N,) indicating which entries are valid.
        """
        if isinstance(node_ids, list):
            if not node_ids:
                return torch.empty((0, self.dim), device=self._device), torch.zeros(0, dtype=torch.bool, device=self._device)
            node_ids = torch.tensor(node_ids, device=self._device, dtype=torch.long)
        else:
            node_ids = node_ids.to(device=self._device, dtype=torch.long)

        n = len(node_ids)
        if n == 0:
            return torch.empty((0, self.dim), device=self._device), torch.zeros(0, dtype=torch.bool, device=self._device)

        # Dense mode: O(1) direct indexing
        if self._dense and self._dense_embeddings is not None:
            max_id = self._dense_embeddings.shape[0]
            # Clamp IDs to valid range
            clamped_ids = node_ids.clamp(0, max_id - 1)
            # Direct index lookup
            result = self._dense_embeddings[clamped_ids]
            # Check validity: in range AND has embedding
            in_range = (node_ids >= 0) & (node_ids < max_id)
            valid_mask = in_range & self._valid_mask[clamped_ids]
            # Set invalid entries to NaN
            result = torch.where(valid_mask.unsqueeze(1), result, torch.full_like(result, float('nan')))
            return result, valid_mask

        # Sparse mode: O(log N) binary search
        result = torch.full((n, self.dim), float('nan'), dtype=self._dtype, device=self._device)
        valid_mask = torch.zeros(n, dtype=torch.bool, device=self._device)

        # Find insertion points
        indices = torch.searchsorted(self._ids, node_ids)

        # Clamp indices to be safe for indexing (handle IDs > max_id)
        clamped_indices = torch.clamp(indices, 0, self.n_docs - 1)

        # Check if the ID at the found index actually matches the query ID
        found_ids = self._ids[clamped_indices]
        valid_mask = (found_ids == node_ids)

        if valid_mask.any():
            result[valid_mask] = self._embeddings[clamped_indices[valid_mask]]

        return result, valid_mask

    def compute_neighbor_similarities(
        self,
        query_vec: torch.Tensor,
        neighbor_ids: torch.Tensor,
        neighbor_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarities for all neighbors in one fused operation.

        FAST PATH: Skips unique→fetch→scatter pipeline by directly indexing
        dense embeddings and computing similarities in-place.

        Args:
            query_vec: Query embedding (D,) or (1, D)
            neighbor_ids: Neighbor IDs tensor (n_agents, max_degree)
            neighbor_mask: Valid neighbor mask (n_agents, max_degree)

        Returns:
            Similarities tensor (n_agents, max_degree), -inf for invalid neighbors
        """
        if not self._dense or self._dense_embeddings is None:
            # Fall back to None to signal caller should use old path
            return None

        n_agents, max_degree = neighbor_ids.shape
        device = self._device

        # Initialize similarities to -inf (invalid)
        similarities = torch.full(
            (n_agents, max_degree), -float('inf'),
            device=device, dtype=torch.float32
        )

        if not neighbor_mask.any():
            return similarities

        # Prepare query vector
        query = query_vec.to(device=device, dtype=torch.float32)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        query = torch.nn.functional.normalize(query, p=2, dim=1)  # (1, D)

        # Get valid neighbor IDs
        valid_ids = neighbor_ids[neighbor_mask]  # (N_valid,)

        # Clamp to valid embedding range
        max_id = self._dense_embeddings.shape[0]
        clamped_ids = valid_ids.clamp(0, max_id - 1)

        # Direct dense lookup - O(1) per ID, no unique needed!
        valid_embs = self._dense_embeddings[clamped_ids]  # (N_valid, D)

        # Check which embeddings are actually valid
        in_range = (valid_ids >= 0) & (valid_ids < max_id)
        has_emb = self._valid_mask[clamped_ids]
        emb_valid = in_range & has_emb

        # Compute similarities for valid embeddings
        # (N_valid, D) @ (D, 1) -> (N_valid, 1) -> (N_valid,)
        valid_sims = torch.mm(valid_embs, query.t()).squeeze(1)

        # Zero out invalid embeddings
        valid_sims = torch.where(emb_valid, valid_sims, torch.full_like(valid_sims, -float('inf')))

        # Scatter back to 2D tensor
        similarities[neighbor_mask] = valid_sims

        return similarities

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
        """
        Compute similarities between query and specific candidates.
        Returns only valid (id, score) pairs.
        """
        candidate_embs, valid_mask = self.fetch_batch(candidate_ids)

        if not valid_mask.any():
            return torch.empty(0, device=self._device), torch.empty(0, device=self._device, dtype=torch.long)

        # Extract only valid embeddings and IDs
        valid_embs = candidate_embs[valid_mask]
        
        # Retrieve IDs tensor corresponding to the valid mask
        if isinstance(candidate_ids, torch.Tensor):
            ids_tensor = candidate_ids.to(self._device)
        else:
            ids_tensor = torch.tensor(candidate_ids, device=self._device)
            
        valid_ids = ids_tensor[valid_mask]

        query = query_vec.to(device=self._device, dtype=self._dtype)
        if query.dim() == 1:
            query = query.unsqueeze(0)
            
        query = torch.nn.functional.normalize(query, p=2, dim=1)
        
        similarities = torch.mm(query, valid_embs.t()).squeeze(0)

        return similarities, valid_ids

    @property
    def device(self) -> str:
        """Return the device this store is on."""
        return self._device

    @property
    def is_gpu(self) -> bool:
        """Check if using GPU (cuda or mps)."""
        return self._device != "cpu"

    @property
    def embeddings(self) -> torch.Tensor:
        """Return raw embeddings tensor (sorted by ID)."""
        return self._embeddings

    def to(self, device: Optional[TorchDeviceStr]) -> "TorchVectorStore":
        """Move store to a different device."""
        logger.warning("TorchVectorStore.to() is deprecated - set device at construction time")
        self._embeddings = self._embeddings.to(device)
        self._ids = self._ids.to(device)
        self._device = device
        return self

    def close(self):
        """Release memory held by this store."""
        if hasattr(self, '_embeddings') and self._embeddings is not None:
            del self._embeddings
            self._embeddings = None
        if hasattr(self, '_ids') and self._ids is not None:
            del self._ids
            self._ids = None
        if hasattr(self, '_dense_embeddings') and self._dense_embeddings is not None:
            del self._dense_embeddings
            self._dense_embeddings = None
        if hasattr(self, '_valid_mask') and self._valid_mask is not None:
            del self._valid_mask
            self._valid_mask = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("TorchVectorStore resources released")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = ['TorchVectorStore', 'TensorSearchResult']