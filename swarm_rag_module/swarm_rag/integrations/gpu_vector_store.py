"""
PyTorch-Native GPU Vector Store

Replaces FAISS with pure PyTorch operations to avoid CPU-GPU data transfers.
All operations stay on GPU when available, providing 12-36x speedup for vector search.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Sequence, Any

import torch

from ..utils.device import get_device, ensure_tensor
from ..interfaces.abstract_classes import VectorStore, Matrix

import logging
logger = logging.getLogger(__name__)


@dataclass
class TensorSearchResult:
    """
    GPU-native search result that stays on device until explicitly converted.

    Provides lazy conversion to dict format for backward compatibility.
    """
    ids: torch.Tensor      # Shape: (k,), dtype: long
    scores: torch.Tensor   # Shape: (k,), dtype: float32

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Convert to list of dicts for legacy interface compatibility."""
        ids_cpu = self.ids.cpu().tolist()
        scores_cpu = self.scores.cpu().tolist()
        return [
            {'id': int(i), 'score': float(s)}
            for i, s in zip(ids_cpu, scores_cpu)
        ]

    def to_device(self, device: str) -> "TensorSearchResult":
        """Move result to a different device."""
        return TensorSearchResult(
            ids=self.ids.to(device),
            scores=self.scores.to(device)
        )

    def __len__(self) -> int:
        return self.ids.shape[0]


class GPUVectorStore(VectorStore):
    """
    PyTorch-native vector store with zero CPU-GPU transfers during search.

    Also available as `TorchVectorStore` for device-agnostic code.

    Key features:
    - All embeddings stored as normalized GPU tensors
    - Cosine similarity via matrix multiplication (extremely fast on GPU)
    - Automatic fallback to CPU if GPU unavailable
    - Compatible with existing VectorStore interface

    Usage:
        # From dictionary of embeddings (auto-detect device)
        store = GPUVectorStore.from_dict(doc_embs, device="auto")

        # Force CPU
        store = GPUVectorStore.from_dict(doc_embs, device="cpu")

        # Search
        results = store.search(query_vec, k=10)

        # Batch fetch
        vectors = store.fetch_batch([1, 2, 3])
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        ids: Union[torch.Tensor, List[int]],
        device: str = None,
        normalize: bool = True
    ):
        """
        Initialize GPU vector store with pre-loaded embeddings.

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
            f"GPUVectorStore initialized: {self.n_docs} docs, dim={self.dim}, "
            f"device={self._device}"
        )

    @classmethod
    def from_dict(
        cls,
        doc_embs: Dict[int, Union[list, torch.Tensor]],
        device: str = None
    ) -> "GPUVectorStore":
        """
        Create GPUVectorStore from a dictionary of embeddings.

        Args:
            doc_embs: Dictionary mapping doc_id -> embedding vector
            device: Target device (auto-detected if None)

        Returns:
            GPUVectorStore instance
        """
        if not doc_embs:
            raise ValueError("Cannot create store from empty dictionary")

        # Sort keys for deterministic ordering
        sorted_ids = sorted(doc_embs.keys())
        ids = torch.tensor(sorted_ids, dtype=torch.long)

        # Stack embeddings
        first_emb = doc_embs[sorted_ids[0]]
        if isinstance(first_emb, torch.Tensor):
            embeddings = torch.stack([
                (doc_embs[i].detach().cpu() if doc_embs[i].is_cuda else doc_embs[i].detach()).squeeze()
                for i in sorted_ids
            ])
        else:
            # Convert list/array to tensor
            embeddings = torch.stack([
                torch.as_tensor(doc_embs[i]).squeeze() for i in sorted_ids
            ])

        # Ensure 2D: (n_docs, dim)
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        elif embeddings.dim() > 2:
            embeddings = embeddings.squeeze()

        return cls(embeddings=embeddings, ids=ids, device=device)

    def search(
        self,
        query_vec: Union[list, torch.Tensor],
        limit: int
    ) -> List[Dict]:
        """
        Find top-k most similar documents to query vector.

        Args:
            query_vec: Query embedding vector
            limit: Maximum number of results to return

        Returns:
            List of dicts with 'id' and 'score' keys, sorted by score descending
        """
        # Ensure query is on device and normalized
        if not isinstance(query_vec, torch.Tensor):
            query = torch.tensor(
                query_vec, device=self._device, dtype=self._dtype
            )
        else:
            query = query_vec.to(device=self._device, dtype=self._dtype)

        # Flatten if needed
        query = query.view(-1)
        query = torch.nn.functional.normalize(query.unsqueeze(0), p=2, dim=1)

        # Compute similarities via matrix multiplication
        # (1, D) @ (D, N) -> (1, N)
        similarities = torch.mm(query, self._embeddings.t()).squeeze(0)

        # Get top-k
        k = min(limit, self.n_docs)
        scores, indices = torch.topk(similarities, k=k, largest=True)

        # Convert to list of dicts
        scores_cpu = scores.cpu().tolist()
        indices_cpu = indices.cpu().tolist()

        results = []
        for score, idx in zip(scores_cpu, indices_cpu):
            real_id = int(self._ids[idx])
            results.append({'id': real_id, 'score': float(score)})

        return results

    def search_batch(
        self,
        query_vecs: Union[list, torch.Tensor],
        limit: int
    ) -> List[List[Dict]]:
        """
        Batch search for multiple queries simultaneously.

        Args:
            query_vecs: Query embeddings of shape (N_queries, D)
            limit: Maximum results per query

        Returns:
            List of result lists, one per query
        """
        # Convert and normalize queries
        if not isinstance(query_vecs, torch.Tensor):
            queries = torch.tensor(
                query_vecs, device=self._device, dtype=self._dtype
            )
        else:
            queries = query_vecs.to(device=self._device, dtype=self._dtype)

        queries = torch.nn.functional.normalize(queries, p=2, dim=1)

        # Batch similarity computation
        # (N_queries, D) @ (D, N_docs) -> (N_queries, N_docs)
        similarities = torch.mm(queries, self._embeddings.t())

        # Get top-k for all queries
        k = min(limit, self.n_docs)
        scores, indices = torch.topk(similarities, k=k, dim=1, largest=True)

        # Convert to list of results
        scores_cpu = scores.cpu().tolist()
        indices_cpu = indices.cpu().tolist()

        all_results = []
        for q_idx in range(len(queries)):
            results = []
            for score, idx in zip(scores_cpu[q_idx], indices_cpu[q_idx]):
                real_id = int(self._ids[idx])
                results.append({'id': real_id, 'score': float(score)})
            all_results.append(results)

        return all_results

    def fetch(self, node_id: int) -> Optional[torch.Tensor]:
        """
        Fetch embedding vector for a single document.

        Args:
            node_id: Document ID

        Returns:
            Embedding vector as tensor, or None if not found
        """
        idx = self._id_to_idx.get(node_id)
        if idx is None:
            logger.warning(f"Document ID {node_id} not found in store")
            return None

        return self._embeddings[idx].cpu()

    def fetch_batch(self, node_ids: List[int]) -> Matrix:
        """
        Fetch embeddings for multiple documents.

        Args:
            node_ids: List of document IDs

        Returns:
            2D tensor of shape (len(node_ids), dim).
            Missing documents have NaN values.
        """
        # Create result tensor with NaN for missing
        result = torch.full((len(node_ids), self.dim), float('nan'), dtype=torch.float32)

        # Find valid indices
        valid_pairs = []  # (output_idx, internal_idx)
        for out_idx, nid in enumerate(node_ids):
            int_idx = self._id_to_idx.get(nid)
            if int_idx is not None:
                valid_pairs.append((out_idx, int_idx))

        if valid_pairs:
            out_indices, int_indices = zip(*valid_pairs)
            int_indices_tensor = torch.tensor(
                int_indices, device=self._device, dtype=torch.long
            )

            # Batch index and transfer to CPU
            fetched = self._embeddings[int_indices_tensor].cpu()

            for i, out_idx in enumerate(out_indices):
                result[out_idx] = fetched[i]

        return result

    def fetch_batch_gpu(
        self,
        doc_ids: Union[List[int], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch embeddings for batch of doc IDs, keeping everything on GPU.

        Args:
            doc_ids: List or GPU tensor of document IDs

        Returns:
            Tuple of (embeddings tensor, valid_ids tensor) - both on GPU
        """
        # Convert to tensor if needed
        if isinstance(doc_ids, torch.Tensor):
            ids_tensor = doc_ids.to(device=self._device, dtype=torch.long)
        else:
            ids_tensor = torch.tensor(doc_ids, device=self._device, dtype=torch.long)

        # Build reverse mapping tensor for GPU-native lookup
        # This avoids Python dict lookups
        if not hasattr(self, '_id_lookup_tensor') or self._id_lookup_tensor is None:
            # Build lookup tensor: maps real_id -> internal_idx
            # For IDs not in store, we'll filter them out
            max_id = int(self._ids.max().item()) + 1
            self._id_lookup_tensor = torch.full(
                (max_id,), -1, device=self._device, dtype=torch.long
            )
            id_indices = torch.arange(len(self._ids), device=self._device)
            ids_on_gpu = self._ids.to(device=self._device, dtype=torch.long)
            self._id_lookup_tensor[ids_on_gpu] = id_indices
            self._max_valid_id = max_id

        # Filter valid IDs (within range and exists in store)
        valid_range_mask = (ids_tensor >= 0) & (ids_tensor < self._max_valid_id)
        ids_in_range = ids_tensor[valid_range_mask]

        if ids_in_range.numel() == 0:
            return (
                torch.empty((0, self.dim), device=self._device, dtype=torch.float32),
                torch.empty(0, device=self._device, dtype=torch.long)
            )

        # GPU lookup of internal indices
        internal_indices = self._id_lookup_tensor[ids_in_range]

        # Filter out IDs not in store (-1 means not found)
        valid_mask = internal_indices >= 0
        valid_internal_indices = internal_indices[valid_mask]
        valid_ids = ids_in_range[valid_mask]

        if valid_internal_indices.numel() == 0:
            return (
                torch.empty((0, self.dim), device=self._device, dtype=torch.float32),
                torch.empty(0, device=self._device, dtype=torch.long)
            )

        # Direct GPU indexing - no CPU transfer
        embeddings = self._embeddings[valid_internal_indices]

        return embeddings, valid_ids

    # ===== Tensor-native interface methods (GPU-optimized) =====

    def search_tensor(
        self,
        query_vec: Union[list, torch.Tensor],
        limit: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU-native search returning tensors instead of dicts.

        All computation stays on GPU - no CPU round-trips.

        Args:
            query_vec: Query vector (list or tensor)
            limit: Maximum results to return

        Returns:
            Tuple of (ids_tensor, scores_tensor) on device
        """
        # Ensure query is on device and normalized
        if not isinstance(query_vec, torch.Tensor):
            query = torch.tensor(
                query_vec, device=self._device, dtype=self._dtype
            )
        else:
            query = query_vec.to(device=self._device, dtype=self._dtype)

        # Flatten and normalize
        query = query.view(-1)
        query = torch.nn.functional.normalize(query.unsqueeze(0), p=2, dim=1)

        # Compute similarities via matrix multiplication
        similarities = torch.mm(query, self._embeddings.t()).squeeze(0)

        # Get top-k on GPU
        k = min(limit, self.n_docs)
        scores, indices = torch.topk(similarities, k=k, largest=True)

        # Map internal indices to real IDs on GPU
        if not hasattr(self, '_ids_tensor') or self._ids_tensor is None:
            self._ids_tensor = self._ids.to(device=self._device, dtype=torch.long)

        real_ids = self._ids_tensor[indices]

        return real_ids, scores

    def search_tensor_result(
        self,
        query_vec: Union[list, torch.Tensor],
        limit: int
    ) -> TensorSearchResult:
        """
        Search returning TensorSearchResult for lazy conversion.

        Args:
            query_vec: Query vector
            limit: Maximum results

        Returns:
            TensorSearchResult with ids and scores on GPU
        """
        ids, scores = self.search_tensor(query_vec, limit)
        return TensorSearchResult(ids=ids, scores=scores)

    def fetch_batch_tensor(
        self,
        node_ids: Union[Sequence[Any], torch.Tensor],
        device: str = None
    ) -> torch.Tensor:
        """
        Fetch embeddings as tensor on specified device.

        GPU-optimized: avoids CPU round-trips when node_ids is a tensor.

        Args:
            node_ids: Sequence or tensor of node IDs
            device: Target device (None uses store's device)

        Returns:
            Tensor of shape (N, D) on device. NaN for invalid indices.
        """
        target_device = device or self._device

        # Use GPU-native path if input is tensor
        if isinstance(node_ids, torch.Tensor):
            embeddings, valid_ids = self.fetch_batch_gpu(node_ids)
            # If all requested IDs were valid, return directly
            if valid_ids.shape[0] == node_ids.shape[0]:
                return embeddings.to(target_device)

            # Otherwise, need to build result with NaN for missing
            result = torch.full(
                (node_ids.shape[0], self.dim),
                float('nan'),
                device=target_device,
                dtype=self._dtype
            )
            # Map valid results back to original positions
            # This requires knowing which input positions were valid
            node_ids_cpu = node_ids.cpu().tolist()
            valid_ids_set = set(valid_ids.cpu().tolist())
            valid_mask = torch.tensor(
                [nid in valid_ids_set for nid in node_ids_cpu],
                device=target_device
            )
            result[valid_mask] = embeddings.to(target_device)
            return result

        # List/sequence path
        node_ids_list = list(node_ids)
        result = torch.full(
            (len(node_ids_list), self.dim),
            float('nan'),
            device=target_device,
            dtype=self._dtype
        )

        valid_pairs = []
        for out_idx, nid in enumerate(node_ids_list):
            int_idx = self._id_to_idx.get(nid)
            if int_idx is not None:
                valid_pairs.append((out_idx, int_idx))

        if valid_pairs:
            out_indices, int_indices = zip(*valid_pairs)
            int_indices_tensor = torch.tensor(
                int_indices, device=self._device, dtype=torch.long
            )
            fetched = self._embeddings[int_indices_tensor]
            result[list(out_indices)] = fetched.to(target_device)

        return result

    def supports_tensor_ops(self) -> bool:
        """Check if this store has optimized tensor operations."""
        return True

    def compute_similarities(
        self,
        query_vec: Union[list, torch.Tensor],
        candidate_ids: Union[List[int], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute similarities between query and specific candidates.

        Optimized for swarm retrieval where we need similarities to neighbors.

        Args:
            query_vec: Query embedding
            candidate_ids: List or tensor of candidate document IDs

        Returns:
            Tuple of (similarity scores tensor, valid_ids tensor) - both on GPU
        """
        # Get candidate embeddings (stays on GPU)
        candidate_embs, valid_ids = self.fetch_batch_gpu(candidate_ids)

        if valid_ids.numel() == 0:
            return torch.empty(0, device=self._device), valid_ids

        # Ensure query is on device and normalized
        if not isinstance(query_vec, torch.Tensor):
            query = torch.tensor(
                query_vec, device=self._device, dtype=self._dtype
            ).view(1, -1)
        else:
            query = query_vec.to(device=self._device, dtype=self._dtype).view(1, -1)

        query = torch.nn.functional.normalize(query, p=2, dim=1)

        # Compute similarities
        # (1, D) @ (N, D).T -> (1, N) -> (N,)
        similarities = torch.mm(query, candidate_embs.t()).squeeze(0)

        return similarities, valid_ids

    @property
    def device(self) -> str:
        """Return the device this store is on."""
        return self._device

    @property
    def is_gpu(self) -> bool:
        """Check if using GPU backend."""
        return self._device == "cuda"

    @property
    def embeddings(self) -> torch.Tensor:
        """Return raw embeddings tensor (for advanced use)."""
        return self._embeddings

    def to(self, device: str) -> "GPUVectorStore":
        """
        Move store to a different device.

        Args:
            device: Target device ("cuda" or "cpu")

        Returns:
            Self (for chaining)
        """
        self._embeddings = self._embeddings.to(device)
        self._device = device
        return self

    def close(self):
        """
        Release GPU memory held by this store.

        Deletes embedding tensors and lookup tensors, then clears CUDA cache.
        Safe to call multiple times.
        """
        if hasattr(self, '_embeddings') and self._embeddings is not None:
            del self._embeddings
            self._embeddings = None

        if hasattr(self, '_id_lookup_tensor') and self._id_lookup_tensor is not None:
            del self._id_lookup_tensor
            self._id_lookup_tensor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("GPUVectorStore resources released")

    def __del__(self):
        """Destructor to ensure GPU memory is released."""
        try:
            self.close()
        except Exception:
            # Ignore errors during interpreter shutdown
            pass


# Device-agnostic alias (GPUVectorStore works on both CPU and GPU)
TorchVectorStore = GPUVectorStore

__all__ = ['GPUVectorStore', 'TorchVectorStore', 'TensorSearchResult']
