from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Sequence, TypeAlias, Union, Tuple
import torch

# Matrix type alias: PyTorch tensors only
Matrix: TypeAlias = torch.Tensor  # Shape (N, D)

class VectorStore(ABC):
    """Abstract contract for Vector Databases (LanceDB, Chroma, Faiss, In-Memory)"""

    @abstractmethod
    def search(self, query_vec: torch.Tensor, limit: int) -> Sequence[Dict[str, Any]]:
        """Returns sequence of dicts: [{'id': ..., 'score': ...}, ...]"""
        pass

    @abstractmethod
    def fetch_batch(self, node_ids: Sequence[Any]) -> Matrix:
        """Returns a 2D matrix of shape (N, D). NaN for invalid indices."""
        pass

    @abstractmethod
    def fetch(self, node_ids: Any) -> Optional[torch.Tensor]:
        """Returns single vector"""
        pass

    # ===== Tensor-native interface methods (optional, for GPU acceleration) =====

    def search_tensor(
        self,
        query_vec: torch.Tensor,
        limit: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU-native search returning tensors instead of dicts.

        Args:
            query_vec: Query vector (tensor)
            limit: Maximum results to return

        Returns:
            Tuple of (ids_tensor, scores_tensor) on device

        Default implementation converts from search() result.
        """
        results = self.search(query_vec, limit)
        ids = torch.tensor([r['id'] for r in results], dtype=torch.long)
        scores = torch.tensor([r['score'] for r in results], dtype=torch.float32)
        return ids, scores

    def fetch_batch_tensor(
        self,
        node_ids: Union[Sequence[Any], torch.Tensor],
        device: str = None
    ) -> torch.Tensor:
        """
        Fetch embeddings as tensor on specified device.

        Args:
            node_ids: Sequence or tensor of node IDs
            device: Target device (None for auto-detect)

        Returns:
            Tensor of shape (N, D) on device. NaN for invalid indices.

        Default implementation converts from fetch_batch() result.
        """
        from ..utils.device import get_device

        if isinstance(node_ids, torch.Tensor):
            node_ids = node_ids.cpu().tolist()

        result = self.fetch_batch(node_ids)
        return result.to(device or get_device())

    def supports_tensor_ops(self) -> bool:
        """Check if this store has optimized tensor operations."""
        return False
    
class GraphStore(ABC):
    """Abstract contract for Graph Structures (NetworkX, PyG, Neo4j)"""

    @abstractmethod
    def get_neighbors(self, node_id: Any) -> torch.Tensor:
        """Returns a 1D tensor of neighbor node IDs"""
        pass

    @abstractmethod
    def contains(self, node_id: Any) -> bool:
        """Checks if a node exists in the graph"""
        pass

    @abstractmethod
    def get_avg_degree(self) -> float:
        """Returns the average degree of the graph"""
        return self.avg_degree

    # ===== Tensor-native interface methods (optional, for GPU acceleration) =====

    def get_neighbors_tensor(
        self,
        node_id: int,
        device: str = None
    ) -> torch.Tensor:
        """
        Get neighbors as tensor on device.

        Args:
            node_id: Node to get neighbors for
            device: Target device (None for auto-detect)

        Returns:
            1D tensor of neighbor IDs on device

        Default implementation converts from get_neighbors() result.
        """
        from ..utils.device import get_device

        neighbors = self.get_neighbors(node_id)
        return neighbors.to(device or get_device())

    def get_neighbors_batch_tensor(
        self,
        node_ids: Union[Sequence[int], torch.Tensor],
        device: str = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch neighbor lookup returning tensors.

        Args:
            node_ids: Sequence or tensor of node IDs
            device: Target device (None for auto-detect)

        Returns:
            Tuple of:
                - neighbors: 2D tensor (batch, max_degree) padded with -1
                - mask: Boolean tensor indicating valid neighbors

        Default implementation calls get_neighbors() for each node.
        """
        from ..utils.device import get_device

        target_device = device or get_device()

        if isinstance(node_ids, torch.Tensor):
            node_ids = node_ids.cpu().tolist()

        all_neighbors = [self.get_neighbors(nid) for nid in node_ids]
        max_degree = max(len(n) for n in all_neighbors) if all_neighbors else 0

        if max_degree == 0:
            batch_size = len(node_ids)
            return (
                torch.full((batch_size, 1), -1, device=target_device, dtype=torch.long),
                torch.zeros((batch_size, 1), device=target_device, dtype=torch.bool)
            )

        batch_size = len(node_ids)
        neighbors = torch.full((batch_size, max_degree), -1, device=target_device, dtype=torch.long)
        mask = torch.zeros((batch_size, max_degree), device=target_device, dtype=torch.bool)

        for i, n in enumerate(all_neighbors):
            if len(n) > 0:
                neighbors[i, :len(n)] = n.to(target_device)
                mask[i, :len(n)] = True

        return neighbors, mask

    def supports_tensor_ops(self) -> bool:
        """Check if this store has optimized tensor operations."""
        return False


class EmbeddingProvider(ABC):
    """Abstract contract for Embedding Models (Cohere, OpenAI, Pre-computed Lookups)"""
    
    @abstractmethod
    def embed_query(self, query: Any) -> torch.Tensor:
        """Embeds a query string OR looks it up if using pre-computed"""
        pass

    @abstractmethod
    def embed_query_batch(self, queries: Sequence) -> Matrix:
        pass