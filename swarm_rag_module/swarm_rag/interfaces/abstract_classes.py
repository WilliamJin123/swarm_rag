from abc import ABC, abstractmethod, abstractproperty
from typing import List, Dict, Any, Optional, Sequence, TypeAlias, Union, Tuple
import torch

# Matrix type alias: PyTorch tensors only
Matrix: TypeAlias = torch.Tensor  # Shape (N, D)

class VectorStore(ABC):
    """
    Abstract contract for Vector Databases (LanceDB, Chroma, Faiss, In-Memory).

    All methods are tensor-native and return data on the store's configured device.
    Callers are responsible for calling .cpu() if they need CPU data.

    Device configuration should be set at construction time and flow down from
    the caller (e.g., EvolutionEngine -> SwarmRetriever -> VectorStore).
    """

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the device this store operates on ('cuda' or 'cpu')."""
        pass

    @abstractmethod
    def search(self, query_vec: torch.Tensor, limit: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find top-k most similar documents.

        Args:
            query_vec: Query embedding vector
            limit: Maximum number of results to return

        Returns:
            Tuple of (ids_tensor, scores_tensor) on device, sorted by score descending
        """
        pass

    @abstractmethod
    def search_batch(self, query_vecs: torch.Tensor, limit: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch search for multiple queries.

        Args:
            query_vecs: Query vectors of shape (n_queries, dim)
            limit: Maximum results per query

        Returns:
            Tuple of (ids_tensor, scores_tensor) with shape (n_queries, limit)
        """
        pass

    @abstractmethod
    def fetch_batch(self, node_ids: Union[Sequence[Any], torch.Tensor]) -> Matrix:
        """
        Fetch embeddings for multiple documents.

        Args:
            node_ids: Sequence or tensor of node IDs

        Returns:
            Tensor of shape (N, D) on device. NaN for invalid indices.
        """
        pass

    @abstractmethod
    def fetch(self, node_id: Any) -> Optional[torch.Tensor]:
        """
        Fetch embedding for a single document.

        Returns:
            Tensor on device, or None if not found
        """
        pass
    
class GraphStore(ABC):
    """
    Abstract contract for Graph Structures (NetworkX, PyG, Neo4j).

    All methods are tensor-native and return data on the store's configured device.
    Callers are responsible for calling .cpu() if they need CPU data.

    Device configuration should be set at construction time and flow down from
    the caller (e.g., EvolutionEngine -> SwarmRetriever -> GraphStore).
    """

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the device this store operates on ('cuda' or 'cpu')."""
        pass

    @abstractmethod
    def get_neighbors(self, node_id: Any) -> torch.Tensor:
        """
        Get neighbors for a single node.

        Returns:
            1D tensor of neighbor IDs on device
        """
        pass

    @abstractmethod
    def get_neighbors_batch(
        self,
        node_ids: Union[Sequence[int], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch neighbor lookup for multiple nodes.

        Args:
            node_ids: Sequence or tensor of node IDs

        Returns:
            Tuple of:
                - neighbors: 2D tensor (batch, max_degree) padded with -1
                - mask: Boolean tensor indicating valid neighbors
        """
        pass

    @abstractmethod
    def contains(self, node_id: Any) -> bool:
        """Checks if a node exists in the graph"""
        pass

    @abstractmethod
    def get_avg_degree(self) -> float:
        """Returns the average degree of the graph"""
        return self.avg_degree

    @abstractmethod
    def get_degree(self, node_id: Any) -> int:
        """Returns the degree (number of neighbors) of a single node."""
        pass

    def get_degrees_batch(
        self,
        node_ids: Union[Sequence[int], torch.Tensor]
    ) -> torch.Tensor:
        """Batch degree lookup for multiple nodes. Returns tensor of degrees on device."""
        pass



class EmbeddingProvider(ABC):
    """Abstract contract for Embedding Models (Cohere, OpenAI, Pre-computed Lookups)"""
    
    @abstractmethod
    def embed_query(self, query: Any) -> torch.Tensor:
        """Embeds a query string OR looks it up if using pre-computed"""
        pass

    @abstractmethod
    def embed_query_batch(self, queries: Sequence) -> Matrix:
        pass