from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Sequence, TypeAlias, Union
import numpy as np
from numpy.typing import NDArray

Matrix: TypeAlias = NDArray[Any]  # Shape (N, D)

class VectorStore(ABC):
    """Abstract contract for Vector Databases (LanceDB, Chroma, Faiss, In-Memory)"""
    
    @abstractmethod
    def search(self, query_vec: np.ndarray, limit: int) -> Sequence[Dict[str, Any]]:
        """Returns sequence of dicts: [{'id': ..., 'score': ...}, ...]"""
        pass

    @abstractmethod
    def fetch_batch(self, node_ids: Sequence[Any]) -> Matrix:
        """Returns a 2D matrix of shape (N, D). NaN for invalid indices."""
        pass

    @abstractmethod
    def fetch(self, node_ids: Any) -> Optional[np.ndarray]:
        """Returns single vector"""
        pass
    
class GraphStore(ABC):
    """Abstract contract for Graph Structures (NetworkX, PyG, Neo4j)"""
    
    @abstractmethod
    def get_neighbors(self, node_id: Any) -> np.ndarray:
        """Returns a sequence of neighbor node IDs (pref np.ndarray)"""
        pass

    @abstractmethod
    def contains(self, node_id: Any) -> bool:
        """Checks if a node exists in the graph"""
        pass

    @abstractmethod
    def get_avg_degree(self) -> float:
        """Returns the average degree of the graph"""
        return self.avg_degree


class EmbeddingProvider(ABC):
    """Abstract contract for Embedding Models (Cohere, OpenAI, Pre-computed Lookups)"""
    
    @abstractmethod
    def embed_query(self, query: Any) -> np.ndarray:
        """Embeds a query string OR looks it up if using pre-computed"""
        pass

    @abstractmethod
    def embed_query_batch(self, queries: Sequence) -> Matrix:
        pass