from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np

class VectorStore(ABC):
    """Abstract contract for Vector Databases (LanceDB, Chroma, Faiss, In-Memory)"""
    
    @abstractmethod
    def search(self, query_vec: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """Returns list of dicts: [{'id': ..., 'score': ...}, ...]"""
        pass

    @abstractmethod
    def fetch_batch(self, node_ids: List[Any]) -> List[Optional[np.ndarray]]:
        """Returns ordered list of vectors"""
        pass

    @abstractmethod
    def fetch(self, node_ids: Any) -> Optional[np.ndarray]:
        """Returns single vector"""
        pass
    
class GraphStore(ABC):
    """Abstract contract for Graph Structures (NetworkX, PyG, Neo4j)"""
    
    @abstractmethod
    def get_neighbors(self, node_id: Any) -> List[Any]:
        """Returns a list of neighbor node IDs"""
        pass

    @abstractmethod
    def contains(self, node_id: Any) -> bool:
        """Checks if a node exists in the graph"""
        pass


class EmbeddingProvider(ABC):
    """Abstract contract for Embedding Models (Cohere, OpenAI, Pre-computed Lookups)"""
    
    @abstractmethod
    def embed_query(self, query: Union[str, Any]) -> np.ndarray:
        """Embeds a query string OR looks it up if using pre-computed"""
        pass

    @abstractmethod
    def embed_query_batch(self, queries: list) -> List[np.ndarray]:
        pass