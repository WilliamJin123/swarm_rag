import math
from typing import Any, List, Optional
import numpy as np
from ..interfaces.base import VectorStore, GraphStore, EmbeddingProvider
from ..utils import fail_on_missing_imports
try:
    import torch
    import faiss
    from stark_qa.skb import SKB
except ImportError:
    fail_on_missing_imports(
                modules=["torch", "stark_qa", "faiss"], 
                extra_name="stark"
            )
    
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
    
    def __init__(self, skb_data: SKB, dataset: str):
        """
        Ingests a STaRK SKB object. 
        """

        print("Initializing graph adapter from STaRK SKB...")
        self.skb = skb_data

        if dataset not in AVG_LOG_DEGREE_BY_DATASET:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.dataset = dataset
        self.avg_log_degree = AVG_LOG_DEGREE_BY_DATASET[dataset]
    
    def get_neighbors(self, node_id) -> list:
        return self.skb.get_neighbor_nodes(node_id)
    
    def contains(self, node_id) -> bool:
        return self.skb.node_info.get(node_id, "") != ""
    
    def centrality_heuristic(self, ctx) -> float:
        graph: StarkSKBAdapter = ctx["graph"]
        degree = len(graph.get_neighbors(ctx["target_id"]))

        return min(1.0, math.log(1 + degree) / graph.avg_log_degree)
    
# --- 2. Vector Store Adapter for STaRK Tensors ---
class StarkInMemoryVectorStore(VectorStore):
    def __init__(self, doc_embs: dict[int, torch.Tensor]):
        """
        Wraps the raw dictionary of {id: tensor} into a FAISS index for speed.
        """
        if not doc_embs:
            raise ValueError("Document embeddings dictionary cannot be empty")
            
        self.doc_embs = doc_embs
        self.ids = list(doc_embs.keys())
        
        # Convert to numpy matrix for FAISS
        # Assuming all tensors are same shape
        first_tensor = doc_embs[self.ids[0]]
        dim = first_tensor.squeeze().shape[0]

        matrix = np.stack([
            t.squeeze().cpu().numpy() 
            for t in doc_embs.values()
        ]).astype('float32')
        
        # Normalize for Cosine Similarity (Inner Product on normalized vectors)
        faiss.normalize_L2(matrix)
        
        self.index = faiss.IndexFlatIP(dim) # Inner Product index
        self.index.add(matrix)
        
        # Map Faiss integer ID back to STaRK Node ID
        self.faiss_id_to_real_id = {i: real_id for i, real_id in enumerate(self.ids)}
        self.real_id_to_tensor = {real_id: matrix[i] for i, real_id in enumerate(self.ids)}

    def search(self, query_vec: np.ndarray, limit: int):
        """
        Search for similar vectors using FAISS.
        """
        if query_vec is None:
            raise ValueError(
                "The query vector provided to the search function is None. "
                "This usually means the embedding for the query ID was not found. "
                "Please check that your query embeddings dictionary contains the ID for the query you are trying to retrieve."
            )
        # Ensure query is 2D float32 and normalized
        q = query_vec.reshape(1, -1).astype('float32')
        faiss.normalize_L2(q)
        
        scores, indices = self.index.search(q, min(limit, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            real_id = self.faiss_id_to_real_id[idx]
            results.append({'id': real_id, 'score': float(score)})
        return results

    def fetch_batch(self, node_ids) -> List[Optional[np.ndarray]]:
        # Return raw vectors for the Swarm logic
        return [self.real_id_to_tensor.get(nid) for nid in node_ids]
    
    def fetch(self, node_id: int) -> Optional[np.ndarray]:
        """
        Provides a fast, direct O(1) lookup for a single vector.
        This is ideal for use with an LRU cache.
        """
        return self.real_id_to_tensor.get(node_id)


# --- 3. Embedding Adapter (Pre-computed Lookup) ---
class StarkPreComputedEmbeddingHandler(EmbeddingProvider):
    def __init__(self, query_embs: dict[int, torch.Tensor]):
        self.query_embs = query_embs

    def embed_query(self, query_id: int) -> np.ndarray:
        # Note: We expect the input to be the Query ID, not the string text,
        # because we are looking up pre-computed values.
        if query_id not in self.query_embs:
            raise ValueError(f"Query ID {query_id} not found in pre-computed embeddings.")
        tensor = self.query_embs[query_id]
        # Handle CUDA tensors by moving to CPU first
        return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
    

    def embed_query_batch(self, query_ids: list[int]) -> np.ndarray:
        return np.stack([self.embed_query(qid) for qid in query_ids])
    
