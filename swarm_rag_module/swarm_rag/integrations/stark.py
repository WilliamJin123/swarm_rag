import math
from typing import Any, Optional
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
    

# --- 1. Graph Adapter for STaRK SKB ---
class StarkSKBAdapter(GraphStore):
    def __init__(self, skb_data: SKB):
        """
        Ingests a STaRK SKB object. 
        """

        print("Initializing graph adapter from STaRK SKB...")
        self.skb = skb_data

        self.max_degree = max(
            len(self.skb.get_neighbor_nodes(node))
            for node in self.skb.node_info.keys()
        )
        self.max_log_degree = math.log(1 + self.max_degree) if self.max_degree > 0 else 1.0
    
    def get_neighbors(self, node_id) -> list:
        # raw_neighbors = self.skb.get_neighbor_nodes(idx=node_id)
        
        # processed_neighbors = []
        # for neighbor in raw_neighbors:
        #     if isinstance(neighbor, dict):
        #         # If it's a dict, find the first value that looks like an ID.
        #         # This is more robust than hardcoding a key like 'id'.
        #         for val in neighbor.values():
        #             if isinstance(val, (int, str)):
        #                 processed_neighbors.append(int(val))
        #                 break  # Found the ID, stop searching this dict
        #     elif isinstance(neighbor, (int, str)):
        #         # If it's already an int or string, just convert and add it.
        #         processed_neighbors.append(int(neighbor))
        
        # return processed_neighbors
        return self.skb.get_neighbor_nodes(node_id)
    
    def contains(self, node_id) -> bool:
        return self.skb.node_info.get(node_id, "") != ""
    
    @staticmethod
    def log_degree_centrality(self, ctx) -> float:
        graph: StarkSKBAdapter = ctx['graph']
        neighbors = graph.get_neighbors(ctx['target_id'])
        degree = len(neighbors)
        return math.log(1 + degree) / self.max_log_degree # Normalized to [0,1]
    
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

    def fetch_batch(self, node_ids) -> dict[Any, np.ndarray]:
        # Return raw vectors for the Swarm logic
        return {
            nid: self.real_id_to_tensor[nid] 
            for nid in node_ids 
            if nid in self.real_id_to_tensor
        }
    
    def get_single_vector(self, node_id: int) -> Optional[np.ndarray]:
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
    
