import math
import os
from typing import Dict, List, Optional
import numpy as np
from numpy.typing import NDArray
from multiprocessing import shared_memory
import scipy.sparse as sp
import atexit

from ..interfaces.abstract_classes import VectorStore, GraphStore, EmbeddingProvider, Matrix
from ..interfaces.enums import HeuristicKey
from ..utils import fail_on_missing_imports, LRUCache
from ..core import HeuristicContext, HeuristicRegistry

try:
    import torch
    import faiss
    from stark_qa.load_skb import SKB
except ImportError:
    fail_on_missing_imports(
                modules=["torch", "stark_qa", "faiss"], 
                extra_name="stark"
            )

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

# --- 1. Graph Adapter for STaRK SKB ---
class StarkSKBAdapter(GraphStore):
    def __init__(
        self, 
        skb_data: 'SKB', 
        dataset: str, 
        cache_neighbors: bool = True,
        cache_degrees: bool = True,
        neighbor_cache_size: int = 10000,
        degree_cache_size: int = 10000,
        adjacency_dict: Optional[Dict[int, List[int]]] = None,
        cache_path=None
    ):
        """
        Uses CSR (Compressed Sparse Row) representation to ingest a STaRK SKB object. 
        """
        self.skb = skb_data
        if dataset not in AVG_LOG_DEGREE_BY_DATASET:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.dataset = dataset
        self.cache_path = cache_path or f"./adjacency_cache/graph_{dataset}.npz"
        self.avg_log_degree = AVG_LOG_DEGREE_BY_DATASET[dataset]
        
        self.cache_neighbors = cache_neighbors
        self.cache_degrees = cache_degrees

        if os.path.exists(self.cache_path):
            logger.info(f"Loading CSR graph from {self.cache_path}...")
            self.adj_matrix = sp.load_npz(self.cache_path)
            self.use_precomputed = True
        elif adjacency_dict is not None:
            logger.info("Converting dict to CSR and saving for next time...")
            self.adj_matrix = self._dict_to_csr(adjacency_dict)
            sp.save_npz(self.cache_path, self.adj_matrix)
            self.use_precomputed = True
        else:
            logger.warning("No pre-computed adjacency provided. Falling back to LRU cache (Slow).")
            self.use_precomputed = False
            
            if self.cache_neighbors:
                self.neighbor_cache = LRUCache(neighbor_cache_size)
            if self.cache_degrees:
                self.degree_cache = LRUCache(degree_cache_size)

    def _dict_to_csr(self, adj_dict: Dict[int, List[int]]) -> sp.csr_matrix:
        """Converts adjacency dictionary to a memory-efficient CSR matrix."""
        nodes = sorted(adj_dict.keys())
        max_node = nodes[-1] if nodes else 0
        
        indptr = [0]
        indices = []
        # We use a mapping if node IDs are non-contiguous, but STaRK is usually 0-indexed
        for i in range(max_node + 1):
            neighbors = adj_dict.get(i, [])
            indices.extend(neighbors)
            indptr.append(len(indices))
        
        # Use dummy data (1s) as we only care about topology/indices
        data = np.ones(len(indices), dtype=np.int8)
        return sp.csr_matrix((data, indices, indptr), shape=(max_node + 1, max_node + 1))

    def get_neighbors(self, node_id: int) -> np.ndarray:
        if self.use_precomputed:
            if node_id >= self.adj_matrix.shape[0]: return np.array([])
            start = self.adj_matrix.indptr[node_id]
            end = self.adj_matrix.indptr[node_id+1]
            # Treat as READ-ONLY.
            return self.adj_matrix.indices[start:end]
        
        if self.cache_neighbors:
            cached = self.neighbor_cache.get(node_id)
            if cached is not None: return np.array(cached)
        
        neighbors = self.skb.get_neighbor_nodes(node_id)
        np_neighbors = np.array(neighbors)
        
        if self.cache_neighbors:
            self.neighbor_cache.set(node_id, np_neighbors)
            
        return np_neighbors
            
    def get_degree(self, node_id: int) -> int:
        if self.use_precomputed:
            if node_id >= self.adj_matrix.shape[0]: return 0
            return self.adj_matrix.indptr[node_id+1] - self.adj_matrix.indptr[node_id]
        
        if self.cache_degrees:
            cached = self.degree_cache.get(node_id)
            if cached is not None: return cached

        neighbors = self.get_neighbors(node_id)
        deg = len(neighbors)
        
        if self.cache_degrees:
            self.degree_cache.set(node_id, deg)
            
        return deg

    def contains(self, node_id: int) -> bool:
        if self.use_precomputed:
            return node_id < self.adj_matrix.shape[0]
        return self.skb.node_info.get(node_id, "") != ""

    def get_avg_degree(self) -> float:
        """Returns the average degree of the graph"""
        if self.dataset in AVG_DEGREE_BY_DATASET:
            return AVG_DEGREE_BY_DATASET[self.dataset]
        return 10.0 # Default fallback
    
    @staticmethod
    @HeuristicRegistry.register_movement(HeuristicKey.STARK_CENTRALITY)
    def centrality_heuristic(ctx :HeuristicContext) -> np.ndarray:
        """
        Vectorized centrality heuristic. 
        Uses pre-fetched ctx.node_degrees for speed.
        """
        graph: StarkSKBAdapter = ctx.graph
        log_degrees = np.log(1 + ctx.node_degrees)

        #Sigmoid normalization
        normalized = log_degrees / (log_degrees + graph.avg_log_degree + 1e-8)
        
        return normalized
        
    
# --- 2. Vector Store Adapter for STaRK Tensors ---
class StarkInMemoryVectorStore(VectorStore):
    def __init__(self, doc_embs: Dict[int, torch.Tensor], shared_name: Optional[str] = None):
        """
        Wraps doc embeddings in SharedMemory to prevent 4x RAM usage in parallel mode.
        """
        self._parent_pid = os.getpid()
        faiss.omp_set_num_threads(1)

        raw_ids = np.array(list(doc_embs.keys()))
        self.id_dtype = raw_ids.dtype
        sorted_keys = np.sort(raw_ids)
        
        first_tensor = doc_embs[sorted_keys[0]].detach().cpu().numpy().squeeze()
        self.dim = first_tensor.shape[0]
        self.vec_dtype = first_tensor.dtype
        self.n_docs = len(sorted_keys)

        matrix_bytes = self.n_docs * self.dim * np.dtype(self.vec_dtype).itemsize
        id_bytes = self.n_docs * np.dtype(self.id_dtype).itemsize

        if shared_name:
            # CHILD PROCESS PATH
            self.shm_matrix = shared_memory.SharedMemory(name=shared_name)
            self.shm_ids = shared_memory.SharedMemory(name=f"{shared_name}_ids")
            self._is_owner = False
        else:
            # PARENT PROCESS PATH
            shared_name = f"stark_vstore_{os.getpid()}"
            self.shm_matrix = shared_memory.SharedMemory(create=True, size=matrix_bytes, name=shared_name)
            self.shm_ids = shared_memory.SharedMemory(create=True, size=id_bytes, name=f"{shared_name}_ids")
            self._is_owner = True

        self.matrix = np.ndarray((self.n_docs, self.dim), dtype=self.vec_dtype, buffer=self.shm_matrix.buf)
        self.ids = np.ndarray((self.n_docs,), dtype=self.id_dtype, buffer=self.shm_ids.buf)

        if self._is_owner:
            logger.info(f"Parent process: Populating shared memory for {self.n_docs} embeddings...")
            tensor_list = [doc_embs[rid] for rid in sorted_keys]
            stacked_matrix = torch.stack(tensor_list).detach().cpu().numpy().astype(self.vec_dtype).squeeze()
            faiss.normalize_L2(stacked_matrix)
            np.copyto(self.matrix, stacked_matrix)
            np.copyto(self.ids, sorted_keys)
            logger.info("Parent process: Shared memory populated.")
        
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.matrix)

        self.real_id_to_idx = {int(real_id): i for i, real_id in enumerate(self.ids)}

        # warmup
        _ = np.sum(self.matrix[0]) 
        _ = np.sum(self.ids[:min(100, len(self.ids))])

        atexit.register(self.close)

    def search(self, query_vec: np.ndarray, limit: int):
        # FAISS requires numpy on CPU
        query_vec_cpu = np.asarray(query_vec)
        q = query_vec_cpu.reshape(1, -1).astype('float32')
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, min(limit, self.n_docs))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            real_id = int(self.ids[idx])
            results.append({'id': real_id, 'score': float(score)})
        return results
    
    def fetch_batch(self, node_ids) -> Matrix:
        indices = [self.real_id_to_idx.get(nid, -1) for nid in node_ids]
        indices_arr = np.array(indices, dtype=np.int64)

        # Use NaN for missing vectors
        result = np.full((len(node_ids), self.dim), np.nan, dtype=self.vec_dtype)

        mask = (indices_arr >= 0)
        
        if np.any(mask):
            valid_internal_indices = indices_arr[mask]
            result[mask] = self.matrix[valid_internal_indices]
            
        return result
    
    def fetch(self, node_id: int) -> Optional[np.ndarray]:
        idx = self.real_id_to_idx.get(node_id)
        if idx is not None:
            return self.matrix[idx]
        logger.warning(
            f"VectorStore: Node ID {node_id} was requested but not found in the embedding map. "
            "This indicates a mismatch between graph nodes and document embeddings."
        )
        return None

    def close(self):
        for shm_attr in ['shm_matrix', 'shm_ids']:
            if hasattr(self, shm_attr):
                shm = getattr(self, shm_attr)
                shm.close()
                # Unlink ONLY in the parent process
                if self._is_owner and os.getpid() == self._parent_pid:
                    try:
                        shm.unlink()
                        logger.info(f"Unlinked {shm_attr}")
                    except:
                        pass


# --- 3. Embedding Adapter (Pre-computed Lookup) ---
class StarkPreComputedEmbeddingHandler(EmbeddingProvider):
    def __init__(self, query_embs: dict[int, torch.Tensor]):
        """Standardized NumPy pre-conversion to avoid GIL/Torch overhead."""
        self.query_embs = {}
        for qid, emb in query_embs.items():
            if hasattr(emb, 'numpy'):
                # Handle torch tensors (move to CPU if needed)
                arr = emb.cpu().detach().numpy() if hasattr(emb, 'is_cuda') and emb.is_cuda else emb.numpy()
            else:
                arr = np.array(emb)
            
            self.query_embs[qid] = arr.squeeze()

    def embed_query(self, query_id: int) -> np.ndarray:
        return self.query_embs[query_id]
        
    def embed_query_batch(self, query_ids: list[int]) -> Matrix:
        """
        Returns a 2D matrix (N_queries, Dimension) of pre-computed embeddings.
        """
        # Fetch individual arrays and stack them into a single (N, D) matrix
        return np.stack([self.query_embs[qid] for qid in query_ids])