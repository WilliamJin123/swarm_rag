"""
GPU Batch Embedding Cache for Evolution.

Pre-computes embeddings for all queries before evolution starts,
avoiding redundant embedding computations during genome evaluation.

Expected Impact: 10-20% speedup if embedding model is the bottleneck.
"""
import logging
from typing import List, Dict, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)

# Optional torch import for GPU operations
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class EmbeddingCacheStats:
    """Statistics about embedding cache usage."""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    precompute_time: float = 0.0
    total_embedding_time: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class QueryEmbeddingCache:
    """
    Cache for pre-computed query embeddings.

    Pre-computes embeddings for all queries at evolution start,
    then serves them from cache during genome evaluation.

    Benefits:
    1. Eliminates redundant embedding computations
    2. Enables GPU batching for faster embedding
    3. Provides consistent embeddings across generations

    Usage:
        # At evolution start
        cache = QueryEmbeddingCache(embedding_fn)
        cache.precompute(queries)

        # During evaluation
        embedding = cache.get(query)  # Fast lookup, no computation
    """

    def __init__(
        self,
        embedding_fn: Callable[[str], np.ndarray] = None,
        batch_embedding_fn: Callable[[List[str]], np.ndarray] = None,
        use_gpu: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize embedding cache.

        Args:
            embedding_fn: Function to embed a single query
            batch_embedding_fn: Optional function to embed multiple queries at once
            use_gpu: Whether to use GPU for batching if available
            batch_size: Batch size for GPU embedding
        """
        self.embedding_fn = embedding_fn
        self.batch_embedding_fn = batch_embedding_fn
        self.use_gpu = use_gpu and _TORCH_AVAILABLE
        self.batch_size = batch_size

        self._cache: Dict[str, np.ndarray] = {}
        self._embedding_dim: Optional[int] = None
        self.stats = EmbeddingCacheStats()

    def precompute(
        self,
        queries: List[str],
        show_progress: bool = True
    ) -> EmbeddingCacheStats:
        """
        Pre-compute embeddings for all queries.

        Args:
            queries: List of query strings
            show_progress: Whether to log progress

        Returns:
            EmbeddingCacheStats with precomputation details
        """
        start_time = time.time()

        unique_queries = list(set(queries))
        self.stats.total_queries = len(unique_queries)

        # Skip already cached queries
        to_embed = [q for q in unique_queries if q not in self._cache]

        if not to_embed:
            logger.info(f"All {len(unique_queries)} queries already cached")
            return self.stats

        logger.info(f"Pre-computing embeddings for {len(to_embed)} queries...")

        # Use batch embedding if available
        if self.batch_embedding_fn is not None:
            self._batch_embed(to_embed, show_progress)
        else:
            self._sequential_embed(to_embed, show_progress)

        self.stats.precompute_time = time.time() - start_time
        logger.info(
            f"Precomputed {len(to_embed)} embeddings in {self.stats.precompute_time:.2f}s "
            f"({len(to_embed) / max(0.01, self.stats.precompute_time):.1f} q/s)"
        )

        return self.stats

    def _batch_embed(self, queries: List[str], show_progress: bool):
        """Embed queries in batches using batch_embedding_fn."""
        total_batches = (len(queries) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            if show_progress and batch_num % 10 == 0:
                logger.debug(f"  Embedding batch {batch_num}/{total_batches}")

            start = time.time()
            embeddings = self.batch_embedding_fn(batch)
            self.stats.total_embedding_time += time.time() - start

            # Handle different return types
            if isinstance(embeddings, np.ndarray):
                for j, q in enumerate(batch):
                    self._cache[q] = embeddings[j]
                    if self._embedding_dim is None:
                        self._embedding_dim = embeddings[j].shape[0]
            elif _TORCH_AVAILABLE and isinstance(embeddings, torch.Tensor):
                embeddings_np = embeddings.cpu().numpy()
                for j, q in enumerate(batch):
                    self._cache[q] = embeddings_np[j]
                    if self._embedding_dim is None:
                        self._embedding_dim = embeddings_np[j].shape[0]
            else:
                # Assume list-like
                for j, q in enumerate(batch):
                    emb = np.array(embeddings[j])
                    self._cache[q] = emb
                    if self._embedding_dim is None:
                        self._embedding_dim = emb.shape[0]

    def _sequential_embed(self, queries: List[str], show_progress: bool):
        """Embed queries one at a time using embedding_fn."""
        for i, q in enumerate(queries):
            if show_progress and (i + 1) % 100 == 0:
                logger.debug(f"  Embedded {i + 1}/{len(queries)} queries")

            start = time.time()
            embedding = self.embedding_fn(q)
            self.stats.total_embedding_time += time.time() - start

            # Convert to numpy if needed
            if _TORCH_AVAILABLE and isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            elif not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)

            self._cache[q] = embedding
            if self._embedding_dim is None:
                self._embedding_dim = embedding.shape[0]

    def get(self, query: str, as_tensor: bool = False) -> Optional[np.ndarray]:
        """
        Get embedding for a query from cache.

        Args:
            query: Query string
            as_tensor: If True, return torch.Tensor on GPU if available.
                       If False (default), return numpy array.

        Returns:
            Embedding as numpy array or torch tensor, None if not cached and no embedding_fn.
        """
        if query in self._cache:
            self.stats.cache_hits += 1
            emb = self._cache[query]

            if as_tensor and _TORCH_AVAILABLE:
                if isinstance(emb, torch.Tensor):
                    return emb
                # Convert numpy to tensor on GPU if available
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                return torch.tensor(emb, device=device, dtype=torch.float32)

            # Return as numpy (default)
            if isinstance(emb, torch.Tensor):
                return emb.cpu().numpy()
            return emb

        self.stats.cache_misses += 1

        # Optionally compute on miss
        if self.embedding_fn is not None:
            start = time.time()
            embedding = self.embedding_fn(query)
            self.stats.total_embedding_time += time.time() - start

            # Store as numpy in cache (canonical format)
            if _TORCH_AVAILABLE and isinstance(embedding, torch.Tensor):
                embedding_np = embedding.cpu().numpy()
            elif not isinstance(embedding, np.ndarray):
                embedding_np = np.array(embedding)
            else:
                embedding_np = embedding

            self._cache[query] = embedding_np

            # Return in requested format
            if as_tensor and _TORCH_AVAILABLE:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                return torch.tensor(embedding_np, device=device, dtype=torch.float32)

            return embedding_np

        return None

    def get_batch(self, queries: List[str]) -> Dict[str, np.ndarray]:
        """
        Get embeddings for multiple queries.

        Returns dict mapping query to embedding.
        Missing queries are computed if embedding_fn is available.
        """
        result = {}
        missing = []

        for q in queries:
            if q in self._cache:
                self.stats.cache_hits += 1
                result[q] = self._cache[q]
            else:
                self.stats.cache_misses += 1
                missing.append(q)

        # Batch compute missing
        if missing and self.batch_embedding_fn is not None:
            self._batch_embed(missing, show_progress=False)
            for q in missing:
                result[q] = self._cache[q]
        elif missing and self.embedding_fn is not None:
            self._sequential_embed(missing, show_progress=False)
            for q in missing:
                result[q] = self._cache[q]

        return result

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._embedding_dim = None
        self.stats = EmbeddingCacheStats()

    @property
    def size(self) -> int:
        """Number of cached embeddings."""
        return len(self._cache)

    @property
    def embedding_dim(self) -> Optional[int]:
        """Dimension of cached embeddings."""
        return self._embedding_dim

    def get_all_embeddings_matrix(
        self,
        queries: List[str],
        as_tensor: bool = False
    ) -> np.ndarray:
        """
        Get embeddings for queries as a single matrix.

        Useful for batch similarity computations.

        Args:
            queries: Ordered list of queries
            as_tensor: If True, return torch.Tensor on GPU if available.

        Returns:
            np.ndarray or torch.Tensor of shape (len(queries), embedding_dim)
        """
        embeddings = []
        for q in queries:
            emb = self.get(q)  # Get as numpy first
            if emb is not None:
                embeddings.append(emb)
            else:
                # Return zero vector for missing
                if self._embedding_dim:
                    embeddings.append(np.zeros(self._embedding_dim, dtype=np.float32))
                else:
                    raise ValueError(f"Cannot get embedding for '{q}' and dimension unknown")

        matrix = np.vstack(embeddings)

        if as_tensor and _TORCH_AVAILABLE:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            return torch.tensor(matrix, device=device, dtype=torch.float32)

        return matrix

    def get_batch_tensor(self, queries: List[str]) -> Optional["torch.Tensor"]:
        """
        Get embeddings for queries as a GPU tensor.

        Convenience method for GPU-accelerated operations.

        Args:
            queries: List of query strings

        Returns:
            torch.Tensor on GPU if available, None if torch not available
        """
        if not _TORCH_AVAILABLE:
            return None

        return self.get_all_embeddings_matrix(queries, as_tensor=True)


class EmbeddingCacheProvider:
    """
    Singleton provider for the global embedding cache.

    Allows sharing the same cache across different parts of the evolution system.
    """

    _instance: Optional[QueryEmbeddingCache] = None

    @classmethod
    def get_or_create(
        cls,
        embedding_fn: Callable = None,
        batch_embedding_fn: Callable = None,
        **kwargs
    ) -> QueryEmbeddingCache:
        """Get or create the global embedding cache."""
        if cls._instance is None:
            cls._instance = QueryEmbeddingCache(
                embedding_fn=embedding_fn,
                batch_embedding_fn=batch_embedding_fn,
                **kwargs
            )
        return cls._instance

    @classmethod
    def get(cls) -> Optional[QueryEmbeddingCache]:
        """Get the global embedding cache if it exists."""
        return cls._instance

    @classmethod
    def clear(cls):
        """Clear and remove the global cache."""
        if cls._instance is not None:
            cls._instance.clear()
        cls._instance = None
