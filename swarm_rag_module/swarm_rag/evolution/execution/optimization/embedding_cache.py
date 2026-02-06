"""
GPU Batch Embedding Cache for Evolution.

Pre-computes embeddings for all queries before evolution starts,
avoiding redundant embedding computations during genome evaluation.

Expected Impact: 10-20% speedup if embedding model is the bottleneck.
"""
import logging
import json
import copy
import threading
from typing import List, Dict, Optional, Any, Callable
import torch
from dataclasses import dataclass, field
import time

from ....utils.device import get_device

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingCacheStats:
    """Statistics about embedding cache usage."""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    precompute_time: float = 0.0
    total_embedding_time: float = 0.0

    # Per-generation tracking (reset each generation)
    generation_hits: int = 0
    generation_misses: int = 0
    compute_time_saved_sec: float = 0.0

    # Running average of embedding time per query (for time-saved calculation)
    _avg_embed_time_sec: float = 0.0
    _embed_time_samples: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def generation_hit_rate(self) -> float:
        """Hit rate for current generation."""
        total = self.generation_hits + self.generation_misses
        return self.generation_hits / total if total > 0 else 0.0

    @property
    def hits(self) -> int:
        """Total cache hits (protocol-compatible alias for cache_hits)."""
        return self.cache_hits

    @property
    def misses(self) -> int:
        """Total cache misses (protocol-compatible alias for cache_misses)."""
        return self.cache_misses

    def reset_generation(self):
        """Reset per-generation counters while keeping cumulative stats."""
        self.generation_hits = 0
        self.generation_misses = 0
        self.compute_time_saved_sec = 0.0


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
        embedding_fn: Callable[[str], torch.Tensor] = None,
        batch_embedding_fn: Callable[[List[str]], torch.Tensor] = None,
        device: str = None,
        batch_size: int = 32,
        maxsize: int = 10000,
    ):
        """
        Initialize embedding cache.

        Args:
            embedding_fn: Function to embed a single query
            batch_embedding_fn: Optional function to embed multiple queries at once
            device: Device for storing embeddings ("cuda", "mps", or "cpu", auto-detected if None)
            batch_size: Batch size for GPU embedding
            maxsize: Maximum number of embeddings to cache (LRU eviction when exceeded)
        """
        self.embedding_fn = embedding_fn
        self.batch_embedding_fn = batch_embedding_fn
        self.batch_size = batch_size
        self.maxsize = maxsize

        # Resolve device
        if device is not None:
            self._device = device
        else:
            self._device = get_device()

        # Validate device
        if self._device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if self._device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")

        self._cache: Dict[str, Any] = {}  # Stores torch.Tensor
        self._access_order: List[str] = []  # Track LRU order for eviction
        self._embedding_dim: Optional[int] = None
        self.stats = EmbeddingCacheStats()

        logger.debug(f"EmbeddingCache initialized with device={self._device}, maxsize={maxsize}")

    def _evict_if_needed(self):
        """Evict oldest entries if cache exceeds maxsize."""
        while len(self._cache) > self.maxsize and self._access_order:
            oldest = self._access_order.pop(0)
            if oldest in self._cache:
                del self._cache[oldest]

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
        """
        Embed queries in batches using batch_embedding_fn.

        Unified storage: converts to target storage format once per batch,
        avoiding bidirectional conversions.
        """
        total_batches = (len(queries) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            if show_progress and batch_num % 10 == 0:
                logger.debug(f"  Embedding batch {batch_num}/{total_batches}")

            start = time.time()
            embeddings = self.batch_embedding_fn(batch)
            elapsed = time.time() - start
            self.stats.total_embedding_time += elapsed

            # Update running average of embedding time per query (EMA with alpha=0.1)
            per_query_time = elapsed / max(1, len(batch))
            if self.stats._embed_time_samples == 0:
                self.stats._avg_embed_time_sec = per_query_time
            else:
                alpha = 0.1
                self.stats._avg_embed_time_sec = (
                    alpha * per_query_time +
                    (1 - alpha) * self.stats._avg_embed_time_sec
                )
            self.stats._embed_time_samples += 1

            # Convert to target storage format once
            storage_embeddings = self._convert_to_storage_format(embeddings)

            # Store each embedding
            for j, q in enumerate(batch):
                self._cache[q] = storage_embeddings[j].detach()  # Detach to prevent memory leaks
                self._access_order.append(q)
                if self._embedding_dim is None:
                    if hasattr(storage_embeddings[j], 'shape'):
                        self._embedding_dim = storage_embeddings[j].shape[0]
                    else:
                        self._embedding_dim = len(storage_embeddings[j])

            # Evict oldest entries if cache exceeds maxsize
            self._evict_if_needed()

    def _to_device(self, embeddings) -> torch.Tensor:
        """
        Convert embeddings to tensor on the configured device.

        Args:
            embeddings: Input embeddings (torch.Tensor or list)

        Returns:
            Tensor on self._device
        """
        if isinstance(embeddings, torch.Tensor):
            if str(embeddings.device).startswith(self._device):
                return embeddings
            return embeddings.to(self._device)
        return torch.as_tensor(embeddings, device=self._device, dtype=torch.float32)

    def _convert_to_storage_format(self, embeddings):
        """Convert embeddings to tensor on configured device. Deprecated: use _to_device."""
        return self._to_device(embeddings)

    def _sequential_embed(self, queries: List[str], show_progress: bool):
        """
        Embed queries one at a time using embedding_fn.

        Uses unified storage format conversion.
        """
        for i, q in enumerate(queries):
            if show_progress and (i + 1) % 100 == 0:
                logger.debug(f"  Embedded {i + 1}/{len(queries)} queries")

            start = time.time()
            embedding = self.embedding_fn(q)
            self.stats.total_embedding_time += time.time() - start

            # Convert single embedding to storage format
            storage_emb = self._convert_single_to_storage_format(embedding)

            self._cache[q] = storage_emb
            self._access_order.append(q)
            if self._embedding_dim is None:
                dim = storage_emb.shape[0] if hasattr(storage_emb, 'shape') else len(storage_emb)
                self._embedding_dim = dim

            # Evict oldest entries if cache exceeds maxsize
            self._evict_if_needed()

    def _convert_single_to_storage_format(self, embedding):
        """Convert single embedding to tensor on configured device. Deprecated: use _to_device."""
        return self._to_device(embedding)

    def get(self, query: str, as_tensor: bool = False) -> Optional[torch.Tensor]:
        """
        Get embedding for a query from cache.

        Args:
            query: Query string
            as_tensor: Deprecated parameter (kept for backward compatibility).
                       Embeddings are always returned on the storage device.

        Returns:
            Embedding as torch tensor on storage device, None if not cached and no embedding_fn.
        """
        if query in self._cache:
            self.stats.cache_hits += 1
            self.stats.generation_hits += 1
            # Track time saved on cache hit
            self.stats.compute_time_saved_sec += self.stats._avg_embed_time_sec
            # Update access order for LRU
            if query in self._access_order:
                self._access_order.remove(query)
            self._access_order.append(query)
            return self._cache[query]

        self.stats.cache_misses += 1
        self.stats.generation_misses += 1

        # Compute on miss if embedding function available
        if self.embedding_fn is not None:
            start = time.time()
            embedding = self.embedding_fn(query)
            elapsed = time.time() - start
            self.stats.total_embedding_time += elapsed

            # Update running average of embedding time per query
            if self.stats._embed_time_samples == 0:
                self.stats._avg_embed_time_sec = elapsed
            else:
                alpha = 0.1
                self.stats._avg_embed_time_sec = (
                    alpha * elapsed +
                    (1 - alpha) * self.stats._avg_embed_time_sec
                )
            self.stats._embed_time_samples += 1

            # Store on configured device
            cache_emb = self._to_device(embedding)
            self._cache[query] = cache_emb.detach() if hasattr(cache_emb, 'detach') else cache_emb
            self._access_order.append(query)
            self._evict_if_needed()
            return cache_emb

        return None

    def get_numpy(self, query: str) -> Optional[torch.Tensor]:
        """
        Get embedding for a query as CPU tensor.

        Legacy compatibility method - converts to CPU if needed.

        Args:
            query: Query string

        Returns:
            Embedding as CPU tensor, None if not cached and no embedding_fn.
        """
        emb = self.get(query)
        if emb is None:
            return None
        if self._device != "cpu":
            return emb.cpu()
        return emb

    def get_tensor(self, query: str, device: str = None) -> Optional["torch.Tensor"]:
        """
        Get embedding for a query as tensor on specified device.

        Args:
            query: Query string
            device: Target device ("cuda" or "cpu"). If None, uses storage device.

        Returns:
            Embedding as torch.Tensor on device, None if not cached and no embedding_fn.
        """
        emb = self.get(query)
        if emb is None:
            return None

        target_device = device or self._device
        if not str(emb.device).startswith(target_device):
            return emb.to(target_device)
        return emb

    @property
    def device(self) -> str:
        """Return the device embeddings are stored on."""
        return self._device

    def get_batch(self, queries: List[str]) -> Dict[str, torch.Tensor]:
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
                self.stats.generation_hits += 1
                # Track time saved on cache hit
                self.stats.compute_time_saved_sec += self.stats._avg_embed_time_sec
                result[q] = self._cache[q]
            else:
                self.stats.cache_misses += 1
                self.stats.generation_misses += 1
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
        self._access_order.clear()
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
    ) -> torch.Tensor:
        """
        Get embeddings for queries as a single matrix.

        Useful for batch similarity computations.

        Args:
            queries: Ordered list of queries
            as_tensor: If True, return torch.Tensor on GPU if available.

        Returns:
            torch.Tensor of shape (len(queries), embedding_dim)
        """
        # If storing on accelerated device and requesting tensor, stack directly on device
        if self._device != "cpu" and as_tensor:
            gpu_embeddings = []
            for q in queries:
                emb = self._cache.get(q)
                if emb is not None:
                    if isinstance(emb, torch.Tensor):
                        gpu_embeddings.append(emb)
                    else:
                        gpu_embeddings.append(torch.as_tensor(emb, device=self._device, dtype=torch.float32))
                else:
                    # Return zero vector for missing
                    if self._embedding_dim:
                        gpu_embeddings.append(torch.zeros(self._embedding_dim, device=self._device, dtype=torch.float32))
                    else:
                        raise ValueError(f"Cannot get embedding for '{q}' and dimension unknown")
            return torch.stack(gpu_embeddings)

        # Standard path: collect as tensors
        embeddings = []
        for q in queries:
            emb = self.get(q)  # Get as tensor
            if emb is not None:
                embeddings.append(emb)
            else:
                # Return zero vector for missing
                if self._embedding_dim:
                    embeddings.append(torch.zeros(self._embedding_dim, dtype=torch.float32))
                else:
                    raise ValueError(f"Cannot get embedding for '{q}' and dimension unknown")

        matrix = torch.stack(embeddings)

        if as_tensor:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            return matrix.to(device)

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
        return self.get_all_embeddings_matrix(queries, as_tensor=True)

    def finalize_generation(self, generation: int) -> EmbeddingCacheStats:
        """
        Finalize stats for a generation and reset per-generation counters.

        Returns a COPY of current stats with per-generation values before resetting.

        Args:
            generation: The generation number (for logging)

        Returns:
            Copy of EmbeddingCacheStats with per-generation values
        """
        # Create a copy of current stats
        stats_copy = copy.copy(self.stats)

        # Log generation summary
        total = self.stats.generation_hits + self.stats.generation_misses
        hit_rate = self.stats.generation_hit_rate
        time_saved = self.stats.compute_time_saved_sec
        logger.info(
            f"EmbeddingCache gen {generation}: {self.stats.generation_hits}/{total} hits "
            f"({hit_rate:.0%}), saved {time_saved:.1f}s"
        )

        # Reset per-generation counters
        self.stats.reset_generation()

        return stats_copy

    def dump_debug_info(
        self,
        path: Optional[str] = None,
        include_embeddings: bool = False
    ) -> dict:
        """
        Export cache debug information to dict/JSON.

        Args:
            path: Optional path to write JSON file
            include_embeddings: If True, include tensor values as lists (warning: large)

        Returns:
            Dict with cache state information
        """
        debug_info = {
            'cache_size': len(self._cache),
            'embedding_dim': self._embedding_dim,
            'device': self._device,
            'maxsize': self.maxsize,
            'stats': {
                'total_queries': self.stats.total_queries,
                'cache_hits': self.stats.cache_hits,
                'cache_misses': self.stats.cache_misses,
                'hit_rate': self.stats.hit_rate,
                'precompute_time': self.stats.precompute_time,
                'total_embedding_time': self.stats.total_embedding_time,
                'generation_hits': self.stats.generation_hits,
                'generation_misses': self.stats.generation_misses,
                'generation_hit_rate': self.stats.generation_hit_rate,
                'compute_time_saved_sec': self.stats.compute_time_saved_sec,
                'avg_embed_time_sec': self.stats._avg_embed_time_sec,
            },
            'cached_queries': list(self._cache.keys()),
        }

        if include_embeddings:
            embeddings_dict = {}
            for q, emb in self._cache.items():
                if isinstance(emb, torch.Tensor):
                    embeddings_dict[q] = emb.cpu().tolist()
                else:
                    embeddings_dict[q] = list(emb)
            debug_info['embeddings'] = embeddings_dict

        if path is not None:
            with open(path, 'w') as f:
                json.dump(debug_info, f, indent=2)
            logger.info(f"Embedding cache debug info written to {path}")

        return debug_info


class EmbeddingCacheProvider:
    """
    Singleton provider for the global embedding cache.

    Allows sharing the same cache across different parts of the evolution system.

    Thread safety: Uses a class-level lock to protect singleton access,
    preventing races when multiple threads call get_or_create() or clear()
    concurrently.
    """

    _instance: Optional[QueryEmbeddingCache] = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_or_create(
        cls,
        embedding_fn: Callable = None,
        batch_embedding_fn: Callable = None,
        **kwargs
    ) -> QueryEmbeddingCache:
        """Get or create the global embedding cache (thread-safe)."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = QueryEmbeddingCache(
                    embedding_fn=embedding_fn,
                    batch_embedding_fn=batch_embedding_fn,
                    **kwargs
                )
            return cls._instance

    @classmethod
    def get(cls) -> Optional[QueryEmbeddingCache]:
        """Get the global embedding cache if it exists (thread-safe)."""
        with cls._lock:
            return cls._instance

    @classmethod
    def clear(cls):
        """Clear and remove the global cache (thread-safe)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.clear()
            cls._instance = None
