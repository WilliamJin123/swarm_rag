"""Performance optimization modules for evaluation."""
from .embedding_cache import QueryEmbeddingCache, EmbeddingCacheProvider, EmbeddingCacheStats
from .shared_precompute import (
    SharedPrecomputeContext,
    prepare_shared_context,
    get_unique_pool_sizes,
    BatchedRetrievalResults,
)
from .stratified_sampler import (
    StratifiedQuerySampler,
    StratifiedSample,
    AdaptiveSampler,
    categorize_by_difficulty,
    categorize_by_query_length,
)

__all__ = [
    "QueryEmbeddingCache", "EmbeddingCacheProvider", "EmbeddingCacheStats",
    "SharedPrecomputeContext", "prepare_shared_context", "get_unique_pool_sizes",
    "BatchedRetrievalResults",
    "StratifiedQuerySampler", "StratifiedSample", "AdaptiveSampler",
    "categorize_by_difficulty", "categorize_by_query_length",
]
