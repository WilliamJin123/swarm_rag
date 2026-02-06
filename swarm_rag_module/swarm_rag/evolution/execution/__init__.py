"""
Evolution execution module.

Provides evaluation, genetic strategies, and utilities for the evolution engine.
"""
# Core evaluation (root level)
from .evaluator import (
    PopulationEvaluator,
    PopulationEvaluatorBuilder,
    EvaluatorConfig,
    EvaluationStats,
)
from .early_exit import DEFAULT_EARLY_EXIT_THRESHOLD
from .factory import GenomeFactory
from .cache_coordinator import CacheCoordinator

# Fitness
from .fitness import (
    FitnessCalculator,
    FitnessCache,
    CacheStats,
    CacheStatsProtocol,
    hash_genome,
)

# Genetics
from .genetics import GeneticStrategies, GeneticRegistry

# Import genetics.llm_strategies to register LLM operators
from .genetics import llm_strategies  # noqa: F401

# Monitoring
from .monitoring import ProgressTracker

# Optimization
from .optimization import (
    StratifiedQuerySampler,
    StratifiedSample,
    AdaptiveSampler,
    categorize_by_difficulty,
    categorize_by_query_length,
    QueryEmbeddingCache,
    EmbeddingCacheProvider,
    EmbeddingCacheStats,
    SharedPrecomputeContext,
    prepare_shared_context,
    get_unique_pool_sizes,
    BatchedRetrievalResults,
)

__all__ = [
    # Core
    "PopulationEvaluator",
    "PopulationEvaluatorBuilder",
    "EvaluatorConfig",
    "FitnessCalculator",
    "GeneticStrategies",
    "GeneticRegistry",
    "ProgressTracker",
    "GenomeFactory",
    "CacheCoordinator",
    # Early exit evaluation
    "EvaluationStats",
    "DEFAULT_EARLY_EXIT_THRESHOLD",
    # Stratified sampling
    "StratifiedQuerySampler",
    "StratifiedSample",
    "AdaptiveSampler",
    "categorize_by_difficulty",
    "categorize_by_query_length",
    # Embedding cache
    "QueryEmbeddingCache",
    "EmbeddingCacheProvider",
    "EmbeddingCacheStats",
    # Shared pre-computation
    "SharedPrecomputeContext",
    "prepare_shared_context",
    "get_unique_pool_sizes",
    "BatchedRetrievalResults",
    # Fitness caching
    "FitnessCache",
    "CacheStats",
    "CacheStatsProtocol",
    "hash_genome",
]
