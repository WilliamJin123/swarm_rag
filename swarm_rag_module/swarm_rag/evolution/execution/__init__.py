"""
Evolution execution module.

Provides evaluation, genetic strategies, and utilities for the evolution engine.
"""
from .evaluator import (
    PopulationEvaluator,
    EvaluationTier,
    EvaluationStats,
    DEFAULT_TIERS,
)
from .fitness import FitnessCalculator
from .strategies import GeneticStrategies, GeneticRegistry
from .tracker import ProgressTracker
from .factory import GenomeFactory

# Import llm_strategies to register LLM mutation with GeneticRegistry
from . import llm_strategies  # noqa: F401

# Efficiency-focused modules
from .stratified_sampler import (
    StratifiedQuerySampler,
    StratifiedSample,
    AdaptiveSampler,
    categorize_by_difficulty,
    categorize_by_query_length,
)
from .embedding_cache import (
    QueryEmbeddingCache,
    EmbeddingCacheProvider,
    EmbeddingCacheStats,
)
from .shared_precompute import (
    SharedPrecomputeContext,
    prepare_shared_context,
    get_unique_pool_sizes,
    BatchedRetrievalResults,
)

__all__ = [
    # Core
    "PopulationEvaluator",
    "FitnessCalculator",
    "GeneticStrategies",
    "GeneticRegistry",
    "ProgressTracker",
    "GenomeFactory",
    # Evaluation tiers
    "EvaluationTier",
    "EvaluationStats",
    "DEFAULT_TIERS",
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
]
