"""
Evolution execution module.

Provides evaluation, genetic strategies, and utilities for the evolution engine.
"""
from .evaluator import PopulationEvaluator
from .fitness import FitnessCalculator
from .strategies import GeneticStrategies, GeneticRegistry
from .tracker import ProgressTracker
from .factory import GenomeFactory

# Import llm_strategies to register LLM mutation with GeneticRegistry
from . import llm_strategies  # noqa: F401

# New efficiency-focused modules
from .adaptive_evaluator import (
    AdaptivePopulationEvaluator,
    EvaluationTier,
    EvaluationStats,
    create_dynamic_tiers,
)
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

__all__ = [
    # Original exports
    "PopulationEvaluator",
    "FitnessCalculator",
    "GeneticStrategies",
    "GeneticRegistry",
    "ProgressTracker",
    "GenomeFactory",
    # Adaptive evaluation
    "AdaptivePopulationEvaluator",
    "EvaluationTier",
    "EvaluationStats",
    "create_dynamic_tiers",
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
]
