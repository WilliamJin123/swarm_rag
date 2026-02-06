"""Fitness calculation, caching, and population ranking strategies."""
from .calculator import (
    FitnessCalculator,
    FitnessConfig,
    FitnessMode,
    MetricConfig,
    FitnessResult,
    create_fitness_calculator,
)
from .cache import FitnessCache, CacheStats, hash_genome
from .cache_stats import CacheStatsProtocol
from .strategies import FitnessStrategy, LexicographicStrategy, ParetoStrategy

__all__ = [
    "FitnessCalculator", "FitnessConfig", "FitnessMode", "MetricConfig",
    "FitnessResult", "create_fitness_calculator",
    "FitnessCache", "CacheStats", "hash_genome",
    "CacheStatsProtocol",
    "FitnessStrategy", "LexicographicStrategy", "ParetoStrategy",
]
