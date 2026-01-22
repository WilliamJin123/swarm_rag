"""
Evolution types module.

Provides configuration, genome, and fitness result types.
"""
from .config import (
    EvolutionConfig,
    MapElitesConfig,
    GeneticConfig,
    BoltzmannConfig,
    LLMConfig,
    CreativeModeConfig,
    ResourceConfig,
    CheckpointConfig,
    SwarmParamRanges,
    EvolutionContext,
    DEFAULT_CONFIG,
    # Backwards compatibility
    EvolutionConfigDict,
    DEFAULT_EVO_CONFIG,
)
from .genome import Genome, SwarmParams, DEFAULT_PARAMS
from .expressions import ExpressionNode, ExpressionEvolution
from .fitness_results import FitnessResult

__all__ = [
    # New config dataclasses
    "EvolutionConfig",
    "MapElitesConfig",
    "GeneticConfig",
    "BoltzmannConfig",
    "LLMConfig",
    "CreativeModeConfig",
    "ResourceConfig",
    "CheckpointConfig",
    "SwarmParamRanges",
    "EvolutionContext",
    "DEFAULT_CONFIG",
    # Backwards compatibility
    "EvolutionConfigDict",
    "DEFAULT_EVO_CONFIG",
    # Genome types
    "Genome",
    "SwarmParams",
    "DEFAULT_PARAMS",
    # Expression types
    "ExpressionNode",
    "ExpressionEvolution",
    # Fitness
    "FitnessResult",
]
