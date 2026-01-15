"""
SwarmRAG Evolution Module

MAP-Elites based evolutionary optimization for retrieval strategies.
"""
from .engine import EvolutionEngine
from .types.config import (
    EvolutionConfig,
    MapElitesConfig,
    GeneticConfig,
    LLMConfig,
    ResourceConfig,
    CheckpointConfig,
    EvolutionContext,
    DEFAULT_CONFIG,
)

__all__ = [
    "EvolutionEngine",
    "EvolutionConfig",
    "MapElitesConfig",
    "GeneticConfig",
    "LLMConfig",
    "ResourceConfig",
    "CheckpointConfig",
    "EvolutionContext",
    "DEFAULT_CONFIG",
]
