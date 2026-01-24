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
    StorageConfig,
    EvolutionContext,
    DEFAULT_CONFIG,
)
from .storage import RunManager

__all__ = [
    "EvolutionEngine",
    "EvolutionConfig",
    "MapElitesConfig",
    "GeneticConfig",
    "LLMConfig",
    "ResourceConfig",
    "StorageConfig",
    "EvolutionContext",
    "DEFAULT_CONFIG",
    "RunManager",
]
