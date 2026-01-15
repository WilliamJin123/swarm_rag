"""
Evolution orchestrators module.

Provides clean separation between different evolutionary algorithms:
- StandardGAOrchestrator: Traditional genetic algorithm with generational replacement
- MAPElitesOrchestrator: Quality-Diversity optimization with archive-based breeding
"""
from .base import BaseOrchestrator
from .standard_ga import StandardGAOrchestrator
from .map_elites import MAPElitesOrchestrator

__all__ = [
    "BaseOrchestrator",
    "StandardGAOrchestrator",
    "MAPElitesOrchestrator"
]
