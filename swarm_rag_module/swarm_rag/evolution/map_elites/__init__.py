"""
MAP-Elites Quality-Diversity module.

Provides:
- MapElitesArchive: Archive for storing elite genomes by behavioral descriptors
- MapElitesLoop: Breeding loop for generating offspring from archive
- DescriptorCalculator: Behavioral descriptor computation
"""
from .archive import MapElitesArchive, MAPStats
from .loop import MapElitesLoop
from .descriptors import DescriptorCalculator, DescriptorRegistry

__all__ = [
    "MapElitesArchive",
    "MAPStats",
    "MapElitesLoop",
    "DescriptorCalculator",
    "DescriptorRegistry",
]
