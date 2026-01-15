"""
MAP-Elites descriptor system.

Provides pluggable behavioral descriptors for quality-diversity optimization.
"""
from typing import List, Tuple

from .base import Descriptor, GenotypicDescriptor, PhenotypicDescriptor
from .registry import DescriptorRegistry, FlexibleDescriptorCalculator

# Import built-in descriptors to trigger registration
from . import builtin  # noqa: F401

# Re-export for convenience
from .builtin import (
    ComplexityDescriptor,
    NAgentsDescriptor,
    CostDescriptor,
    QualityDescriptor,
    RecallDescriptor,
    AggressivenessDescriptor,
    LatencyDescriptor,
    StepsDescriptor,
    DecayDescriptor
)


class DescriptorCalculator:
    """
    Calculates behavioral descriptors for a Genome.

    Used to place Genomes into the MAP-Elites archive.

    This is a backwards-compatible wrapper around the new
    FlexibleDescriptorCalculator and DescriptorRegistry.
    """

    def __init__(self, dimensions: List[str], ranges: List[Tuple[float, float]]):
        """
        Initialize calculator.

        Args:
            dimensions: List of descriptor names (e.g., ["complexity", "n_agents"])
            ranges: (min, max) for each dimension
        """
        self.dimensions = dimensions
        self.ranges = ranges

        # Create calculator using new registry
        self._calculator = DescriptorRegistry.create_calculator(dimensions, ranges)

    def get_descriptor(self, genome) -> Tuple[float, ...]:
        """
        Returns a tuple of descriptor values corresponding to configured dimensions.

        Args:
            genome: Genome to evaluate

        Returns:
            Tuple of descriptor values

        Example:
            (25.0, 0.85) for ["complexity", "cost"]
        """
        return self._calculator.get_descriptor(genome)

    @classmethod
    def available_dimensions(cls) -> List[str]:
        """Return list of available dimension names."""
        return DescriptorRegistry.available()


__all__ = [
    # Main calculator (backwards compatible)
    "DescriptorCalculator",
    # Base classes
    "Descriptor",
    "GenotypicDescriptor",
    "PhenotypicDescriptor",
    # Registry
    "DescriptorRegistry",
    "FlexibleDescriptorCalculator",
    # Built-in descriptors
    "ComplexityDescriptor",
    "NAgentsDescriptor",
    "CostDescriptor",
    "QualityDescriptor",
    "RecallDescriptor",
    "AggressivenessDescriptor",
    "LatencyDescriptor",
    "StepsDescriptor",
    "DecayDescriptor"
]
