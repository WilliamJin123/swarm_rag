"""
MAP-Elites descriptor system.

Provides pluggable behavioral descriptors for quality-diversity optimization.
"""
from .base import Descriptor, GenotypicDescriptor, PhenotypicDescriptor
from .registry import DescriptorRegistry, FlexibleDescriptorCalculator

# Import built-in descriptors to trigger registration
from . import builtin  # noqa: F401

# Re-export for convenience
from .builtin import (
    ComplexityDescriptor,
    NAgentsDescriptor,
    StabilityDescriptor,
    QualityDescriptor,
    RecallDescriptor,
    AggressivenessDescriptor,
    LatencyDescriptor,
    StepsDescriptor,
    DecayDescriptor
)

# Alias for backwards compatibility
DescriptorCalculator = FlexibleDescriptorCalculator


__all__ = [
    # Main calculator
    "DescriptorCalculator",
    "FlexibleDescriptorCalculator",
    # Base classes
    "Descriptor",
    "GenotypicDescriptor",
    "PhenotypicDescriptor",
    # Registry
    "DescriptorRegistry",
    # Built-in descriptors
    "ComplexityDescriptor",
    "NAgentsDescriptor",
    "StabilityDescriptor",
    "QualityDescriptor",
    "RecallDescriptor",
    "AggressivenessDescriptor",
    "LatencyDescriptor",
    "StepsDescriptor",
    "DecayDescriptor"
]
