"""
Base classes for behavioral descriptors.
"""
from abc import ABC, abstractmethod

from ...types.genome import Genome


class Descriptor(ABC):
    """
    Abstract base class for behavioral descriptors.

    Descriptors define dimensions in the MAP-Elites phenotypic space.
    Each descriptor extracts a single numeric value from a genome.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name for this descriptor.

        Used for registration and configuration.
        """
        pass

    @abstractmethod
    def calculate(self, genome: Genome) -> float:
        """
        Calculate the descriptor value for a genome.

        Args:
            genome: Genome to evaluate

        Returns:
            Numeric descriptor value
        """
        pass


class GenotypicDescriptor(Descriptor):
    """
    Descriptor based on genome structure (not performance).

    Genotypic descriptors can be calculated without evaluating the genome.
    Examples: complexity, n_agents, parameter values
    """
    pass


class PhenotypicDescriptor(Descriptor):
    """
    Descriptor based on genome behavior/performance.

    Phenotypic descriptors require the genome to be evaluated first.
    Examples: cost, latency, recall, quality
    """
    pass
