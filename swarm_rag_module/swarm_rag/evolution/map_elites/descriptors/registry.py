"""
Descriptor registry for pluggable behavioral descriptors.
"""
from typing import Dict, Type, List, Tuple, Callable
import logging

from .base import Descriptor
from ...types.genome import Genome

logger = logging.getLogger(__name__)


class DescriptorRegistry:
    """
    Registry for behavioral descriptors.

    Allows registration of custom descriptors and lookup by name.
    """

    _descriptors: Dict[str, Type[Descriptor]] = {}

    @classmethod
    def register(cls, descriptor_class: Type[Descriptor] = None):
        """
        Register a descriptor class.

        Can be used as a decorator:
            @DescriptorRegistry.register
            class MyDescriptor(Descriptor):
                name = "my_descriptor"
                ...

        Or called directly:
            DescriptorRegistry.register(MyDescriptor)

        Args:
            descriptor_class: Descriptor class to register

        Returns:
            The registered class (for decorator use)
        """
        def wrapper(klass: Type[Descriptor]) -> Type[Descriptor]:
            # Instantiate temporarily to get the name
            instance = klass()
            name = instance.name
            cls._descriptors[name] = klass
            logger.debug(f"Registered descriptor: {name}")
            return klass

        if descriptor_class is not None:
            return wrapper(descriptor_class)
        return wrapper

    @classmethod
    def get(cls, name: str) -> Type[Descriptor]:
        """
        Get a descriptor class by name.

        Args:
            name: Descriptor name

        Returns:
            Descriptor class

        Raises:
            KeyError: If descriptor not found
        """
        if name not in cls._descriptors:
            available = list(cls._descriptors.keys())
            raise KeyError(
                f"Unknown descriptor: '{name}'. Available: {available}"
            )
        return cls._descriptors[name]

    @classmethod
    def all(cls) -> Dict[str, Type[Descriptor]]:
        """Return all registered descriptors."""
        return cls._descriptors.copy()

    @classmethod
    def available(cls) -> List[str]:
        """Return list of available descriptor names."""
        return list(cls._descriptors.keys())

    @classmethod
    def create_calculator(
        cls,
        dimensions: List[str],
        ranges: List[Tuple[float, float]]
    ) -> 'FlexibleDescriptorCalculator':
        """
        Factory method to create a calculator from dimension names.

        Args:
            dimensions: List of descriptor names
            ranges: (min, max) ranges for each dimension

        Returns:
            FlexibleDescriptorCalculator instance
        """
        descriptors = [cls.get(dim)() for dim in dimensions]
        return FlexibleDescriptorCalculator(descriptors, ranges)


class FlexibleDescriptorCalculator:
    """
    Calculates behavioral descriptors using registered descriptor classes.

    Drop-in replacement for the legacy DescriptorCalculator.
    """

    def __init__(
        self,
        descriptors: List[Descriptor],
        ranges: List[Tuple[float, float]]
    ):
        """
        Initialize calculator.

        Args:
            descriptors: List of Descriptor instances
            ranges: (min, max) for each dimension
        """
        if len(descriptors) != len(ranges):
            raise ValueError(
                f"Number of descriptors ({len(descriptors)}) must match "
                f"number of ranges ({len(ranges)})"
            )
        self.descriptors = descriptors
        self.ranges = ranges
        self.dimensions = [d.name for d in descriptors]

    def get_descriptor(self, genome: Genome) -> Tuple[float, ...]:
        """
        Calculate all descriptor values for a genome.

        Args:
            genome: Genome to evaluate

        Returns:
            Tuple of descriptor values
        """
        return tuple(d.calculate(genome) for d in self.descriptors)
