"""Genetic operators: mutations, crossovers, selections, initialization."""
from .registry import GeneticRegistry
from .strategies import GeneticStrategies

# Import operator modules to trigger @GeneticRegistry.register_* decorators
from . import mutations  # noqa: F401
from . import crossovers  # noqa: F401
from . import selections  # noqa: F401
from . import initialization  # noqa: F401

__all__ = [
    "GeneticRegistry",
    "GeneticStrategies",
]
