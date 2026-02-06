"""
Genetic operator registry.

Provides GeneticRegistry, the central registration mechanism for all
genetic operators (selection, crossover, mutation, creation). Extracted
into its own module to avoid circular imports between the category
strategy files and the backward-compatible strategies.py facade.
"""
from typing import Callable, Union

from ....interfaces.registry import (
    _MutationRegistry,
    _CrossoverRegistry,
    _SelectionRegistry,
    _CreationRegistry,
)
from ....interfaces.enums import GeneticKey


class GeneticRegistry:
    selection = _SelectionRegistry
    crossover = _CrossoverRegistry
    mutation  = _MutationRegistry
    creation  = _CreationRegistry

    @classmethod
    def register_selection(cls, name: "GeneticKey"):
        return cls.selection.register(name)

    @classmethod
    def register_crossover(cls, name: "GeneticKey"):
        return cls.crossover.register(name)

    @classmethod
    def register_mutation(cls, name: "GeneticKey"):
        return cls.mutation.register(name)

    @classmethod
    def register_creation(cls, name: "GeneticKey"):
        return cls.creation.register(name)

    @classmethod
    def get_selection(cls, name: Union["GeneticKey", str]) -> Callable:
        return cls.selection.get(name)

    @classmethod
    def get_crossover(cls, name: Union["GeneticKey", str]) -> Callable:
        return cls.crossover.get(name)

    @classmethod
    def get_mutation(cls, name: Union["GeneticKey", str]) -> Callable:
        return cls.mutation.get(name)

    @classmethod
    def get_creation(cls, name: Union["GeneticKey", str]) -> Callable:
        return cls.creation.get(name)

    @classmethod
    def get(cls, name: Union["GeneticKey", str]) -> Callable:
        """
        Search **all** genetic registries for name
        """
        try: return cls.selection.get(name)
        except KeyError: pass
        try: return cls.crossover.get(name)
        except KeyError: pass
        try: return cls.mutation.get(name)
        except KeyError: pass
        try: return cls.creation.get(name)
        except KeyError: raise KeyError(f"Genetic heuristic '{name}' is not registered.") from None

    @classmethod
    def all_selection(cls):
        return cls.selection.all()

    @classmethod
    def all_crossover(cls):
        return cls.crossover.all()

    @classmethod
    def all_mutation(cls):
        return cls.mutation.all()

    @classmethod
    def all_creation(cls):
        return cls.creation.all()

    @classmethod
    def all(cls):
        return {
            **cls.selection.all(),
            **cls.crossover.all(),
            **cls.mutation.all(),
            **cls.creation.all(),
        }
