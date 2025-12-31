from typing import List, Callable
import random
from .genome import Genome

class GeneticRegistry:
    _selection_registry = {}
    _crossover_registry = {}
    _mutation_registry = {}

    @classmethod
    def register_selection(cls, name: str):
        def decorator(fn):
            cls._selection_registry[name] = fn
            return fn
        return decorator

    @classmethod
    def register_crossover(cls, name: str):
        def decorator(fn):
            cls._crossover_registry[name] = fn
            return fn
        return decorator

    @classmethod
    def register_mutation(cls, name: str):
        def decorator(fn):
            cls._mutation_registry[name] = fn
            return fn
        return decorator

    @classmethod
    def get_selection(cls, name: str) -> Callable:
        return cls._selection_registry[name]

    @classmethod
    def get_crossover(cls, name: str) -> Callable:
        return cls._crossover_registry[name]

    @classmethod
    def get_mutation(cls, name: str) -> Callable:
        return cls._mutation_registry[name]
    
    @classmethod
    def all_selection(cls) -> dict[str, Callable]:
        """
        Return the complete selection registry.
        """
        return cls._selection_registry

    @classmethod
    def all_crossover(cls) -> dict[str, Callable]:
        """
        Return the complete crossover registry.
        """
        return cls._crossover_registry

    @classmethod
    def all_mutation(cls) -> dict[str, Callable]:
        """
        Return the complete mutation registry.
        """
        return cls._mutation_registry
    
class GeneticStrategies:
    """
    Standard library of genetic operators.
    """

    # --- SELECTION ---

    @staticmethod
    @GeneticRegistry.register_selection("tournament")
    def tournament_selection(population: List[Genome], k: int = 3, **kwargs) -> Genome:
        candidates = random.sample(population, k)
        return max(candidates, key=lambda g: g.fitness)

    @staticmethod
    @GeneticRegistry.register_selection("stochastic_universal_sampling")
    def stochastic_universal_sampling(population: List[Genome], **kwargs) -> Genome:
        total_fitness = sum(g.fitness for g in population)
        pick = random.uniform(0, total_fitness)
        current = 0
        for genome in population:
            current += genome.fitness
            if current > pick:
                return genome
        return population[-1]

    # --- CROSSOVER ---

    @staticmethod
    @GeneticRegistry.register_crossover("uniform_parameter_mix")
    def uniform_parameter_mix(parent1: Genome, parent2: Genome, **kwargs) -> Genome:
        child_params = {}
        # Mix simple hyperparams
        for key in parent1.hyperparams:
            child_params[key] = (
                parent1.hyperparams[key] if random.random() > 0.5 
                else parent2.hyperparams[key]
            )
        
        # Mix trees (simple random inheritance)
        child_trees = {}
        for key in parent1.expression_trees:
            child_trees[key] = (
                parent1.expression_trees[key].copy() if random.random() > 0.5 
                else parent2.expression_trees[key].copy()
            )

        return Genome(
            hyperparams=child_params, 
            expression_trees=child_trees,
            config_schema=parent1.config_schema
        )

