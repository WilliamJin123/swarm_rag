import random
from typing import List
from ..types.genome import Genome
from ..types.config import EvolutionContext
from .strategies import GeneticRegistry

class EvolutionLoop:
    """
    Pure algorithmic logic: Selection, Crossover, Mutation.
    Zero knowledge of retrieval or IO.
    """
    def __init__(self, context: EvolutionContext):
        self.context = context
        # Resolve Strategies ONCE
        self.selection_fn = GeneticRegistry.get_selection(context.config.selection_strategy)
        self.crossover_fn = GeneticRegistry.get_crossover(context.config.crossover_strategy)
        self.mutation_fn = GeneticRegistry.get_mutation(context.config.mutation_strategy)

    def step(self, population: List[Genome]) -> List[Genome]:
        """
        Produces the NEXT generation from the CURRENT one.
        """
        # 1. Update Context
        self.context.population = population
        self.context.generation += 1 # Increment logical generation

        # 2. Elitism (Preserve best)
        population.sort(key=lambda g: g.fitness, reverse=True)
        elite_count = int(self.context.config.population_size * self.context.config.elite_fraction)
        offspring = population[:elite_count] 
        
        # 3. Breed rest
        while len(offspring) < self.context.config.population_size:
            # Selection
            p1 = self.selection_fn(self.context)
            p2 = self.selection_fn(self.context)
            
            # Crossover
            if random.random() < self.context.config.crossover_rate:
                child = self.crossover_fn(p1, p2, self.context)
            else:
                child = p1.copy()
            
            # Mutation
            child = self.mutation_fn(child, self.context)
            
            # Reset metadata
            child.fitness = 0.0 # Reset fitness
            child.metrics = {}
            child.evaluated = False
            
            offspring.append(child)
            
        return offspring