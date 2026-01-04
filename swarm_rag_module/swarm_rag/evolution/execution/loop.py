import random
from typing import List

import numpy as np
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
        self.selection_fn = GeneticRegistry.get_selection(context.config["selection_strategy"])
        self.crossover_fn = GeneticRegistry.get_crossover(context.config["crossover_strategy"])
        self.mutation_fn = GeneticRegistry.get_mutation(context.config["mutation_strategy"])

        self.schedule_type = context.config.get('mutation_schedule', 'constant') # constant, decay, adaptive
        self.base_rate = context.config['mutation_rate']

    def get_mutation_rate(self, population: List[Genome]) -> float:
        """Calculates the effective mutation rate for this generation."""
        gen = self.context.generation
        total_gens = self.context.config['n_generations']
        
        if self.schedule_type == 'constant':
            return self.base_rate
            
        elif self.schedule_type == 'decay':
            # Linear decay: Start at base_rate, end at 0.05
            progress = gen / max(1, total_gens)
            return max(0.05, self.base_rate * (1.0 - progress))
            
        elif self.schedule_type == 'adaptive':
            # Calculate fitness diversity (variance of quality score)
            scores = [g.fitness.quality_score for g in population]
            variance = np.var(scores) if scores else 0.0
            
            # If variance is low (< 0.001), boosting mutation to escape stagnation
            if variance < 0.001:
                return min(0.5, self.base_rate * 2.0)
            return self.base_rate
            
        return self.base_rate

    def step(self, population: List[Genome]) -> List[Genome]:
        """
        Produces the NEXT generation from the CURRENT one.
        """
        # Update Context
        self.context.population = population
        self.context.generation += 1 

        # Calculate dynamic rate
        effective_rate = self.get_mutation_rate(population)
        # Inject into context so strategies can see it
        self.context.current_mutation_rate = effective_rate

        print(f"  > Gen {self.context.generation} Mutation Rate: {effective_rate:.4f} ({self.schedule_type})")

        # Elitism 
        population.sort(key=lambda g: g.fitness, reverse=True)
        elite_count = int(self.context.config['population_size'] * self.context.config['elite_fraction'])
        offspring = population[:elite_count]
        
        # Breed 
        while len(offspring) < self.context.config["population_size"]:
            # Selection
            p1 = self.selection_fn(self.context)
            p2 = self.selection_fn(self.context)
            
            # Crossover
            if random.random() < self.context.config["crossover_rate"]:
                child = self.crossover_fn(p1, p2, self.context)
            else:
                child = p1.copy()
            
            # Mutation
            child = self.mutation_fn(child, self.context)
            
            # Reset metadata
            child.fitness = None 
            child.metrics = {}
            child.evaluated = False
            
            offspring.append(child)
            
        return offspring