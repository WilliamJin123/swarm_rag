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

    def get_global_multiplier(self, population: List[Genome]) -> float:
        """
        Calculates a SCALING FACTOR for mutation rates.
        1.0 = Use genome's native rate.
        < 1.0 = Dampen mutation (late stage cooling).
        > 1.0 = Boost mutation (break stagnation).
        """
        gen = self.context.generation
        total_gens = self.context.config['n_generations']
        
        if self.schedule_type == 'constant':
            return 1.0
            
        elif self.schedule_type == 'decay':
            # Linear decay: Start at 1.0, end at 0.1 (10% of native rate)
            progress = gen / max(1, total_gens)
            return max(0.1, 1.0 - (0.9 * progress))
            
        elif self.schedule_type == 'adaptive':
            # Calculate fitness diversity (variance of quality score)
            scores = [g.fitness.quality_score for g in population]
            variance = np.var(scores) if scores else 0.0
            
            # If variance is dangerously low, DOUBLE the effective mutation
            # to kick the population out of the local optimum.
            if variance < 0.001:
                return 2.0
            return 1.0
            
        return 1.0

    def step(self, population: List[Genome]) -> List[Genome]:
        """
        Produces the NEXT generation from the CURRENT one.
        """
        # Update Context
        self.context.population = population
        self.context.generation += 1 
        current_gen_idx = self.context.generation

        # Calculate dynamic rate
        global_multiplier = self.get_global_multiplier(population)
        self.context.global_mutation_multiplier = global_multiplier

        print(f"  > Gen {self.context.generation} Global Multiplier: {global_multiplier:.2f}x ({self.schedule_type})")

        # Elitism 
        population.sort(key=lambda g: g.fitness, reverse=True)
        elite_count = int(self.context.config['population_size'] * self.context.config['elite_fraction'])
        offspring = population[:elite_count]
        
        needed = self.context.config['population_size'] - len(offspring)

        parents: List[Genome] = self.selection_fn(self.context, k=needed * 2)

        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents): break
            
            p1 = parents[i]
            p2 = parents[i+1]
            
            # Crossover (Standard)
            if random.random() < self.context.config['crossover_rate']:
                child = self.crossover_fn(p1, p2, self.context)
            else:
                child = p1.copy()
            
            # Mutation (Standard)
            child = self.mutation_fn(child, self.context)
            
            # ... (ID assignment & Reset) ...
            offspring.append(child)
            
        return offspring