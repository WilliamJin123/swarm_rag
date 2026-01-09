import random
from typing import List
import numpy as np

from ..types.genome import Genome
from ..types.config import EvolutionContext
from ..execution.strategies import GeneticRegistry
from .archive import MapElitesArchive

class MapElitesLoop:
    """
    Implements the MAP-Elites main loop logic:
    1. Select parents from Archive.
    2. Apply genetic operators (Mutation/Crossover) to create offspring.
    """
    def __init__(self, context: EvolutionContext):
        self.context = context
        self.mutation_fn = GeneticRegistry.get_mutation(context.config["mutation_strategy"])
        self.crossover_fn = GeneticRegistry.get_crossover(context.config["crossover_strategy"])
        
        # Batch size is effectively population_size in the config
        self.batch_size = context.config["population_size"]

    def step(self, archive: MapElitesArchive) -> List[Genome]:
        """
        Generates a new batch of offspring from the archive.
        """
        self.context.generation += 1
        
        offspring: List[Genome] = []
        
        # If archive is empty, we can't breed. 
        # (This should be handled by initialization, but safety check)
        if not archive.grid:
            return offspring

        # Generate batch
        while len(offspring) < self.batch_size:
            # 1. Selection (Random Elite)
            p1 = archive.select_random()
            
            # 2. Crossover (Optional)
            if random.random() < self.context.config['crossover_rate']:
                p2 = archive.select_random()
                child = self.crossover_fn(p1, p2, self.context)
            else:
                child = p1.copy()
            
            # 3. Mutation
            child = self.mutation_fn(child, self.context)
            
            offspring.append(child)
            
        return offspring
