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
        # Access config via dataclass attributes
        self.mutation_fn = GeneticRegistry.get_mutation(context.config.genetic.mutation_strategy)
        self.crossover_fn = GeneticRegistry.get_crossover(context.config.genetic.crossover_strategy)

        # Batch size from MAP-Elites config
        self.batch_size = context.config.map_elites.batch_size

    def step(self, archive: MapElitesArchive) -> List[Genome]:
        """
        Generates a new batch of offspring from the archive.

        Note: Generation counter is managed by the orchestrator, not here.
        """
        offspring: List[Genome] = []

        # If archive is empty, we can't breed.
        # (This should be handled by initialization, but safety check)
        if not archive.grid:
            return offspring

        # Get crossover rate from config
        crossover_rate = self.context.config.genetic.crossover_rate

        # Generate batch
        while len(offspring) < self.batch_size:
            # 1. Selection (Random Elite)
            p1 = archive.select_random()

            # 2. Crossover (Optional)
            if random.random() < crossover_rate:
                p2 = archive.select_random()
                child = self.crossover_fn(p1, p2, self.context)
            else:
                child = p1.copy()

            # 3. Mutation
            child = self.mutation_fn(child, self.context)

            offspring.append(child)

        return offspring
