import random
from typing import Callable, List

from ..types.genome import Genome
from .base import EvolutionExtension


class RandomImmigrationExtension(EvolutionExtension):
    def __init__(
        self, 
        rate: float = 0.05, 
        genome_factory: Callable[..., List[Genome]] = None):
        
        self.rate = rate
        self.genome_factory = genome_factory

    def on_before_breeding(self, ctx):
        """
        Replace the bottom X% of the population with fresh random genomes.
        """
        pop_size = len(ctx.population)
        n_immigrants = int(pop_size * self.rate)
        
        if n_immigrants == 0: return

        # Sort: Best -> Worst
        ctx.population.sort(key=lambda g: g.fitness, reverse=True)
        
        new_blood = self.genome_factory(n_immigrants) 
        ctx.population[-n_immigrants:] = new_blood
        
        print(f"  [Extension] Immigrated {n_immigrants} new random genomes.")