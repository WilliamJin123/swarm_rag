import random
from .base import EvolutionExtension


class RandomImmigrationExtension(EvolutionExtension):
    def __init__(self, rate: float = 0.05, engine_ref=None):
        self.rate = rate
        self.engine = engine_ref

    def on_before_breeding(self, ctx):
        """
        Replace the bottom X% of the population with fresh random genomes.
        """
        pop_size = len(ctx.population)
        n_immigrants = int(pop_size * self.rate)
        
        if n_immigrants == 0: return

        # Sort: Best -> Worst
        ctx.population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Generate fresh randoms (using the engine's helper if available, or manual)
        # Assuming you expose a helper in engine or context to create randoms
        new_blood = self.engine.create_initial_genomes()[:n_immigrants] 
        
        # Replace the worst
        # (The loop.step() uses the whole population, so we modify it in-place)
        ctx.population[-n_immigrants:] = new_blood
        
        print(f"  [Extension] Immigrated {n_immigrants} new random genomes.")