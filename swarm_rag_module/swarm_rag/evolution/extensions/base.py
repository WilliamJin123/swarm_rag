from typing import List, Optional
from ..types.config import EvolutionContext

class EvolutionExtension():
    """
    Base class for extensions (Plugins).
    Override these methods to inject custom logic.
    """
    
    def on_init(self, ctx: EvolutionContext):
        """Called once before the evolution starts."""
        pass

    def on_generation_start(self, ctx: EvolutionContext):
        """Called at the start of every generation."""
        pass

    def on_after_evaluation(self, ctx: EvolutionContext):
        """
        Called after fitness is calculated but BEFORE selection.
        PERFECT FOR: Niching, Fitness Sharing, Species protection.
        """
        pass

    def on_before_breeding(self, ctx: EvolutionContext):
        """
        Called before the selection/crossover/mutation loop.
        PERFECT FOR: Random Immigration (injecting new blood).
        """
        pass

    def on_generation_end(self, ctx: EvolutionContext):
        """
        Called after the new population is created.
        PERFECT FOR: Logging, Checkpointing, or Migration triggers.
        """
        pass