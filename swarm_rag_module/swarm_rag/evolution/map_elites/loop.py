import random
import logging
from typing import List, Optional

from ..types.genome import Genome
from ..types.config import EvolutionContext
from ..execution.strategies import GeneticRegistry
from .archive import MapElitesArchive

logger = logging.getLogger(__name__)


class MapElitesLoop:
    """
    Implements the MAP-Elites main loop logic:
    1. Select parents from Archive.
    2. Apply genetic operators (Mutation/Crossover) to create offspring.

    Enhanced with three-tier architecture support:
    - Tracks parent IDs for journal lineage
    - Supports strategic directive updates
    """

    def __init__(self, context: EvolutionContext):
        self.context = context
        # Access config via dataclass attributes
        self.mutation_fn = GeneticRegistry.get_mutation(context.config.genetic.mutation_strategy)
        self.crossover_fn = GeneticRegistry.get_crossover(context.config.genetic.crossover_strategy)

        # Batch size from MAP-Elites config
        self.batch_size = context.config.map_elites.batch_size

        # Track offspring count for unique IDs
        self._offspring_counter = 0

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
                # Track both parents for crossover
                child._parent_id = p1.id
                child._parent2_id = p2.id
            else:
                # Generate unique child ID
                self._offspring_counter += 1
                child_id = f"g{self.context.generation}_c{self._offspring_counter}"
                child = p1.copy(new_id=child_id)
                # Track parent for lineage
                child._parent_id = p1.id

            # 3. Mutation
            child = self.mutation_fn(child, self.context)

            offspring.append(child)

        return offspring

    def update_strategic_directive(self, archive: MapElitesArchive) -> Optional["StrategicDirective"]:
        """
        Update the strategic directive if needed.

        This is called from the orchestrator to check if the Strategic Oracle
        should be invoked based on generation interval or stagnation.

        Args:
            archive: Current archive for stats

        Returns:
            New StrategicDirective if updated, None otherwise
        """
        from ..execution.llm_strategies import LLMStrategies

        # Check if we should update
        if not LLMStrategies.should_update_oracle(self.context):
            return None

        # Get archive stats and population
        stats = archive.stats()
        population = archive.as_population()

        # Update directive
        directive = LLMStrategies.update_strategic_directive(
            self.context,
            stats,
            population,
        )

        if directive:
            logger.info(
                f"Strategic directive updated: mode={directive.mode.value}, "
                f"focus={directive.focus_component.value}, "
                f"temp={directive.exploration_temperature:.2f}"
            )

        return directive
