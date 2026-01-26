import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

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

        Uses parallel mutation when parallel_mutation_workers > 1 and LLM is enabled,
        otherwise falls back to sequential processing for efficiency.

        Note: Generation counter is managed by the orchestrator, not here.
        """
        # If archive is empty, we can't breed.
        if not archive.grid:
            return []

        # Check if parallel mutation is beneficial
        parallel_workers = self.context.config.genetic.parallel_mutation_workers
        llm_enabled = self.context.config.llm.enabled

        # Only use parallel mode when LLM is enabled (mutations are I/O-bound)
        if parallel_workers > 1 and llm_enabled:
            return self._step_parallel(archive, parallel_workers)
        else:
            return self._step_sequential(archive)

    def _step_sequential(self, archive: MapElitesArchive) -> List[Genome]:
        """Sequential offspring generation (original implementation)."""
        offspring: List[Genome] = []
        crossover_rate = self.context.config.genetic.crossover_rate

        while len(offspring) < self.batch_size:
            # 1. Selection (Random Elite)
            p1 = archive.select_random()

            # 2. Crossover (Optional)
            if random.random() < crossover_rate:
                p2 = archive.select_random()
                child = self.crossover_fn(p1, p2, self.context)
                child._parent_id = p1.id
                child._parent2_id = p2.id
            else:
                self._offspring_counter += 1
                child_id = f"g{self.context.generation}_c{self._offspring_counter}"
                child = p1.copy(new_id=child_id)
                child._parent_id = p1.id

            # 3. Mutation
            child = self.mutation_fn(child, self.context)
            offspring.append(child)

        return offspring

    def _step_parallel(self, archive: MapElitesArchive, max_workers: int) -> List[Genome]:
        """
        Parallel offspring generation using ThreadPoolExecutor.

        Beneficial when LLM mutations are enabled since they are I/O-bound.
        Creates children in parallel batches to speed up mutation phase.
        """
        crossover_rate = self.context.config.genetic.crossover_rate
        offspring: List[Genome] = []

        # Pre-create children without mutation (fast, single-threaded)
        children_to_mutate: List[Tuple[int, Genome]] = []

        for i in range(self.batch_size):
            p1 = archive.select_random()

            if random.random() < crossover_rate:
                p2 = archive.select_random()
                child = self.crossover_fn(p1, p2, self.context)
                child._parent_id = p1.id
                child._parent2_id = p2.id
            else:
                self._offspring_counter += 1
                child_id = f"g{self.context.generation}_c{self._offspring_counter}"
                child = p1.copy(new_id=child_id)
                child._parent_id = p1.id

            children_to_mutate.append((i, child))

        # Mutate in parallel (I/O-bound with LLM calls)
        def mutate_child(idx_child: Tuple[int, Genome]) -> Tuple[int, Genome]:
            idx, child = idx_child
            mutated = self.mutation_fn(child, self.context)
            return idx, mutated

        results = [None] * self.batch_size

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(mutate_child, item): item[0]
                for item in children_to_mutate
            }

            for future in as_completed(futures):
                try:
                    idx, mutated_child = future.result()
                    results[idx] = mutated_child
                except Exception as e:
                    logger.error(f"Parallel mutation failed: {e}")
                    # Fallback: use original child without mutation
                    idx = futures[future]
                    results[idx] = children_to_mutate[idx][1]

        # Filter out any None results
        offspring = [r for r in results if r is not None]

        logger.debug(f"Parallel mutation complete: {len(offspring)} offspring created")
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
