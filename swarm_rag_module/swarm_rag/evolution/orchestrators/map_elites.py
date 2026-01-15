"""
MAP-Elites Quality-Diversity orchestrator.
"""
from typing import List, Optional, Any

from tqdm.auto import tqdm

from .base import BaseOrchestrator
from ..types.genome import Genome
from ..types.config import EvolutionContext
from ..map_elites.archive import MapElitesArchive
from ..map_elites.loop import MapElitesLoop
from ..execution.factory import GenomeFactory
from ..execution.evaluator import PopulationEvaluator
from ..execution.tracker import ProgressTracker
from ..execution.fitness_strategies import FitnessStrategy
from ..extensions.base import EvolutionExtension


class MAPElitesOrchestrator(BaseOrchestrator):
    """
    Orchestrator for MAP-Elites Quality-Diversity evolution.

    Uses an archive-based approach with behavioral descriptors:
    1. Initialize population and seed archive
    2. Select parents from archive
    3. Apply crossover/mutation to create offspring
    4. Evaluate offspring
    5. Attempt to add to archive (only if better than current cell occupant)
    6. Repeat

    MAP-Elites maintains diversity by storing elites across a multi-dimensional
    phenotypic grid defined by behavioral descriptors.
    """

    def __init__(
        self,
        context: EvolutionContext,
        evaluator: PopulationEvaluator,
        fitness_strategy: FitnessStrategy,
        tracker: ProgressTracker,
        val_query_ids: List[Any],
        val_ground_truth: List[List[Any]],
        archive: MapElitesArchive,
        me_loop: MapElitesLoop,
        genome_factory: GenomeFactory,
        extensions: List[EvolutionExtension] = None
    ):
        """
        Initialize MAP-Elites orchestrator.

        Args:
            context: Evolution context
            evaluator: Population evaluator
            fitness_strategy: Fitness assignment strategy
            tracker: Progress tracker
            val_query_ids: Validation queries
            val_ground_truth: Validation ground truth
            archive: MAP-Elites archive for storing elites
            me_loop: MAP-Elites breeding loop
            genome_factory: Factory for creating genomes
            extensions: Optional evolution extensions
        """
        super().__init__(
            context=context,
            evaluator=evaluator,
            fitness_strategy=fitness_strategy,
            tracker=tracker,
            val_query_ids=val_query_ids,
            val_ground_truth=val_ground_truth,
            extensions=extensions
        )
        self.archive = archive
        self.me_loop = me_loop
        self.genome_factory = genome_factory

    def optimize(self, initial_population: List[Genome] = None) -> Genome:
        """
        Run MAP-Elites optimization.

        Args:
            initial_population: Optional starting population to seed archive

        Returns:
            Best genome found during evolution
        """
        self.logger = self.setup_logging()

        # Initialize population
        population = initial_population
        if not population:
            self.logger.info(
                f"Generating initial population of {self.config['map_elites_initial_fill']}..."
            )
            population = self.genome_factory.create_population(
                self.config['map_elites_initial_fill']
            )

        best_genome = self.restored_best_genome
        n_gen = self.config["n_generations"]
        start_gen = self.context.generation

        if start_gen > 0:
            self.logger.info(f"Resuming MAP-Elites from Gen {start_gen}")

        # Initial evaluation and archive seeding
        if population:
            to_eval = [g for g in population if not g.evaluated]
            if to_eval:
                self.logger.info(f"Evaluating {len(to_eval)} initial agents...")
                self.evaluator.evaluate(to_eval)

            # Assign fitness for sorting
            self.fitness_strategy.assign_fitness(population, generation=start_gen)

            # Seed archive
            self.logger.info("Seeding archive...")
            for g in population:
                self.archive.add(g)
                if best_genome is None or g.fitness > best_genome.fitness:
                    best_genome = g.copy()

        pbar = tqdm(range(start_gen, n_gen), desc="MAP-Elites", unit="gen", position=0)

        for gen in pbar:
            # Set generation (single source of truth)
            self.context.generation = gen

            # Pre-generation hook
            self.invoke_extension("on_generation_start")

            # BREED: Generate offspring from archive
            # Note: me_loop.step() should NOT increment generation anymore
            offspring = self.me_loop.step(self.archive)

            if not offspring:
                self.logger.warning("Archive empty or no offspring produced. Stopping.")
                break

            # Post-breeding hook
            self.invoke_extension("on_after_evaluation")

            # EVALUATE offspring
            self.evaluator.evaluate(offspring)
            self.fitness_strategy.assign_fitness(offspring, generation=gen)

            # ARCHIVE INSERTION
            added_count = 0
            for child in offspring:
                if self.archive.add(child):
                    added_count += 1

                if best_genome is None or child.fitness > best_genome.fitness:
                    best_genome = child.copy()
                    self.logger.info(
                        f"Gen {gen}: New Global Best! {best_genome.fitness.quality_score:.4f}"
                    )

            # STATS & LOGGING
            stats = self.archive.stats()

            pbar.set_postfix({
                "Cover": f"{stats['coverage']:.2%}",
                "Best": f"{stats['max_fitness']:.4f}",
                "Added": added_count
            })

            log_stats = {
                "best_quality": stats['max_fitness'],
                "avg_quality": stats['qd_score'],  # QD Score as proxy
                "best_stability": 0.0,
                "best_cost": 0.0,
                "best_complexity": 0.0,
                "coverage": stats['coverage'],
                "filled_cells": stats['filled_cells']
            }

            # VALIDATION (periodically)
            val_stats = None
            if best_genome:
                val_stats = self.run_validation(best_genome, gen)
                if val_stats:
                    self.logger.info(
                        f"--> Validation Gen {gen}: Recall {val_stats.get('recall', 0):.4f}"
                    )

            self.tracker.log(gen, log_stats, val_stats)

            # CHECKPOINTING
            if gen % self.config["checkpoint_frequency"] == 0:
                self.save_checkpoint(
                    population=self.archive.as_population(),
                    best_genome=best_genome,
                    generation=gen,
                    extra_state=self._serialize_archive_state()
                )

            # Post-generation hook
            self.invoke_extension("on_generation_end")

        # Cleanup
        pbar.close()
        self.cleanup_logging()
        self.save_checkpoint(
            population=self.archive.as_population(),
            best_genome=best_genome,
            generation=n_gen - 1,
            extra_state=self._serialize_archive_state()
        )
        self.tracker.plot(
            save_path=self.config["plot_path"],
            title=self.config["plot_title"]
        )

        return best_genome

    def _serialize_archive_state(self) -> dict:
        """Serialize archive metadata for checkpoint restoration."""
        return {
            "archive_bins": self.archive.bins,
            "archive_ranges": self.archive.ranges,
            "archive_grid_keys": list(self.archive.grid.keys())
        }
