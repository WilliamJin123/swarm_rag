"""
Standard Genetic Algorithm orchestrator.
"""
from typing import List, Optional, Any

import numpy as np
from tqdm.auto import tqdm

from .base import BaseOrchestrator
from ..types.genome import Genome
from ..types.config import EvolutionContext
from ..execution.loop import EvolutionLoop
from ..execution.factory import GenomeFactory
from ..execution.evaluator import PopulationEvaluator
from ..execution.tracker import ProgressTracker
from ..execution.fitness_strategies import FitnessStrategy
from ..extensions.base import EvolutionExtension


class StandardGAOrchestrator(BaseOrchestrator):
    """
    Orchestrator for standard Genetic Algorithm evolution.

    Uses generational replacement with elitism:
    1. Evaluate population
    2. Assign fitness (lexicographic, pareto, or phased)
    3. Select elites
    4. Breed new population (selection -> crossover -> mutation)
    5. Repeat
    """

    def __init__(
        self,
        context: EvolutionContext,
        evaluator: PopulationEvaluator,
        fitness_strategy: FitnessStrategy,
        tracker: ProgressTracker,
        val_query_ids: List[Any],
        val_ground_truth: List[List[Any]],
        loop: EvolutionLoop,
        genome_factory: GenomeFactory,
        extensions: List[EvolutionExtension] = None
    ):
        """
        Initialize StandardGA orchestrator.

        Args:
            context: Evolution context
            evaluator: Population evaluator
            fitness_strategy: Fitness assignment strategy
            tracker: Progress tracker
            val_query_ids: Validation queries
            val_ground_truth: Validation ground truth
            loop: Evolution loop for breeding
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
        self.loop = loop
        self.genome_factory = genome_factory

    def optimize(self, initial_population: List[Genome] = None) -> Genome:
        """
        Run standard GA optimization.

        Args:
            initial_population: Optional starting population

        Returns:
            Best genome found during evolution
        """
        self.logger = self.setup_logging()

        # Population initialization
        if self.context.population:
            population = self.context.population
            print(f"Resuming evolution with existing population of {len(population)}")
        else:
            population = initial_population or self.genome_factory.create_population(
                self.config['population_size']
            )

        # Restore or find best genome
        best_genome: Genome = self.restored_best_genome
        if best_genome is None:
            best_genome = self._find_best_in_population(population)

        n_gen = self.config["n_generations"]

        # Determine starting generation
        if self.context.population:
            start_gen = self.context.generation + 1
        else:
            start_gen = 0

        print(f"Starting evolution: {len(population)} agents, Gens {start_gen} to {n_gen-1}.")

        pbar = tqdm(range(start_gen, n_gen), desc="Evolution", unit="gen", position=0)

        for gen in pbar:
            # Set generation (single source of truth)
            self.context.generation = gen

            # Pre-evaluation hook
            self.invoke_extension("on_generation_start")

            # EVALUATE
            self.evaluator.evaluate(population)

            # Post-evaluation hook
            self.invoke_extension("on_after_evaluation")

            # ASSIGN FITNESS / RANKING
            self.fitness_strategy.assign_fitness(population, generation=gen)

            # ELITISM & BEST TRACKING
            population.sort(key=lambda g: g.fitness, reverse=True)
            current_best = population[0]
            avg_qual = np.mean([g.fitness.quality_score for g in population])

            if best_genome is None or current_best.fitness > best_genome.fitness:
                self.logger.info(
                    f"Gen {gen}: New Best Found! Score: {current_best.fitness.quality_score:.4f}"
                )
                best_genome = current_best.copy(new_id=current_best.id)

            pbar.set_postfix({
                "Best": f"{current_best.fitness.quality_score:.4f}",
                "Avg": f"{avg_qual:.4f}",
                "Cost": f"{current_best.fitness.cost_score:.2f}"
            })

            # VALIDATION
            val_stats = self.run_validation(current_best, gen)
            if val_stats:
                self.logger.info(f"--> Validation Gen {gen}: Recall {val_stats.get('recall', 0):.4f}")

            # LOGGING
            train_stats = self._compute_train_stats(population, current_best)
            self.tracker.log(gen, train_stats, val_stats)
            self.tracker.print_summary(gen, printer=tqdm.write)

            # CHECKPOINTING
            if gen % self.config["checkpoint_frequency"] == 0:
                self.save_checkpoint(population, best_genome, gen)

            # Pre-breeding hook
            self.invoke_extension("on_before_breeding")

            # BREED (Skip on last gen)
            if gen < n_gen - 1:
                population = self.loop.step(population)

            # Post-breeding hook
            self.invoke_extension("on_generation_end")

        # Cleanup
        pbar.close()
        self.cleanup_logging()
        self.save_checkpoint(population, best_genome, n_gen - 1)
        self.tracker.plot(
            save_path=self.config["plot_path"],
            title=self.config["plot_title"]
        )

        return best_genome

    def _compute_train_stats(self, population: List[Genome], best: Genome) -> dict:
        """Compute training statistics for logging."""
        avg_qual = np.mean([g.fitness.quality_score for g in population])

        stats = {
            "best_quality": best.fitness.quality_score,
            "avg_quality": avg_qual,
            "best_stability": best.fitness.stability_score,
            "best_cost": best.fitness.cost_score,
            "best_complexity": best.complexity()
        }

        # Include per-metric details
        for k, v in best.metrics.items():
            stats[f"best_metric_{k}"] = v

        return stats
