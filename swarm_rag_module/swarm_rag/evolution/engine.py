import os
import random
import pickle
import numpy as np
from typing import List, Any
from tqdm.auto import tqdm
import logging

from ..interfaces.enums import GeneticKey

from ..core.swarm_retriever import SwarmRetriever
from ..core.heuristics import HeuristicRegistry
from ..eval.metrics import Evaluator

from .types.genome import Genome, DEFAULT_PARAMS
from .types.config import EvolutionContext, EvolutionConfigDict, DEFAULT_EVO_CONFIG
from .types.expressions import ExpressionEvolution

from .execution.evaluator import PopulationEvaluator
from .execution.loop import EvolutionLoop
from .execution.fitness import FitnessCalculator
from .execution.tracker import ProgressTracker
from .execution.factory import GenomeFactory
from .execution.fitness_strategies import (
    FitnessStrategy, 
    LexicographicStrategy, 
    ParetoStrategy, 
    PhasedStrategy
)
from .extensions.base import EvolutionExtension

# MAP-Elites Imports
from .map_elites.archive import MapElitesArchive
from .map_elites.descriptors import DescriptorCalculator
from .map_elites.loop import MapElitesLoop

from ..utils import TqdmLoggingHandler


class EvolutionEngine:
    def __init__(
        self,
        retriever: SwarmRetriever,
        fitness_calculator: FitnessCalculator,
        evaluator: Evaluator,
        train_query_ids: List[Any],
        train_ground_truth: List[List[Any]],
        val_query_ids: List[Any],
        val_ground_truth: List[List[Any]],
        config: EvolutionConfigDict = None,
        genome_factory: 'GenomeFactory' = None,
        extensions: List['EvolutionExtension'] = None,
        overwrite_logs: bool = True
    ):
        logger = logging.getLogger(__name__)
        config = config or DEFAULT_EVO_CONFIG.copy()
        
        self.train_query_ids = train_query_ids
        self.train_gt = train_ground_truth
        self.val_query_ids = val_query_ids
        self.val_gt = val_ground_truth

        if genome_factory is None:
            self.evo_context = EvolutionContext(
                config=config,
                generation=0,
                available_features=list(HeuristicRegistry.all().keys()),
                expression_features={
                    "movement": list(HeuristicRegistry.all_movement().keys()),
                    "ranking": list(HeuristicRegistry.all_ranking().keys()),
                    "deposit": list(HeuristicRegistry.all_deposit().keys()),
                }
            )
            from .execution.factory import GenomeFactory
            self.genome_factory = GenomeFactory(self.evo_context)
        else:
            self.genome_factory = genome_factory
            # override init level config
            self.evo_context = genome_factory.context

        self.config = self.evo_context.config

        self.population_evaluator = PopulationEvaluator(
            retriever=retriever, 
            evaluator=evaluator, 
            fitness_calc=fitness_calculator, 
            queries=train_query_ids, 
            ground_truth=train_ground_truth,
            concurrent_evaluations=self.config["concurrent_evaluations"],
            max_workers_per_retrieval=self.config["max_workers_per_retrieval"]
        )
        
        self.loop = EvolutionLoop(self.evo_context)

        # Initialize LLM Optimizer if using LLM Mutation
        if self.config.get("mutation_strategy") == GeneticKey.LLM_MUTATION:
            from .llm.optimizer import LLMOptimizer
            model_name = self.config.get("llm_model", "zai-glm-4.7")
            provider = self.config.get("llm_provider", "cerebras")
            
            # Attach to context so the strategy function can access it
            self.evo_context.llm_optimizer = LLMOptimizer(
                provider=provider,
                model=model_name,
                env_path=self.config.get("llm_env_path", ".env")
            )
            logger.info("ENABLED: LLM-Guided Mutation Strategy")            
        # Initialize MAP-Elites if enabled
        if self.config.get("map_elites_enabled", False):
            logger.info("ENABLED: MAP-Elites Mode")
            descriptor_calc = DescriptorCalculator(
                dimensions=self.config["map_elites_dims"],
                ranges=self.config["map_elites_ranges"]
            )
            self.map_elites_archive = MapElitesArchive(
                descriptor_calc=descriptor_calc,
                bins=self.config["map_elites_bins"],
                ranges=self.config["map_elites_ranges"]
            )
            self.map_elites_loop = MapElitesLoop(self.evo_context)

        self.tracker = ProgressTracker(
            log_path=self.config["log_path"], 
            plot_path=self.config["plot_path"],
            plot_title=self.config["plot_title"],
            overwrite=overwrite_logs
        )

        # Initialize Fitness Strategy
        strategy_name = self.config.get("fitness_strategy", "lexicographic")
        if strategy_name == "pareto":
            self.fitness_strategy = ParetoStrategy()
        elif strategy_name == "phased":
            switch_gen = self.config.get("phased_switch_gen", 10)
            self.fitness_strategy = PhasedStrategy(switch_gen=switch_gen)
        else:
            self.fitness_strategy = LexicographicStrategy()
            
        logger.info(f"Using Fitness Strategy: {self.fitness_strategy.__class__.__name__}")

        self.extensions = extensions or []
        for ext in self.extensions:
            # Automatically link the factory to the extension if it needs it
            if hasattr(ext, 'genome_factory') and ext.genome_factory is None:
                ext.genome_factory = self.genome_factory.create_population
            ext.on_init(self.evo_context)

    def optimize(self, initial_population: List[Genome] = None) -> Genome:
        if self.config.get("map_elites_enabled", False):
            return self._optimize_map_elites(initial_population)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        if root_logger.hasHandlers(): root_logger.handlers.clear()

        fh = logging.FileHandler(self.config["log_path"].replace(".json", ".log"))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(fh)

        # Console Handler
        th = TqdmLoggingHandler()
        th.setFormatter(logging.Formatter('%(message)s'))
        root_logger.addHandler(th)

        logger = logging.getLogger("evolution")

        if self.evo_context.population:
            population = self.evo_context.population
            print(f"Resuming evolution with existing population of {len(population)}")
        else:
            population = initial_population or self.genome_factory.create_population(
                self.config['population_size']
            )

        best_genome: Genome = getattr(self, 'restored_best_genome', None)
        
        # Fallback: If no best_genome restored, try to find one in the current population
        if best_genome is None and population:
             # Check if any have been evaluated
             evaluated_pop = [g for g in population if g.fitness.quality_score > -float('inf')]
             if evaluated_pop:
                 evaluated_pop.sort(key=lambda g: g.fitness, reverse=True)
                 best_genome = evaluated_pop[0].copy()

        n_gen = self.config["n_generations"]

        start_gen = self.evo_context.generation if self.evo_context.population else 0

        if self.evo_context.population:
            start_gen += 1
        else:
            start_gen = 0

        print(f"Starting evolution: {len(population)} agents, Gens {start_gen} to {n_gen-1}.")

        pbar = tqdm(range(start_gen, n_gen), desc="Evolution", unit="gen", position=0)
        for gen in pbar: 
            self.evo_context.generation = gen      

            # Prehook
            for ext in self.extensions: ext.on_generation_start(self.evo_context)

            # EVALUATE
            self.population_evaluator.evaluate(population)
            # Should default to training queries and gts
            
            # Posthook
            for ext in self.extensions: ext.on_after_evaluation(self.evo_context)

            # ASSIGN FITNESS / RANKING
            self.fitness_strategy.assign_fitness(population, generation=gen)

            # ELITISM
            population.sort(key=lambda g: g.fitness, reverse=True)
            current_best = population[0]
            avg_qual = np.mean([g.fitness.quality_score for g in population])
            if best_genome is None or current_best.fitness > best_genome.fitness:
                logger.info(f"Gen {gen}: New Best Found! Score: {current_best.fitness.quality_score:.4f}")
                best_genome = current_best.copy(new_id=current_best.id)

            pbar.set_postfix({
                "Best": f"{current_best.fitness.quality_score:.4f}",
                "Avg": f"{avg_qual:.4f}",
                "Cost": f"{current_best.fitness.cost_score:.2f}"
            })

            # VALIDATION
            val_stats = None
            if (gen % self.config["validation_frequency"] == 0) or (gen == n_gen - 1):
                print(f"Running Validation on Gen {gen} Best...")
                # Create a copy to not mess up the training metrics/state
                val_candidate = current_best.copy()
                val_candidate.evaluated = False # Force evaluation

                # Evaluate on VALIDATION set
                self.population_evaluator.evaluate(
                    [val_candidate], 
                    queries=self.val_query_ids, 
                    ground_truth=self.val_gt
                )
                
                val_stats = {
                    "best_quality": val_candidate.fitness.quality_score,
                    "recall": val_candidate.metrics.get("Recall@20", 0.0)
                }

                logger.info(f"--> Validation Gen {gen}: Recall {val_stats.get('recall', 0):.4f}")
            
            # LOGGING
            train_stats = {
                "best_quality": current_best.fitness.quality_score,
                "avg_quality": avg_qual,
                "best_stability": current_best.fitness.stability_score,
                "best_cost": current_best.fitness.cost_score,
                "best_complexity": current_best.complexity()
            }

            for k, v in current_best.metrics.items():
                train_stats[f"best_metric_{k}"] = v

            self.tracker.log(gen, train_stats, val_stats)
            self.tracker.print_summary(gen, printer=tqdm.write)

            # CHECKPOINTING
            if (gen % self.config["checkpoint_frequency"] == 0):
                self.save_checkpoint(population, best_genome, gen)

            # Prehook
            for ext in self.extensions: ext.on_before_breeding(self.evo_context)

            # BREED (Skip on last gen)
            if gen < n_gen - 1:
                population = self.loop.step(population)

            # Posthook
            for ext in self.extensions: ext.on_generation_end(self.evo_context)

        # Cleanup
        pbar.close()
        logger = logging.getLogger()
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)

        self.save_checkpoint(population, best_genome, n_gen - 1)
        self.tracker.plot(save_path=self.config["plot_path"], title=self.config["plot_title"])
        
        return best_genome

    def _optimize_map_elites(self, initial_population: List[Genome] = None) -> Genome:
        """
        Specialized optimization loop for MAP-Elites.
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        if root_logger.hasHandlers(): root_logger.handlers.clear()
        
        fh = logging.FileHandler(self.config["log_path"].replace(".json", ".log"))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(fh)
        
        th = TqdmLoggingHandler()
        th.setFormatter(logging.Formatter('%(message)s'))
        root_logger.addHandler(th)
        
        logger = logging.getLogger("evolution")
        
        # 1. Initialization
        # If resuming, initial_population will be populated from checkpoint
        population = initial_population 
        if not population:
            logger.info(f"Generating initial population of {self.config['map_elites_initial_fill']}...")
            population = self.genome_factory.create_population(
                self.config['map_elites_initial_fill']
            )
        
        best_genome = getattr(self, 'restored_best_genome', None)
        n_gen = self.config["n_generations"]
        start_gen = self.evo_context.generation
        
        if start_gen > 0:
            logger.info(f"Resuming MAP-Elites from Gen {start_gen}")

        pbar = tqdm(range(start_gen, n_gen), desc="MAP-Elites", unit="gen", position=0)
        
        # Initial Evaluation & Archive Fill
        if population:
            # Only evaluate if needed (resumed pops are evaluated)
            to_eval = [g for g in population if not g.evaluated]
            if to_eval:
                logger.info(f"Evaluating {len(to_eval)} initial agents...")
                self.population_evaluator.evaluate(to_eval)
            
            # Assign fitness
            self.fitness_strategy.assign_fitness(population, generation=start_gen)
            
            # Fill Archive
            logger.info("Seeding archive...")
            for g in population:
                self.map_elites_archive.add(g)
                if best_genome is None or g.fitness > best_genome.fitness:
                    best_genome = g.copy()

        for gen in pbar:
            self.evo_context.generation = gen
            
            # BREED
            # We generate a batch of offspring from the archive
            offspring = self.map_elites_loop.step(self.map_elites_archive)
            
            if not offspring:
                logger.warning("Archive empty or no offspring produced. Stopping.")
                break

            # EVALUATE
            self.population_evaluator.evaluate(offspring)
            self.fitness_strategy.assign_fitness(offspring, generation=gen)
            
            # ARCHIVE ADDITION
            added_count = 0
            for child in offspring:
                if self.map_elites_archive.add(child):
                    added_count += 1
                
                if best_genome is None or child.fitness > best_genome.fitness:
                    best_genome = child.copy()
                    logger.info(f"Gen {gen}: New Global Best! {best_genome.fitness.quality_score:.4f}")
            
            # STATS
            stats = self.map_elites_archive.stats()
            
            pbar.set_postfix({
                "Cover": f"{stats['coverage']:.2%}",
                "Best": f"{stats['max_fitness']:.4f}",
                "Added": added_count
            })

            # LOGGING
            # Hijacking standard fields for MAP-Elites metrics
            log_stats = {
                "best_quality": stats['max_fitness'],
                "avg_quality": stats['qd_score'], # QD Score
                "best_stability": 0.0,
                "best_cost": 0.0,
                "best_complexity": 0.0,
                "coverage": stats['coverage'],
                "filled_cells": stats['filled_cells']
            }
            
            self.tracker.log(gen, log_stats, None)
            
            # CHECKPOINTING
            if (gen % self.config["checkpoint_frequency"] == 0):
                # We save the archive as the population
                self.save_checkpoint(self.map_elites_archive.as_population(), best_genome, gen)

        # Cleanup
        pbar.close()
        logger = logging.getLogger()
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)
        
        self.save_checkpoint(self.map_elites_archive.as_population(), best_genome, n_gen - 1)
        self.tracker.plot(save_path=self.config["plot_path"], title=self.config["plot_title"])
        
        return best_genome
    
    def save_checkpoint(self, population: List[Genome], best_genome: Genome, generation: int):
        """Saves the full state of evolution to a pickle file."""
        state = {
            "generation": generation,
            "population": population,
            "best_genome": best_genome,
            "random_state": random.getstate(),
            "np_random_state": np.random.get_state(),
            "tracker_history": self.tracker.history
        }
        
        # Save to a temporary file first, then rename to avoid corruption if interrupted
        ckpt_path = self.config["checkpoint_path"]
        base, ext = os.path.splitext(ckpt_path)
        numbered_path = f"{base}_gen_{generation}{ext}"
        with open(numbered_path, "wb") as f:
            pickle.dump(state, f)

        temp_latest = ckpt_path + ".tmp"
        with open(temp_latest, "wb") as f:
            pickle.dump(state, f)

        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        os.rename(temp_latest, ckpt_path)

        print(f"--> Checkpoint saved: {numbered_path} (Latest updated)")

    @classmethod
    def load_checkpoint(
        cls, 
        checkpoint_path: str,
        retriever: Any,
        fitness_calculator: Any,
        evaluator: Any,
        train_query_ids: List[Any],
        train_ground_truth: List[List[Any]],
        val_query_ids: List[Any],
        val_ground_truth: List[List[Any]],
        config: EvolutionConfigDict,
        extensions: List['EvolutionExtension'] = None,
    ) -> 'EvolutionEngine':
        """
        Factory method: Creates a NEW engine instance and restores its state from disk.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        print(f"--> Loading checkpoint from: {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)

        # Initialize a fresh Engine with the provided dependencies
        # This handles the "unpicklable" stuff like DB connections
        engine = cls(
            retriever=retriever,
            fitness_calculator=fitness_calculator,
            evaluator=evaluator,
            train_query_ids=train_query_ids,
            train_ground_truth=train_ground_truth,
            val_query_ids=val_query_ids,
            val_ground_truth=val_ground_truth,
            config=config,
            extensions=extensions,
            overwrite_logs=False
        )

        # Restore Evolutionary State
        engine.evo_context.population = state['population']
        engine.evo_context.generation = state['generation']
        
        # Restore RNG (Crucial for reproducibility)
        if 'random_state' in state:
            random.setstate(state['random_state'])
        if 'np_random_state' in state:
            np.random.set_state(state['np_random_state'])

        # Restore Logs/Tracker
        if 'tracker_history' in state:
            engine.tracker.history = state['tracker_history']
            
        engine.restored_best_genome = state.get('best_genome')
        print(f"  ✓ State restored. Resuming from Generation {state['generation']}")
        return engine