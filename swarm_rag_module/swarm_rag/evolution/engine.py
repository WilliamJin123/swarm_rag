import os
import time
import random
import pickle
import numpy as np
from typing import List, Any

from ..core.swarm_retriever import SwarmRetriever
from ..core.heuristics import HeuristicRegistry
from ..eval.metrics import Evaluator

from .types.genome import Genome
from .types.config import EvolutionConfig, EvolutionContext
from .types.expressions import ExpressionEvolution

from .execution.evaluator import PopulationEvaluator
from .execution.loop import EvolutionLoop
from .execution.fitness import FitnessCalculator
from .execution.tracker import ProgressTracker
class EvolutionEngine:
    def __init__(
        self,
        retriever: SwarmRetriever,
        fitness_calculator: FitnessCalculator,
        evaluator: Evaluator,
        train_queries: List[str],
        train_ground_truth: List[List[Any]],
        val_queries: List[str],
        val_ground_truth: List[List[Any]],
        config: EvolutionConfig = None,
    ):
        self.config = config or EvolutionConfig()

        # Data
        self.train_queries = train_queries
        self.train_gt = train_ground_truth
        self.val_queries = val_queries
        self.val_gt = val_ground_truth

        # Components
        self.evo_context = EvolutionContext(
            config=self.config,
            generation=0,
            available_features=list(HeuristicRegistry.all().keys()),
            expression_features={
                "movement": list(HeuristicRegistry.all_movement().keys()),
                "ranking": list(HeuristicRegistry.all_ranking().keys()),
                "deposit": list(HeuristicRegistry.all_deposit().keys()),
            }
        )
        self.population_evaluator = PopulationEvaluator(
            retriever=retriever, 
            evaluator=evaluator, 
            fitness_calc=fitness_calculator, 
            queries=train_queries, 
            ground_truth=train_ground_truth,
            global_max_threads=self.config.global_max_threads,
            concurrent_evaluations=self.config.concurrent_evaluations
        )
        self.loop = EvolutionLoop(self.evo_context)

        # Tracking
        self.tracker = ProgressTracker(log_path=self.config.log_file)

    def create_initial_genomes(self) -> List[Genome]:
        """Boots up the first random population."""
        count = self.config.population_size
        cfg = self.config
        
        movement_exprs = ExpressionEvolution.generate_ramped_half_and_half(
            features=self.evo_context.expression_features["movement"],
            population_size=count,
            max_depth=cfg.expr_max_depth
        )
        ranking_exprs = ExpressionEvolution.generate_ramped_half_and_half(
            features=self.evo_context.expression_features["ranking"],
            population_size=count,
            max_depth=cfg.expr_max_depth
        )
        deposit_exprs = ExpressionEvolution.generate_ramped_half_and_half(
            features=self.evo_context.expression_features["deposit"],
            population_size=count,
            max_depth=cfg.expr_max_depth
        )

        # 2. Assemble Genomes
        population = []
        for i in range(count):
            genome = Genome(
                # Hyperparameters (Uniform Random within Config Ranges)
                n_agents=random.randint(cfg.n_agents_min, cfg.n_agents_max),
                steps=random.randint(cfg.steps_min, cfg.steps_max),
                decay=random.uniform(cfg.decay_min, cfg.decay_max),
                initial_pool_size=random.randint(cfg.initial_pool_size_min, cfg.initial_pool_size_max),
                start_subset=random.randint(cfg.start_subset_min, cfg.start_subset_max),
                
                # Assign the pre-generated trees
                movement_expr=movement_exprs[i],
                ranking_expr=ranking_exprs[i],
                deposit_expr=deposit_exprs[i],
            )
            population.append(genome)
            
        return population

    def optimize(self, initial_population: List[Genome] = None) -> Genome:
        population = initial_population or self.create_initial_genomes()
        best_genome: Genome = None

        print(f"Starting evolution: {len(population)} agents, {self.config.n_generations} gens.")
        print(f"Global Thread Cap: {self.config.global_max_threads}")

        for gen in range(self.config.n_generations):
            t0 = time.time()
            
            # EVALUATE
            self.population_evaluator.evaluate(population)
            # Should default to training queries and gts
            
            # STATS & ELITISM
            population.sort(key=lambda g: g.fitness, reverse=True)
            current_best = population[0]
            
            if best_genome is None or current_best.fitness > best_genome.fitness:
                best_genome = current_best.copy() # Important: Copy so it doesn't get mutated later

            # VALIDATION
            val_stats = None
            if (gen % self.config.validation_frequency == 0) or (gen == self.config.n_generations - 1):
                print(f"Running Validation on Gen {gen} Best...")
                # Create a copy to not mess up the training metrics/state
                val_candidate = current_best.copy()
                val_candidate.evaluated = False # Force evaluation

                # Evaluate on VALIDATION set
                self.population_evaluator.evaluate(
                    [val_candidate], 
                    queries=self.val_queries, 
                    ground_truth=self.val_gt
                )
                
                val_stats = {
                    "best_fitness": val_candidate.fitness, # This is now the VAL fitness
                    "recall": val_candidate.metrics.get("Recall", 0.0) # Example metric
                }
            
            # LOGGING
            train_stats = {
                "best_fitness": current_best.fitness,
                "avg_fitness": np.mean([g.fitness for g in population]),
                "best_complexity": current_best.complexity()
            }  
            self.tracker.log(gen, train_stats, val_stats)
            self.tracker.print_summary(gen)

            # CHECKPOINTING
            if (gen % self.config.checkpoint_frequency == 0):
                self.save_checkpoint(population, best_genome, gen)

            # BREED (Skip on last gen)
            if gen < self.config.n_generations - 1:
                population = self.loop.step(population)

        self.save_checkpoint(population, best_genome, self.config.n_generations - 1)
        self.tracker.plot(save_path=self.config.plot_file)
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
        temp_path = self.config.checkpoint_path + ".tmp"
        with open(temp_path, "wb") as f:
            pickle.dump(state, f)
        os.replace(temp_path, self.config.checkpoint_path)
        print(f"--> Checkpoint saved to {self.config.checkpoint_path} (Gen {generation})")

    def load_checkpoint(self) -> tuple[int, List[Genome], Genome]:
        """Loads state from pickle if it exists."""
        if not os.path.exists(self.config.checkpoint_path):
            return 0, None, None

        print(f"--> Resuming from checkpoint: {self.config.checkpoint_path}")
        with open(self.config.checkpoint_path, "rb") as f:
            state = pickle.load(f)

        # Restore RNG states
        random.setstate(state["random_state"])
        np.random.set_state(state["np_random_state"])
        
        # Restore Tracker History (so plots don't start from empty)
        if "tracker_history" in state:
            self.tracker.history = state["tracker_history"]

        return state["generation"], state["population"], state["best_genome"]