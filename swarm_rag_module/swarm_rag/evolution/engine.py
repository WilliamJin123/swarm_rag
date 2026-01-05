import os
import random
import pickle
import numpy as np
from typing import List, Any

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

from .extensions.base import EvolutionExtension

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
        config: EvolutionConfigDict = None,
        extensions: List['EvolutionExtension'] = None,
        overwrite_logs: bool = True
    ):
        self.config = config or DEFAULT_EVO_CONFIG

        self.train_queries = train_queries
        self.train_gt = train_ground_truth
        self.val_queries = val_queries
        self.val_gt = val_ground_truth

        self.evo_context = EvolutionContext(
            config=self.config,
            generation=0,
            available_features=list(HeuristicRegistry.all().keys()),
            # Maps "movement" -> ["semantic", "degree", ...]
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
            concurrent_evaluations=self.config["concurrent_evaluations"]
        )
        self.loop = EvolutionLoop(self.evo_context)
        self.tracker = ProgressTracker(
            log_path=self.config["log_path"], 
            plot_path=self.config["plot_path"],
            plot_title=self.config["plot_title"],
            overwrite=overwrite_logs
        )

        self.extensions = extensions or []
        for ext in self.extensions:
            ext.on_init(self.evo_context)

    def create_initial_genomes(self) -> List[Genome]:
        """Boots up the first random population."""
        count = self.config['population_size']
        ranges = self.config['param_ranges']
        max_d = self.config['expr_max_depth']
        
        strat_trees = {}
        for strat_type in ["movement", "ranking", "deposit"]:
            features = self.evo_context.expression_features[strat_type]
            strat_trees[strat_type] = ExpressionEvolution.generate_ramped_half_and_half(
                features=features,
                population_size=count,
                max_depth=max_d
            )

        population = []
        for i in range(count):
            # Randomize Params
            params = DEFAULT_PARAMS.copy()
            for key, (min_v, max_v) in ranges.items():
                if isinstance(min_v, int):
                    params[key] = random.randint(min_v, max_v)
                else:
                    params[key] = random.uniform(min_v, max_v)

            # Assign Trees (Pop from the generated lists)
            strategies = {
                "movement": strat_trees["movement"][i],
                "ranking": strat_trees["ranking"][i],
                "deposit": strat_trees["deposit"][i]
            }

            genome = Genome(
                id=f"gen0_{i}",
                params=params,
                strategies=strategies
            )
            population.append(genome)
            
        return population

    def optimize(self, initial_population: List[Genome] = None) -> Genome:
        if self.evo_context.population:
            population = self.evo_context.population
            print(f"Resuming evolution with existing population of {len(population)}")
        else:
            population = initial_population or self.create_initial_genomes()

        best_genome: Genome = None
        n_gen = self.config["n_generations"]

        start_gen = self.evo_context.generation

        if self.evo_context.population:
            start_gen += 1
        else:
            start_gen = 0

        print(f"Starting evolution: {len(population)} agents, Gens {start_gen} to {n_gen-1}.")

        for gen in range(start_gen, n_gen): 
            self.evo_context.generation = gen      

            # HOOK: Gen Start
            for ext in self.extensions: ext.on_generation_start(self.evo_context)

            # EVALUATE
            self.population_evaluator.evaluate(population)
            # Should default to training queries and gts
            
            # HOOK: Post-Eval (Niching happens here)
            for ext in self.extensions: ext.on_after_evaluation(self.evo_context)

            # STATS & ELITISM
            population.sort(key=lambda g: g.fitness, reverse=True)
            current_best = population[0]
            
            if best_genome is None or current_best.fitness > best_genome.fitness:
                best_genome = current_best.copy() # Important: Copy so it doesn't get mutated later

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
                    queries=self.val_queries, 
                    ground_truth=self.val_gt
                )
                
                val_stats = {
                    "best_quality": val_candidate.fitness.quality_score,
                    "recall": val_candidate.metrics.get("Recall@20", 0.0)
                }
            
            # LOGGING
            train_stats = {
                "best_quality": current_best.fitness.quality_score,
                "avg_quality": np.mean([g.fitness.quality_score for g in population]),
                "best_stability": current_best.fitness.stability_score,
                "best_cost": current_best.fitness.cost_score,
                "best_complexity": current_best.complexity()
            }

            for k, v in current_best.metrics.items():
                train_stats[f"best_metric_{k}"] = v

            self.tracker.log(gen, train_stats, val_stats)
            self.tracker.print_summary(gen)

            # CHECKPOINTING
            if (gen % self.config["checkpoint_frequency"] == 0):
                self.save_checkpoint(population, best_genome, gen)

            # HOOK: Pre-Breed (Random Immigration happens here)
            for ext in self.extensions: ext.on_before_breeding(self.evo_context)

            # BREED (Skip on last gen)
            if gen < n_gen - 1:
                population = self.loop.step(population)

            # HOOK: Gen End
            for ext in self.extensions: ext.on_generation_end(self.evo_context)

        self.save_checkpoint(population, best_genome, n_gen - 1)
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
        train_queries: List[str],
        train_ground_truth: List[List[Any]],
        val_queries: List[str],
        val_ground_truth: List[List[Any]],
        config: EvolutionConfigDict
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
            train_queries=train_queries,
            train_ground_truth=train_ground_truth,
            val_queries=val_queries,
            val_ground_truth=val_ground_truth,
            config=config,
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
            
        print(f"  ✓ State restored. Resuming from Generation {state['generation']}")
        return engine