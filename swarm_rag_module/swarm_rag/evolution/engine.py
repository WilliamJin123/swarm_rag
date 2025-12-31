import time
import random
import numpy as np
from typing import List, Any

from ..core.swarm_retriever import SwarmRetriever
from ..eval.metrics import Evaluator
from .genome import Genome
from .fitness import FitnessCalculator
from .context import EvolutionConfig, EvolutionContext
from swarm_rag.evolution.expressions import ExpressionEvolution
from swarm_rag.core.heuristics import HeuristicRegistry

# New Components
from .population_eval import PopulationEvaluator
from .loop import EvolutionLoop

class EvolutionEngine:
    def __init__(
        self,
        retriever: SwarmRetriever,
        fitness_calculator: FitnessCalculator,
        evaluator: Evaluator,
        queries: List[str],
        ground_truth: List[List[Any]],
        config: EvolutionConfig = None,
        max_workers: int = 16
    ):
        self.config = config or EvolutionConfig()
        self.max_workers = max_workers

        # 1. Setup Context
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
        
        # 2. Initialize Sub-Systems
        self.population_evaluator = PopulationEvaluator(
            retriever, evaluator, fitness_calculator, queries, ground_truth,
            max_workers=self.max_workers
        )
        self.loop = EvolutionLoop(self.evo_context)

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

    def optimize(self, initial_population: List[Genome] = None, parallel = True) -> Genome:
        population = initial_population or self.create_initial_genomes()
        best_genome: Genome = None

        print(f"Starting evolution: {len(population)} agents, {self.config.n_generations} gens.")
        print(f"Global Concurrency: {self.max_workers} workers")

        for gen in range(self.config.n_generations):
            t0 = time.time()
            
            # 1. EVALUATE
            self.population_evaluator.evaluate(population, parallel_eval=parallel)
            
            # 2. STATS
            population.sort(key=lambda g: g.fitness, reverse=True)
            current_best = population[0]
            
            if best_genome is None or current_best.fitness > best_genome.fitness:
                best_genome = current_best.copy() # Important: Copy so it doesn't get mutated later

            avg_fit = np.mean([g.fitness for g in population])
            print(f"Gen {gen+1}: Best={current_best.fitness:.4f} (All Time: {best_genome.fitness:.4f}) | Avg={avg_fit:.4f} | Time={time.time()-t0:.2f}s")
            
            # 3. BREED (Skip on last gen)
            if gen < self.config.n_generations - 1:
                population = self.loop.step(population)

        print(f"Optimization Complete. Best Fitness: {best_genome.fitness:.4f}")
        return best_genome