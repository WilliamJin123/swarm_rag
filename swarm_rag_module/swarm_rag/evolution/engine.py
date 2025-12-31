import random
import copy
import time
from typing import Any, List, Dict, Optional, Callable, Set
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from swarm_rag.core.heuristics import HeuristicContext, HeuristicRegistry
from swarm_rag.core.swarm_retriever import SwarmRetriever
from swarm_rag.eval.metrics import Evaluator
from swarm_rag.evolution.evolution_context import EvolutionConfig, EvolutionContext
from .genome import Genome
from .genetic_strategies import GeneticRegistry
from .fitness import FitnessCalculator
from .expressions import ExpressionEvolution, ExpressionNode

# Time complexity: O(population × generations × evaluation) 
  
class EvolutionEngine:
    """
    Evolves hyperparameters and expression trees together.
    Single population / evolution loop for both (might change in the future)
    Orchestrates the evolution of Genomes by delegating to:
    1. SwarmRetriever (Evaluation)
    2. FitnessCalculator (Scoring)
    3. GeneticRegistry (Breeding)
    """

    # Default configuration
    # _DEFAULT_CONFIG = {
    #     "n_generations": 20,
    #     "population_size": 30,
    #     "elite_fraction": 0.1,
    #     "mutation_rate": 0.2,
    #     "crossover_rate": 0.6,
    #     "selection_strategy": "tournament",
    #     "crossover_strategy": "uniform_parameter_mix",
    #     "mutation_strategy": "expression_tree_mutation",
    #     "selection_params": {"k": 3},
    #     "mutation_params": {"max_expr_size": 25}
    # }

    def __init__(
        self,
        retriever: SwarmRetriever,
        fitness_calculator: FitnessCalculator,
        evaluator: Evaluator,
        queries: List[str],
        ground_truth: List[List[Any]],
        config: EvolutionConfig = None
    ):
        self.retriever = retriever
        self.fitness_calc = fitness_calculator
        self.evaluator = evaluator
        self.queries = queries
        self.ground_truth = ground_truth

        self.config = config or EvolutionConfig() # Uses defaults if None
        
        # Resolve Strategies from Registry
        self.selection_fn = GeneticRegistry.get_selection(self.config["selection_strategy"])
        self.crossover_fn = GeneticRegistry.get_crossover(self.config["crossover_strategy"])
        self.mutation_fn = GeneticRegistry.get_mutation(self.config["mutation_strategy"])

        # TO CHANGE
        self.available_features = list(HeuristicRegistry.all().keys())

    def optimize(self, initial_population: List[Genome]) -> Genome:
        population = initial_population
        best_genome: Genome = None
        
        print(f"Starting evolution: {len(population)} agents, {self.config.n_generations} gens.")

        for gen in range(self.config.n_generations):
            t0 = time.time()
            
            # Evaluate Fitness
            self._evaluate_population(population)
            
            # Sort and Stats
            population.sort(key=lambda g: g.fitness, reverse=True)
            if best_genome is None or population[0].fitness > best_genome.fitness:
                best_genome = population[0].copy()

            avg_fit = np.mean([g.fitness for g in population])
            print(f"Gen {gen+1}: Best={population[0].fitness:.4f} | Avg={avg_fit:.4f} | Time={time.time()-t0:.2f}s")
            
            # Breed
            if gen < self.config.n_generations - 1:
                population = self._breed_next_generation(population, gen)

        print(f"Optimization Complete. Best Fitness: {best_genome.fitness:.4f}")

        return best_genome    
    
    def _breed_next_generation(self, population: List[Genome], generation_idx: int) -> List[Genome]:
        """
        Uses EvolutionContext to standardize the breeding pipeline.
        """
        next_gen = []
        pop_size = self.config.population_size
        
        # Create Context
        ctx = EvolutionContext(
            population=population,
            generation=generation_idx,
            config=self.config
        )
        
        # Elitism (Top N)
        elite_count = int(pop_size * self.config.elite_fraction)
        # Deep copy elites to ensure they aren't mutated in the next generation
        next_gen.extend([g.copy() for g in population[:elite_count]])
        
        # Breeding
        while len(next_gen) < pop_size:
            p1 = self.selection_fn(ctx)
            p2 = self.selection_fn(ctx)
            
            # Crossover (Pass Parents + Context)
            if random.random() < self.config.crossover_rate:
                child = self.crossover_fn(p1, p2, ctx)
            else:
                child = p1.copy()
            
            # Mutation (Pass Child + Context)
            child = self.mutation_fn(child, ctx)
            next_gen.append(child)
            
        return next_gen
    
    def _evaluate_population(self, population: List[Genome]):
        """
        Evaluates the entire population.
        """
        for i, genome in enumerate(population):
            # Skip evaluation if we already know the fitness (e.g., Elitism)
            if genome.fitness > 0 and genome.metrics:
                continue
                
            # 1. Convert Genome to Retriever Arguments
            # This translates the ExpressionTrees into callable functions
            retriever_kwargs = self._genome_to_retriever_args(genome)
            
            # 2. Run Batch Retrieval
            start_time = time.time()
            batch_results = self.retriever.retrieve_batch(
                queries=self.queries,
                parallel_queries=True, 
                **retriever_kwargs
            )
            total_latency = time.time() - start_time
            avg_latency = total_latency / len(self.queries) if self.queries else 0.0

            # TO CHANGE

            # 3. Compute Metrics
            all_query_metrics = []
            
            for q_idx, retrieved_items in enumerate(batch_results):
                q_metrics = self.evaluator.calculate_metrics(
                    retrieved_nodes=retrieved_items, 
                    ground_truth_ids=self.ground_truth[q_idx], 
                    latency_sec=avg_latency 
                )
                all_query_metrics.append(q_metrics)
            
            averaged_metrics = self._mean_metrics(all_query_metrics)

            genome.fitness = self.fitness_calc.calculate(averaged_metrics)
            genome.metrics = averaged_metrics

    def _mean_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Averages metric dictionaries into a single result."""
        if not all_metrics: return {}
        keys = all_metrics[0].keys()
        # Filter for numeric values only
        return {
            k: float(np.mean([m[k] for m in all_metrics])) 
            for k in keys if isinstance(all_metrics[0][k], (int, float))
        }

    def _genome_to_retriever_args(self, genome: Genome) -> Dict[str, Any]:
        """Compiles Genome fields into SwarmRetriever arguments."""
        kwargs = {
            'n_agents': genome.n_agents,
            'steps': genome.steps,
            'decay': genome.decay,
            'initial_pool_size': genome.initial_pool_size,
            'start_subset': genome.start_subset,
            'movement_strategies': {'evolved_move': (self._compile_tree(genome.movement_expr), 1.0)},
            'ranking_strategies': {'evolved_rank': (self._compile_tree(genome.ranking_expr), 1.0)},
            'deposit_strategies': {'evolved_deposit': (self._compile_tree(genome.deposit_expr), 1.0)}
        }
        return kwargs

    def _compile_tree(self, expr_tree: ExpressionNode) -> Callable[[HeuristicContext], float]:
        """Compiles expression tree into a callable strategy."""
        required_features = self._extract_features(expr_tree)
        
        def strategy_wrapper(ctx: HeuristicContext) -> float:
            feature_values = {}
            for name in required_features:
                func = HeuristicRegistry.get(name)
                feature_values[name] = func(ctx)
            
            # Evaluate the tree logic
            score = expr_tree.evaluate(feature_values)
            # Clamp for safety            
            return max(0.0, min(1.0, score))
            
        return strategy_wrapper

    def _extract_features(self, node: ExpressionNode) -> Set[str]:
        """Helper to find all unique feature names in a tree."""
        features = set()
        if node.type == 'feature':
            features.add(node.value)
        for child in node.children:
            features.update(self._extract_features(child))
        return features
