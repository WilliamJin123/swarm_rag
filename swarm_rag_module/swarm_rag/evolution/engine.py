import dataclasses
import json
import random
import copy
import time
from typing import Any, List, Dict, Optional, Callable, Set, Tuple
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

        self.evo_context = EvolutionContext(
            config = config or EvolutionConfig(),
            available_features=list(HeuristicRegistry.all().keys()),
            expression_features={
                "movement": HeuristicRegistry.all_movement().keys(),
                "ranking": HeuristicRegistry.all_ranking().keys(),
                "deposit": HeuristicRegistry.all_deposit().keys(),
            }
        )        
        # Resolve Strategies from Registry
        self.selection_fn = GeneticRegistry.get_selection(self.evo_context.config.selection_strategy)
        self.crossover_fn = GeneticRegistry.get_crossover(self.evo_context.config.crossover_strategy)
        self.mutation_fn = GeneticRegistry.get_mutation(self.evo_context.config.mutation_strategy)

    def create_initial_genomes(self, count: int) -> List[Genome]:
        """Creates a diverse initial population using ramped half-and-half."""
        # Generate diverse expression trees for each heuristic
        movement_exprs = ExpressionEvolution.generate_ramped_half_and_half(
            features=self.evo_context.expression_features.get("movement", ['semantic_similarity', 'centrality', 'pheromone_repulsion', 'random_jitter']),
            population_size=count,
            max_depth=5
        )
        ranking_exprs = ExpressionEvolution.generate_ramped_half_and_half(
            features=self.evo_context.expression_features.get("ranking", ['percentage_visited', 'semantic_rank']),
            population_size=count,
            max_depth=5
        )
        deposit_exprs = ExpressionEvolution.generate_ramped_half_and_half(
            features=self.evo_context.expression_features.get("deposit", ['flat', 'semantic', 'hub', 'exploration_bonus', "collaborative_amp"]),
            population_size=count,
            max_depth=5
        )

        # TO CHANGE
        population = []
        for i in range(count):
            genome = Genome(
                # Randomly initialize hyperparameters
                n_agents=random.randint(5, 30),
                steps=random.randint(5, 20),
                decay=random.uniform(0.85, 0.99),
                initial_pool_size=random.randint(10, 50),
                start_subset=random.randint(5, 15),
                
                # Assign the pre-generated, diverse trees
                movement_expr=movement_exprs[i],
                ranking_expr=ranking_exprs[i],
                deposit_expr=deposit_exprs[i],
            )
            population.append(genome)
            
        return population

    def optimize(self, initial_population: List[Genome]) -> Genome:
        population = initial_population
        best_genome: Genome = None
        
        print(f"Starting evolution: {len(population)} agents, {self.evo_context.config.n_generations} gens.")

        for gen in range(self.evo_context.config.n_generations):
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
            if gen < self.evo_context.config.n_generations - 1:
                population = self._breed_next_generation(population, gen)

        print(f"Optimization Complete. Best Fitness: {best_genome.fitness:.4f}")

        return best_genome    
    
    def _breed_next_generation(self, population: List[Genome], generation_idx: int) -> List[Genome]:
        """
        Uses EvolutionContext to standardize the breeding pipeline.
        """
        next_gen = []
        pop_size = self.evo_context.config.population_size
        
        # Update Context
        self.evo_context.population = population
        self.evo_context.generation = generation_idx
        # Elitism (Top N)
        elite_count = int(pop_size * self.evo_context.config.elite_fraction)
        # Deep copy elites to ensure they aren't mutated in the next generation
        next_gen.extend([g.copy() for g in population[:elite_count]])
        
        # Breeding
        while len(next_gen) < pop_size:
            p1 = self.selection_fn(self.evo_context)
            p2 = self.selection_fn(self.evo_context)
            
            # Crossover (Pass Parents + Context)
            if random.random() < self.evo_context.config.crossover_rate:
                child = self.crossover_fn(p1, p2, self.evo_context)
            else:
                child = p1.copy()
            
            # Mutation (Pass Child + Context)
            child = self.mutation_fn(child, self.evo_context)
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

            genome.fitness = self.fitness_calc.calculate(averaged_metrics, genome)
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
        """Compiles expression tree into a callable strategy using a fast lambda."""
        
        # 1. Extract the feature names the tree needs to run.
        required_features = self._extract_features(expr_tree)
        
        # 2. Compile the tree structure into a lambda function ONCE.
        compiled_lambda = expr_tree.compile()
        
        # 3. Return a wrapper that prepares the inputs for the lambda at runtime.
        def strategy_wrapper(ctx: HeuristicContext) -> float:
            # Build the feature dictionary from the HeuristicContext.
            feature_values = {name: HeuristicRegistry.get(name)(ctx) for name in required_features}
            
            # Execute the pre-compiled, fast lambda.
            score = compiled_lambda(feature_values)
            
            # Clamp the final score for safety.
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

    def save_checkpoint(self, filepath: str, population: List[Genome], generation: int, best_genome: Genome):
        state = {
            "population": [dataclasses.asdict(g) for g in population],
            "generation": generation,
            "best_genome": dataclasses.asdict(best_genome),
            "config": dataclasses.asdict(self.config)
        }
        # You'll need a custom way to serialize/deserialize ExpressionNodes
        # e.g., using to_string() and a parser, or custom JSON encoder.
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_checkpoint(self, filepath: str) -> Tuple[List[Genome], int, Genome]:
    # ... logic to load and reconstruct Genome objects from the dict ...
    # This requires a parser for the expression strings.
        pass