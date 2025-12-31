import random
import copy
from typing import Any, List, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

from swarm_rag.core.heuristics import HeuristicRegistry
from .genome import Genome
from .genetic_strategies import GeneticRegistry
from .fitness import FitnessCalculator
from .expressions import ExpressionEvolution

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
    _DEFAULT_CONFIG = {
        "n_generations": 20,
        "population_size": 30,
        "elite_fraction": 0.1,
        "mutation_rate": 0.2,
        "crossover_rate": 0.6,
        "selection_strategy": "tournament",
        "crossover_strategy": "uniform_parameter_mix",
        "mutation_strategy": "expression_tree_mutation",
        "selection_params": {"k": 3},
        "mutation_params": {"max_expr_size": 25}
    }

    def __init__(
        self,
        retriever,
        fitness_calculator: FitnessCalculator,
        queries: List[str],
        ground_truth: List[List[Any]],
        config: Optional[Dict[str, Any]] = None
    ):
        self.retriever = retriever
        self.fitness_calc = fitness_calculator
        self.queries = queries
        self.ground_truth = ground_truth
        self.config = self._DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # Resolve Strategies from Registry
        self.selection_fn = GeneticRegistry.get_selection(self.config["selection_strategy"])
        self.crossover_fn = GeneticRegistry.get_crossover(self.config["crossover_strategy"])
        self.mutation_fn = GeneticRegistry.get_mutation(self.config["mutation_strategy"])

        # TO CHANGE
        self.available_features = list(HeuristicRegistry.all().keys())

    def optimize(self, initial_population: List[Genome], generations: int) -> Genome:
        population = initial_population
        
        for gen in range(generations):
            # 1. Evaluate
            self._evaluate_population(population)
            
            # 2. Sort
            population.sort(key=lambda g: g.fitness, reverse=True)
            print(f"Gen {gen}: Best Fitness {population[0].fitness:.4f}")
            
            # 3. Evolve (Delegate to internal pipeline)
            population = self._breed_next_generation(population)

        return population[0]
    
    def _breed_next_generation(self, population: List[Genome]) -> List[Genome]:
        """
        The breeding loop is now generic. 
        It supports Elitism -> Selection -> Crossover -> Mutation.
        """
        next_gen = []
        pop_size = len(population)
        
        # 1. Elitism (Preserve best)
        elite_count = int(pop_size * self.config["elite_fraction"])
        next_gen.extend([g.copy() for g in population[:elite_count]])
        
        # 2. Breeding Loop
        while len(next_gen) < pop_size:
            # Selection Strategy
            p1 = self.selection_fn(population, **self.config["selection_params"])
            p2 = self.selection_fn(population, **self.config["selection_params"])
            
            # Crossover Strategy
            if random.random() < self.config["crossover_rate"]:
                child = self.crossover_fn(p1, p2)
            else:
                child = p1.copy()
            
            # Mutation Strategy
            # (You would likely call ExpressionEvolution inside your registered mutation strategy)
            # child = self.mutation_fn(child, rate=self.config['mutation_rate'])
            
            next_gen.append(child)
            
        return next_gen
    
    def _evaluate_population(self, population: List[Genome]):
        """
        Runs the retriever using the genome's configuration.
        """
        for genome in population:
            # Inject Genome into Retriever
            # This requires your SwarmRetriever to accept these raw params/trees
            # or a helper to convert Genome -> SwarmRetriever args
            
            # Simulated result for now
            raw_metrics = self.run_simulation(genome) 
            genome.metrics = raw_metrics
            genome.fitness = self.fitness_calc.calculate(raw_metrics)

    def run(self, genome):
        pass