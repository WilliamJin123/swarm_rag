import numpy as np
from typing import List, Dict, Callable, Any, Optional, Tuple, Union
import json
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import math
import random
import time

from ..heuristics import HeuristicContext, Heuristics
from ..swarm_retriever import SwarmRetriever
from ...interfaces.base import GraphStore, VectorStore, EmbeddingProvider
from ...eval.metrics import Evaluator

from .expressions import ExpressionNode, ExpressionEvolution
from .genome import Genome

# Time complexity: O(population × generations × evaluation) 
  
class EvolutionOptimizer:
    """
    Evolves hyperparameters and expression trees together.
    Single population / evolution loop for both (might change in the future)
    """

    # Default configuration
    _DEFAULT_PARAMS = dict(
        population_size=30,
        n_generations=20,
        top_k=20,
        mutation_rate=0.3,
        crossover_rate=0.6,
        max_expr_size=25,
        elite_fraction=0.1,
        tournament_size=3,
        convergence_threshold=0.01,
        convergence_window=3,
        early_stopping=True,
        # Evaluation metrics to optimize
        primary_metric='Recall@20',  # Main metric for fitness
        secondary_metrics=['MRR', 'Hit@10'],  # Additional metrics to track
        fitness_weights={
            'primary': 0.7,      # Weight for primary metric
            'secondary': 0.2,    # Weight for secondary metrics (split equally)
            'complexity': 0.05,  # Penalty for complex expressions
            'latency': 0.05      # Penalty for slow retrieval
        },
        evaluator=None,  # Custom evaluator instance (uses Evaluator if None)
        parallel_queries=False,
        max_concurrent_queries=None,
        show_progress=True,
        print_generation_stats=True,
        save_history=True,
        max_history_size=5  # Keep top N genomes per generation
    )

    def __init__(
        self,
        retriever,
        queries: List[str],
        ground_truth: List[List[int]],
        # Evolution parameters (all optional, use defaults if None)
        population_size: Optional[int] = None,
        n_generations: Optional[int] = None,
        top_k: Optional[int] = None,
        mutation_rate: Optional[float] = None,
        crossover_rate: Optional[float] = None,
        max_expr_size: Optional[int] = None,
        elite_fraction: Optional[float] = None,
        tournament_size: Optional[int] = None,
        convergence_threshold: Optional[float] = None,
        convergence_window: Optional[int] = None,
        early_stopping: Optional[bool] = None,
        # Evaluation configuration
        primary_metric: Optional[str] = None,
        secondary_metrics: Optional[List[str]] = None,
        fitness_weights: Optional[Dict[str, float]] = None,
        evaluator: Optional[Any] = None,
        # Performance
        parallel_queries: Optional[bool] = None,
        max_concurrent_queries: Optional[int] = None,
        # Display
        show_progress: Optional[bool] = None,
        print_generation_stats: Optional[bool] = None,
        save_history: Optional[bool] = None,
        max_history_size: Optional[int] = None
    ):
        """
        Initialize the evolutionary optimizer.
        
        Args:
            retriever: SwarmRetriever instance to evaluate strategies
            queries: List of query strings for evaluation
            ground_truth: List of lists of relevant doc IDs per query
            
            Evolution Control:
                population_size: Number of genomes in population (default: 30)
                n_generations: Maximum generations to evolve (default: 20)
                top_k: Number of results to retrieve per query (default: 20)
                
            Genetic Operations:
                mutation_rate: Probability of mutation (default: 0.3)
                crossover_rate: Probability of crossover (default: 0.6)
                max_expr_size: Maximum expression tree size (default: 25)
                elite_fraction: Fraction of top genomes to preserve (default: 0.1)
                tournament_size: Tournament selection size (default: 3)
                
            Convergence:
                convergence_threshold: Fitness difference for convergence (default: 0.01)
                convergence_window: Generations to check convergence (default: 3)
                early_stopping: Stop if converged (default: True)
                
            Evaluation:
                primary_metric: Main metric to optimize (default: 'Recall@20')
                                Available: 'Recall@K', 'Hit@K', 'MRR', 'DR@20', etc.
                secondary_metrics: List of additional metrics to track (default: ['MRR', 'Hit@10'])
                fitness_weights: Dict with keys 'primary', 'secondary', 'complexity', 'latency'
                                (default: {primary: 0.7, secondary: 0.2, complexity: 0.05, latency: 0.05})
                evaluator: Custom evaluator instance (default: uses your Evaluator class)
                           Must have calculate_metrics(retrieved_nodes, ground_truth_ids, latency_sec)
                
            Performance:
                parallel_queries: Enable parallel query processing (default: False)
                max_concurrent_queries: Max concurrent queries if parallel (default: auto)
                
            Display:
                show_progress: Show tqdm progress bars (default: True)
                print_generation_stats: Print stats each generation (default: True)
                save_history: Save generation history (default: True)
                max_history_size: Max genomes to save per generation (default: 5)
        """
        self.retriever = retriever
        self.queries = queries
        self.ground_truth = ground_truth
        
        # Resolve parameters using defaults
        params = self._resolve_params(
            population_size=population_size,
            n_generations=n_generations,
            top_k=top_k,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            max_expr_size=max_expr_size,
            elite_fraction=elite_fraction,
            tournament_size=tournament_size,
            convergence_threshold=convergence_threshold,
            convergence_window=convergence_window,
            early_stopping=early_stopping,
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics,
            fitness_weights=fitness_weights,
            evaluator=evaluator,
            parallel_queries=parallel_queries,
            max_concurrent_queries=max_concurrent_queries,
            show_progress=show_progress,
            print_generation_stats=print_generation_stats,
            save_history=save_history,
            max_history_size=max_history_size
        )

        # Assign all parameters as instance variables
        for key, value in params.items():
            setattr(self, key, value)

        

    def optimize(self) -> Genome:
        """Main evolution loop"""

        population = self._initialize_population()
        
        for generation in range(self.n_generations):
            print(f"\n{'='*80}")
            print(f"GENERATION {generation + 1}/{self.n_generations}")
            print(f"{'='*80}")
            # Eval fitness
            population = self._evaluate_population(population)

            # Sort and track
            population.sort(key=lambda g: g.fitness, reverse=True)
            self.population_history.append(copy.deepcopy(population[:5]))  # Save top 5

            if self.best_genome is None or population[0].fitness > self.best_genome.fitness:
                self.best_genome = copy.deepcopy(population[0])

            self._print_generation_stats(generation, population)

            if self._has_converged(population):
                print("\n⚡ Converged - stopping early")
                break

            if generation < self.n_generations - 1:
                population = self._evolve_population(population)
        
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        self._print_best_genome()
        
        return self.best_genome
    
    def _initialize_population(self) -> List[Genome]:
        """Create initial population with diverse strategies."""
        population = []
        
        # Seed with some known good baselines
        for _ in range(3):
            genome = self._create_baseline_genome()
            population.append(genome)
        
        # Fill with random genomes
        while len(population) < self.population_size:
            population.append(self._random_genome())
        
        return population
    
    def _create_baseline_genome(self) -> Genome:
        """Create a simple baseline genome."""
        features_mov = ['semantic', 'centrality']
        features_rank = ['semantic', 'votes']
        features_dep = ['flat']
        
        # Simple weighted sum
        movement = ExpressionNode('op', '+', [
            ExpressionNode('op', '*', [
                ExpressionNode('const', 0.5),
                ExpressionNode('feature', 'semantic')
            ]),
            ExpressionNode('op', '*', [
                ExpressionNode('const', 0.5),
                ExpressionNode('feature', 'centrality')
            ])
        ])
        
        ranking = ExpressionNode('op', '+', [
            ExpressionNode('op', '*', [
                ExpressionNode('const', 0.6),
                ExpressionNode('feature', 'semantic')
            ]),
            ExpressionNode('op', '*', [
                ExpressionNode('const', 0.4),
                ExpressionNode('feature', 'votes')
            ])
        ])
        
        deposit = ExpressionNode('feature', 'flat')
        
        return Genome(
            n_agents=20,
            steps=4,
            decay=0.5,
            initial_pool_size=30,
            start_subset=10,
            movement_expr=movement,
            ranking_expr=ranking,
            deposit_expr=deposit
        )
    
    def _random_genome(self) -> Genome:
        """Generate random genome with random hyperparams and expressions."""
        features_mov = ['semantic', 'centrality', 'diversity', 'jitter']
        features_rank = ['semantic', 'votes', 'centrality']
        features_dep = ['flat', 'semantic', 'hub', 'explorer']
        
        return Genome(
            n_agents=random.choice([10, 20, 30, 50]),
            steps=random.randint(2, 6),
            decay=random.uniform(0.3, 0.8),
            initial_pool_size=random.choice([30, 50, 100]),
            start_subset=random.randint(5, 20),
            movement_expr=ExpressionEvolution.random_tree(features_mov, max_depth=4),
            ranking_expr=ExpressionEvolution.random_tree(features_rank, max_depth=3),
            deposit_expr=ExpressionEvolution.random_tree(features_dep, max_depth=3)
        )
    
    def _evaluate_population(self, population: List[Genome]) -> List[Genome]:
        """Evaluate all genomes in parallel."""
        print(f"\nEvaluating {len(population)} genomes...")
        
        if self.n_workers > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(self._evaluate_genome, genome): i
                    for i, genome in enumerate(population)
                }
                
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        population[idx] = future.result()
                        print(f"  ✓ {idx+1}/{len(population)} - Fitness: {population[idx].fitness:.4f}")
                    except Exception as e:
                        print(f"  ✗ {idx+1} failed: {e}")
                        population[idx].fitness = 0.0
        else:
            for i, genome in enumerate(population):
                population[i] = self._evaluate_genome(genome)
                print(f"  ✓ {i+1}/{len(population)} - Fitness: {population[i].fitness:.4f}")
        
        return population
    
    def _evaluate_genome(self, genome: Genome) -> Genome:
        """
        Evaluate a single genome by running retrieval.
        NO symbolic regression here - just direct evaluation!
        """        
        # Convert expression trees to callable functions
        strategies = self._genome_to_strategies(genome)
        
        # Run retrieval
        start_time = time.time()
        all_recalls = []
        all_mrrs = []
        all_hits = []

        #TO CHANGE

    def _genome_to_strategies(self, genome: Genome) -> Dict:
        """Convert expression trees to callable strategy functions."""
        
        def make_heuristic(expr_tree: ExpressionNode, feature_list: List[str]) -> Callable:
            """Create a heuristic function from expression tree."""
            def heuristic(ctx: HeuristicContext) -> float:
                # Evaluate base features
                features = {
                    name: self.feature_funcs[name](ctx)
                    for name in feature_list
                }
                # Evaluate expression tree
                score = expr_tree.evaluate(features)
                # Clamp to valid range
                return max(0.0, min(1.0, score))
            return heuristic
        
        return {
            'movement_strategies': {
                'evolved': (
                    make_heuristic(genome.movement_expr, genome.available_movement_features),
                    1.0
                )
            },
            'ranking_strategies': {
                'evolved': (
                    make_heuristic(genome.ranking_expr, genome.available_ranking_features),
                    1.0
                )
            },
            'deposit_strategies': {
                'evolved': (
                    make_heuristic(genome.deposit_expr, genome.available_deposit_features),
                    1.0
                )
            }
        }
    
    def _evolve_population(self, population: List[Genome]) -> List[Genome]:
        """Create next generation."""
        elite_size = max(2, self.population_size // 10)
        next_gen = [g.copy() for g in population[:elite_size]]
        
        while len(next_gen) < self.population_size:
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)
            
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)
            
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            # Enforce complexity limit
            child.movement_expr = ExpressionEvolution.simplify_tree(
                child.movement_expr, self.max_expr_size
            )
            child.ranking_expr = ExpressionEvolution.simplify_tree(
                child.ranking_expr, self.max_expr_size
            )
            child.deposit_expr = ExpressionEvolution.simplify_tree(
                child.deposit_expr, self.max_expr_size
            )
            
            next_gen.append(child)
        
        return next_gen
    
    def _tournament_select(self, population: List[Genome], k: int = 3) -> Genome:
        """Tournament selection."""
        tournament = random.sample(population, k)
        return max(tournament, key=lambda g: g.fitness)
    
    def _crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Crossover BOTH hyperparameters AND expression trees."""
        child = Genome(
            # Average hyperparameters
            n_agents=int((parent1.n_agents + parent2.n_agents) / 2),
            steps=int((parent1.steps + parent2.steps) / 2),
            decay=(parent1.decay + parent2.decay) / 2,
            initial_pool_size=int((parent1.initial_pool_size + parent2.initial_pool_size) / 2),
            start_subset=int((parent1.start_subset + parent2.start_subset) / 2),
            
            # Crossover expression trees
            movement_expr=ExpressionEvolution.crossover_trees(
                parent1.movement_expr, parent2.movement_expr
            )[0],
            ranking_expr=ExpressionEvolution.crossover_trees(
                parent1.ranking_expr, parent2.ranking_expr
            )[0],
            deposit_expr=ExpressionEvolution.crossover_trees(
                parent1.deposit_expr, parent2.deposit_expr
            )[0]
        )
        
        return child
    
    def _mutate(self, genome: Genome) -> Genome:
        """Mutate BOTH hyperparameters AND expression trees."""
        
        # Mutate hyperparameters
        if random.random() < 0.5:
            mutation_target = random.choice(['n_agents', 'steps', 'decay'])
            if mutation_target == 'n_agents':
                genome.n_agents = max(5, genome.n_agents + random.randint(-10, 10))
            elif mutation_target == 'steps':
                genome.steps = max(2, genome.steps + random.randint(-1, 2))
            else:
                genome.decay = np.clip(genome.decay + random.uniform(-0.1, 0.1), 0.2, 0.9)
        
        # Mutate expression trees
        if random.random() < 0.7:
            strategy = random.choice(['movement', 'ranking', 'deposit'])
            if strategy == 'movement':
                genome.movement_expr = ExpressionEvolution.mutate_tree(
                    genome.movement_expr, 
                    genome.available_movement_features,
                    0.2
                )
            elif strategy == 'ranking':
                genome.ranking_expr = ExpressionEvolution.mutate_tree(
                    genome.ranking_expr,
                    genome.available_ranking_features,
                    0.2
                )
            else:
                genome.deposit_expr = ExpressionEvolution.mutate_tree(
                    genome.deposit_expr,
                    genome.available_deposit_features,
                    0.2
                )
        
        return genome
    
    def _has_converged(self, population: List[Genome], threshold: float = 0.01) -> bool:
        """Check convergence."""
        if len(population) < 3:
            return False
        top3 = [g.fitness for g in population[:3]]
        return max(top3) - min(top3) < threshold
    
    def _print_generation_stats(self, generation: int, population: List[Genome]):
        """Print generation statistics."""
        fitnesses = [g.fitness for g in population]
        print(f"\n📊 Generation {generation + 1}:")
        print(f"   Best: {max(fitnesses):.4f} | Avg: {np.mean(fitnesses):.4f}")
        
        best = population[0]
        print(f"\n🏆 Top Genome:")
        print(f"   Agents={best.n_agents}, Steps={best.steps}, Decay={best.decay:.2f}")
        print(f"   Movement: {best.movement_expr.to_string()}")
        print(f"   Complexity: {best.complexity()}")
    
    def _print_best_genome(self):
        """Print best genome details."""
        g = self.best_genome
        print(f"\n🎯 Best Strategy:")
        print(f"   Fitness: {g.fitness:.4f} | Recall@{self.top_k}: {g.recall_at_k:.4f}")
        print(f"   Agents: {g.n_agents} | Steps: {g.steps} | Decay: {g.decay:.3f}")
        print(f"\n   Movement: {g.movement_expr.to_string()}")
        print(f"   Ranking:  {g.ranking_expr.to_string()}")
        print(f"   Deposit:  {g.deposit_expr.to_string()}")
    
    def save(self, filepath: str):
        """Save best genome."""
        state = {
            'n_agents': self.best_genome.n_agents,
            'steps': self.best_genome.steps,
            'decay': self.best_genome.decay,
            'movement_expr': self.best_genome.movement_expr.to_string(),
            'ranking_expr': self.best_genome.ranking_expr.to_string(),
            'deposit_expr': self.best_genome.deposit_expr.to_string(),
            'fitness': self.best_genome.fitness
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"\n💾 Saved to {filepath}")