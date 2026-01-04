import dataclasses
from typing import Any, Callable
import random

import numpy as np

from swarm_rag.evolution.types.config import EvolutionContext
from swarm_rag.evolution.types.expressions import ExpressionEvolution, ExpressionNode
from ..types.genome import Genome

class GeneticRegistry:
    _selection_registry = {}
    _crossover_registry = {}
    _mutation_registry = {}

    @classmethod
    def register_selection(cls, name: str):
        def decorator(fn):
            cls._selection_registry[name] = fn
            return fn
        return decorator

    @classmethod
    def register_crossover(cls, name: str):
        def decorator(fn):
            cls._crossover_registry[name] = fn
            return fn
        return decorator

    @classmethod
    def register_mutation(cls, name: str):
        def decorator(fn):
            cls._mutation_registry[name] = fn
            return fn
        return decorator

    @classmethod
    def get_selection(cls, name: str) -> Callable[..., Genome]:
        return cls._selection_registry[name]

    @classmethod
    def get_crossover(cls, name: str) -> Callable[..., Genome]:
        return cls._crossover_registry[name]

    @classmethod
    def get_mutation(cls, name: str) -> Callable[..., Genome]:
        return cls._mutation_registry[name]
    
    @classmethod
    def all_selection(cls) -> dict[str, Callable[..., Genome]]:
        """
        Return the complete selection registry.
        """
        return cls._selection_registry

    @classmethod
    def all_crossover(cls) -> dict[str, Callable[..., Genome]]:
        """
        Return the complete crossover registry.
        """
        return cls._crossover_registry

    @classmethod
    def all_mutation(cls) -> dict[str, Callable[..., Genome]]:
        """
        Return the complete mutation registry.
        """
        return cls._mutation_registry
    
    @classmethod
    def all(cls) -> dict[str, Callable[..., Genome]]:
        return {
            **cls._selection_registry,
            **cls._crossover_registry,
            **cls._mutation_registry,
        }

class GeneticStrategies:
    """
    Standard library of genetic operators.
    """

    # --- SELECTION ---

    @staticmethod
    @GeneticRegistry.register_selection("tournament")
    def tournament_selection(ctx: EvolutionContext) -> Genome:
        k = ctx.config.selection_k
        candidates = random.sample(ctx.population, k)
        return max(candidates, key=lambda g: g.fitness)

    @staticmethod
    @GeneticRegistry.register_selection("roulette")
    def roulette_selection(ctx: EvolutionContext) -> Genome:
        scores = np.array([max(0.001, g.fitness.quality_score) for g in ctx.population])
        total_fit = np.sum(scores)
        
        pick = random.uniform(0, total_fit)
        current = 0
        for i, g in enumerate(ctx.population):
            current += scores[i]
            if current > pick:
                return g
        return ctx.population[-1]

    @staticmethod
    @GeneticRegistry.register_selection("truncation")
    def truncation_selection(ctx: EvolutionContext) -> Genome:
        """
        Adaptive Truncation:
        - Early Gens: Wide pool (Top 50%) -> encourages diversity/exploration.
        - Late Gens: Narrow pool (Top 10%) -> forces convergence/exploitation.
        """
        # Calculate Progress (0.0 to 1.0)
        max_gens = ctx.config['n_generations']
        progress = ctx.generation / max(1, max_gens)
        
        # Define Range (Start Loose -> End Strict)
        start_k = 0.5  # Top 50%
        end_k = 0.1    # Top 10%
        
        # Linear Interpolation (Annealing)
        current_k = start_k - ((start_k - end_k) * progress)
        
        # Determine Cutoff index
        # We assume population is sorted Best -> Worst by the Loop
        pop_size = len(ctx.population)
        cutoff = max(1, int(pop_size * current_k))
        
        # Pick
        pool = ctx.population[:cutoff]
        return random.choice(pool)

    @staticmethod
    @GeneticRegistry.register_selection("diversity_truncation")
    def diversity_truncation_selection(ctx: EvolutionContext) -> Genome:
        # Calculate standard deviation of Quality Scores
        qualities = [g.fitness.quality_score for g in ctx.population]
        diversity = np.std(qualities) if qualities else 0.0
        
        # Heuristic: If diversity drops below 0.01, Panic Mode!
        if diversity < 0.01:
            current_k = 0.6 # Open the floodgates (Top 60%)
        else:
            current_k = 0.2 # Standard strict mode (Top 20%)
            
        cutoff = max(1, int(len(ctx.population) * current_k))
        return random.choice(ctx.population[:cutoff])

    # --- CROSSOVER ---

    @staticmethod
    @GeneticRegistry.register_crossover("uniform_parameter_mix")
    def uniform_parameter_mix(parent1: Genome, parent2: Genome, ctx: EvolutionContext) -> Genome:
        """Mixes traits 50/50. """
        # Create a shallow copy of Parent 1 using its exact class
        child = parent1.copy()
        for key in child.params:
            if random.random() > 0.5:
                child.params[key] = parent2.params[key]
       
        for key in child.strategies:
            if random.random() > 0.5:
                child.strategies[key] = parent2.strategies[key].copy()

        child.clear_cache()
        return child
    
    # --- MUTATION ---

    @staticmethod
    @GeneticRegistry.register_mutation("expression_tree_mutation")
    def expression_tree_mutation(genome: Genome, ctx: EvolutionContext) -> Genome:
        rate = ctx.config['mutation_rate']
        
        # Parameter Mutation (Numeric Jitter)
        for key, val in genome.params.items():
            if random.random() < rate:
                if isinstance(val, int):
                    # Integer Jitter (+/- 1 to 3)
                    delta = random.randint(-2, 2)
                    new_val = max(1, val + delta)
                    genome.params[key] = new_val
                elif isinstance(val, float):
                    # Float Jitter (+/- 10%)
                    new_val = val * random.uniform(0.9, 1.1)
                    # Clamp to reasonable bounds (0-1 for decay usually)
                    if 0.0 < val < 1.0: 
                        new_val = max(0.01, min(0.999, new_val))
                    genome.params[key] = new_val

        # Strategy Tree Mutation
        for key, tree in genome.strategies.items():
            if random.random() < rate:
                # Find available features for this specific strategy type (e.g. 'movement')
                # Assumes config maps 'movement' -> ['degree', 'cosine']
                feature_list = ctx.expression_features.get(key, [])
                
                mutated_tree = ExpressionEvolution.mutate_tree(
                    tree,
                    features=feature_list,
                    mutation_rate=rate,
                    inplace=True 
                )
                genome.strategies[key] = mutated_tree

        genome.clear_cache()
        return genome
    
