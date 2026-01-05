
from typing import Callable, ClassVar, Dict, List, Union
import random
import numpy as np

from ..types.config import EvolutionContext
from ..types.expressions import ExpressionEvolution
from ...interfaces.registry import _MutationRegistry, _CrossoverRegistry, _SelectionRegistry
from ...interfaces.enums import GeneticKey
from ..types.genome import Genome

class GeneticRegistry:
    selection = _SelectionRegistry
    crossover = _CrossoverRegistry
    mutation  = _MutationRegistry

    @classmethod
    def register_selection(cls, name: "GeneticKey"):
        return cls.selection.register(name)

    @classmethod
    def register_crossover(cls, name: "GeneticKey"):
        return cls.crossover.register(name)

    @classmethod
    def register_mutation(cls, name: "GeneticKey"):
        return cls.mutation.register(name)

    @classmethod
    def get_selection(cls, name: Union["GeneticKey", str]) -> Callable:
        return cls.selection.get(name)

    @classmethod
    def get_crossover(cls, name: Union["GeneticKey", str]) -> Callable:
        return cls.crossover.get(name)

    @classmethod
    def get_mutation(cls, name: Union["GeneticKey", str]) -> Callable:
        return cls.mutation.get(name)

    @classmethod
    def get(cls, name: Union["GeneticKey", str]) -> Callable:
        """
        Search **all** genetic registries for name
        """
        try: return cls.selection.get(name)
        except KeyError: pass
        try: return cls.crossover.get(name)
        except KeyError: pass
        try: return cls.mutation.get(name)
        except KeyError: raise KeyError(f"Genetic heuristic '{name}' is not registered.") from None

    @classmethod
    def all_selection(cls):
        return cls.selection.all()

    @classmethod
    def all_crossover(cls):
        return cls.crossover.all()

    @classmethod
    def all_mutation(cls):
        return cls.mutation.all()

    @classmethod
    def all(cls):
        return {
            **cls.selection.all(),
            **cls.crossover.all(),
            **cls.mutation.all(),
        }
class GeneticStrategies:
    """
    Standard library of genetic operators.
    """

    # --- SELECTION ---

    @staticmethod
    @GeneticRegistry.register_selection(GeneticKey.TOURNAMENT)
    def tournament_selection(ctx: EvolutionContext, k: int = 1) -> List[Genome]:
        """
        Selects 'k' parents using Tournament logic.
        """
        tourn_size = ctx.config["selection_k"]
        pop_size = len(ctx.population)
        contestant_indices = np.random.randint(0, pop_size, size=(k, tourn_size))
        winner_indices = np.min(contestant_indices, axis=1)
        return [ctx.population[i] for i in winner_indices]

    @staticmethod
    @GeneticRegistry.register_selection(GeneticKey.ROULETTE)
    def roulette_selection(ctx: EvolutionContext, k: int = 1) -> List[Genome]:
        """
        Vectorized Roulette Selection (O(N) setup + O(k) sampling).
        Much faster than calling single roulette k times.
        """
        scores = np.array([max(0.001, g.fitness.quality_score) for g in ctx.population])
        total = np.sum(scores)
        probs = scores / total
        return list(np.random.choice(ctx.population, size=k, p=probs))

    @staticmethod
    @GeneticRegistry.register_selection(GeneticKey.STOCHASTIC_UNIVERSAL_SAMPLING)
    def stochastic_universal_sampling(ctx: EvolutionContext, k: int = 1) -> List[Genome]:
        """
        TVectorized SUS (Stochastic Universal Sampling).
        Uses searchsorted for O(log N) lookup instead of O(N) linear scan.
        """
        scores = np.array([max(0.001, g.fitness.quality_score) for g in ctx.population])

        cum_scores = np.cumsum(scores)
        total_fit = cum_scores[-1]
        
        if total_fit <= 0:
            return random.choices(ctx.population, k=k)
        
        step = total_fit / k
        start = random.uniform(0, step)
        points = start + np.arange(k) * step
        
        indices = np.searchsorted(cum_scores, points)
        indices = np.clip(indices, 0, len(ctx.population) - 1)
        return [ctx.population[i] for i in indices]

    @staticmethod
    @GeneticRegistry.register_selection(GeneticKey.TRUNCATION)
    def truncation_selection(ctx: EvolutionContext, k: int = 1) -> List[Genome]:
        """
        Adaptive Truncation (Batched).
        """
        max_gens = ctx.config['n_generations']
        progress = ctx.generation / max(1, max_gens)
        
        start_k = 0.5 
        end_k = 0.1    
        current_k = start_k - ((start_k - end_k) * progress)
        
        pop_size = len(ctx.population)
        cutoff = max(1, int(pop_size * current_k))
        
        pool = ctx.population[:cutoff]
        return random.choices(pool, k=k)

    @staticmethod
    @GeneticRegistry.register_selection(GeneticKey.DIVERSITY_TRUNCATION)
    def diversity_truncation_selection(ctx: EvolutionContext, k: int = 1) -> List[Genome]:
        """
        Diversity Truncation (Batched).
        """
        qualities = [g.fitness.quality_score for g in ctx.population]
        diversity = np.std(qualities) if qualities else 0.0
        
        # Dynamic cutoff based on population stagnation
        if diversity < 0.01:
            current_k = 0.6  # High diversity mode
        else:
            current_k = 0.2  # High exploitation mode
            
        cutoff = max(1, int(len(ctx.population) * current_k))
        pool = ctx.population[:cutoff]
        return random.choices(pool, k=k)

    # --- CROSSOVER ---

    @staticmethod
    @GeneticRegistry.register_crossover(GeneticKey.UNIFORM_PARAMETER_MIX)
    def uniform_parameter_mix(parent1: Genome, parent2: Genome, ctx: EvolutionContext) -> Genome:
        """Mixes traits 50/50. """
        child = parent1.copy()
        for key in child.params:
            if random.random() > 0.5:
                child.params[key] = parent2.params[key]
       
        for key in child.group_ratios:
            if key in parent2.group_ratios and random.random() > 0.5:
                child.group_ratios[key] = parent2.group_ratios[key]

        for key in child.strategies:
            if random.random() > 0.5:
                child.strategies[key] = parent2.strategies[key].copy()

        child.clear_cache()
        return child
    
    # --- MUTATION ---

    @staticmethod
    @GeneticRegistry.register_mutation(GeneticKey.EXPRESSION_TREE_MUTATION)
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

        # Group Ratio Mutation (Heterogeneous Logic)
        for key, val in genome.group_ratios.items():
            if random.random() < rate:
                # Jitter ratio
                genome.group_ratios[key] = max(0.05, min(1.0, val * random.uniform(0.8, 1.2)))

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
    
