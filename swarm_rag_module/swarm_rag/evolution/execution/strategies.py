
from typing import Callable, ClassVar, Dict, List, Union
import random
import numpy as np

from ..types.config import EvolutionContext
from ..types.expressions import ExpressionEvolution
from ...interfaces.registry import _MutationRegistry, _CrossoverRegistry, _SelectionRegistry, _CreationRegistry
from ...interfaces.enums import GeneticKey
from ..types.genome import Genome, DEFAULT_PARAMS

class GeneticRegistry:
    selection = _SelectionRegistry
    crossover = _CrossoverRegistry
    mutation  = _MutationRegistry
    creation  = _CreationRegistry

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
    def register_creation(cls, name: "GeneticKey"):
        return cls.creation.register(name)

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
    def get_creation(cls, name: Union["GeneticKey", str]) -> Callable:
        return cls.creation.get(name)

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
        except KeyError: pass
        try: return cls.creation.get(name)
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
    def all_creation(cls):
        return cls.creation.all()

    @classmethod
    def all(cls):
        return {
            **cls.selection.all(),
            **cls.crossover.all(),
            **cls.mutation.all(),
            **cls.creation.all(),
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
        winners = []
        for _ in range(k):
            indices = np.random.randint(0, pop_size, size=tourn_size)
            contestants = [ctx.population[i] for i in indices]
            # Select winner by FITNESS, not index
            winner = max(contestants, key=lambda g: g.fitness)
            winners.append(winner)
            
        return winners

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
        child.mutation_rate = (parent1.mutation_rate + parent2.mutation_rate) / 2.0

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
    
    # --- CREATION ---

    @staticmethod
    @GeneticRegistry.register_creation(GeneticKey.STANDARD_INITIALIZATION)
    def standard_initialization(ctx: EvolutionContext, count: int) -> List[Genome]:
        """
        Default initialization strategy:
        - Ramped Half-and-Half for expression trees.
        - Uniform random sampling for scalar parameters.
        """
        ranges = ctx.config['param_ranges']
        max_d = ctx.config['expr_max_depth']
        n_groups = ctx.config["n_agent_groups"]

        strat_trees = {}
        for strat_type in ["movement", "deposit"]:
            features = ctx.expression_features[strat_type]
            total_trees = count * n_groups
            
            flat_list = ExpressionEvolution.generate_ramped_half_and_half(
                features=features,
                population_size=total_trees,
                max_depth=max_d
            )
            strat_trees[strat_type] = flat_list

        ranking_features = ctx.expression_features["ranking"]
        ranking_trees = ExpressionEvolution.generate_ramped_half_and_half(
            features=ranking_features,
            population_size=count,
            max_depth=max_d
        )

        base_rate = ctx.config['base_mutation_rate']

        population = []
        for i in range(count):
            # Randomize Global Params
            params = DEFAULT_PARAMS.copy()
            for key in params.keys():
                if key in ranges:
                    min_v, max_v = ranges[key]
                    if isinstance(min_v, int):
                        params[key] = random.randint(min_v, max_v)
                    else:
                        params[key] = random.uniform(min_v, max_v)

            # Randomize Group Ratios & Assign Trees
            strategies = {}
            group_ratios = {}

            strategies["ranking"] = ranking_trees[i]
            
            for g_idx in range(n_groups):
                # Ratio
                min_r, max_r = ranges.get("group_ratio", (0.1, 1.0))
                group_ratios[f"g{g_idx}"] = random.uniform(min_r, max_r)
                
                # Strategies (Pop from pre-generated list)
                strategies[f"g{g_idx}_movement"] = strat_trees["movement"].pop()
                strategies[f"g{g_idx}_deposit"] = strat_trees["deposit"].pop()
            
            # Jitter the initial rate so the population starts diverse
            start_rate = max(0.01, min(0.5, random.gauss(base_rate, 0.05)))

            genome = Genome(
                id=f"gen0_{i}",
                params=params,
                group_ratios=group_ratios,
                strategies=strategies,
                mutation_rate=start_rate
            )
            population.append(genome)
            
        return population

    @staticmethod
    @GeneticRegistry.register_creation(GeneticKey.SHALLOW_GROWTH_INITIALIZATION)
    def shallow_growth_initialization(ctx: EvolutionContext, count: int) -> List[Genome]:
        """
        Alternative initialization:
        - Forces shallow trees (max_depth=2) using 'grow' method.
        - Useful for starting with simple, interpretable strategies.
        """
        ranges = ctx.config['param_ranges']
        n_groups = ctx.config["n_agent_groups"]
        base_rate = ctx.config['base_mutation_rate']
        
        population = []
        for i in range(count):
            # 1. Params
            params = DEFAULT_PARAMS.copy()
            for key in params.keys():
                if key in ranges:
                    min_v, max_v = ranges[key]
                    if isinstance(min_v, int):
                        params[key] = random.randint(min_v, max_v)
                    else:
                        params[key] = random.uniform(min_v, max_v)

            # 2. Strategies (Generated on the fly per genome, shallow)
            strategies = {}
            group_ratios = {}
            
            # Ranking (Depth 2, Grow)
            strategies["ranking"] = ExpressionEvolution.random_tree(
                features=ctx.expression_features["ranking"], 
                max_depth=2, 
                method='grow'
            )

            for g_idx in range(n_groups):
                min_r, max_r = ranges.get("group_ratio", (0.1, 1.0))
                group_ratios[f"g{g_idx}"] = random.uniform(min_r, max_r)
                
                strategies[f"g{g_idx}_movement"] = ExpressionEvolution.random_tree(
                    features=ctx.expression_features["movement"], 
                    max_depth=2, 
                    method='grow'
                )
                strategies[f"g{g_idx}_deposit"] = ExpressionEvolution.random_tree(
                    features=ctx.expression_features["deposit"], 
                    max_depth=2, 
                    method='grow'
                )

            start_rate = max(0.01, min(0.5, random.gauss(base_rate, 0.05)))
            
            genome = Genome(
                id=f"gen0_shallow_{i}",
                params=params,
                group_ratios=group_ratios,
                strategies=strategies,
                mutation_rate=start_rate
            )
            population.append(genome)

        return population

    # --- MUTATION ---

    @staticmethod
    @GeneticRegistry.register_mutation(GeneticKey.EXPRESSION_TREE_MUTATION)
    def expression_tree_mutation(genome: Genome, ctx: EvolutionContext) -> Genome:
        tau = 0.2
        genome.mutation_rate = genome.mutation_rate * np.exp(tau * np.random.normal(0, 1))
        genome.mutation_rate = max(0.01, min(0.5, genome.mutation_rate))
        rate = genome.mutation_rate * ctx.global_mutation_multiplier

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

        # Group Ratio Mutation
        for key, val in genome.group_ratios.items():
            if random.random() < rate:
                # Jitter ratio
                genome.group_ratios[key] = max(0.05, min(1.0, val * random.uniform(0.8, 1.2)))

        # Strategy Tree Mutation
        for key, tree in genome.strategies.items():
            if random.random() < rate:
                feature_list = ctx.expression_features.get(key)
                
                if not feature_list:
                    if key.endswith("_movement"):
                        feature_list = ctx.expression_features.get("movement")
                    elif key.endswith("_deposit"):
                        feature_list = ctx.expression_features.get("deposit")
                    elif key == "ranking":
                        feature_list = ctx.expression_features.get("ranking")

                mutated_tree = ExpressionEvolution.mutate_tree(
                    tree,
                    features=feature_list,
                    mutation_rate=rate,
                    inplace=True 
                )
                genome.strategies[key] = mutated_tree

        genome.clear_cache()
        return genome

    @staticmethod
    @GeneticRegistry.register_mutation(GeneticKey.AGGRESSIVE_MUTATION)
    def aggressive_mutation(genome: Genome, ctx: EvolutionContext) -> Genome:
        """
        High-impact mutation strategy.
        - Higher base rate.
        - Parameters are often re-sampled from global ranges instead of jittered.
        - Tree mutations prefer 'subtree' replacement.
        """
        # 1. Boost rate
        genome.mutation_rate = 0.4 # Lock to high rate
        rate = genome.mutation_rate * ctx.global_mutation_multiplier
        ranges = ctx.config['param_ranges']

        # 2. Aggressive Parameter Resampling
        for key in genome.params.keys():
            if random.random() < rate:
                if key in ranges:
                    # 50% chance to purely resample from global range (Big Jump)
                    if random.random() < 0.5:
                        min_v, max_v = ranges[key]
                        if isinstance(ranges[key][0], int):
                            genome.params[key] = random.randint(int(min_v), int(max_v))
                        else:
                            genome.params[key] = random.uniform(min_v, max_v)
                    else:
                        # 50% chance of large jitter (+/- 30%)
                        val = genome.params[key]
                        if isinstance(val, int):
                            genome.params[key] = max(1, val + random.randint(-5, 5))
                        else:
                            genome.params[key] = max(0.01, val * random.uniform(0.7, 1.3))

        # 3. Aggressive Tree Mutation
        for key, tree in genome.strategies.items():
            if random.random() < rate:
                feature_list = ctx.expression_features.get(key)
                if not feature_list:
                    if "movement" in key: feature_list = ctx.expression_features.get("movement")
                    elif "deposit" in key: feature_list = ctx.expression_features.get("deposit")
                    elif "ranking" in key: feature_list = ctx.expression_features.get("ranking")
                
                # Force a subtree replacement (structural change) 
                # rather than just changing a node value
                # We do this by manually generating a new random subtree and swapping
                if random.random() < 0.7:
                     genome.strategies[key] = ExpressionEvolution.random_tree(feature_list, max_depth=3)
                else:
                     # Fallback to standard mutation
                     genome.strategies[key] = ExpressionEvolution.mutate_tree(
                         tree, features=feature_list, mutation_rate=1.0, inplace=True
                     )

        genome.clear_cache()
        return genome
    
