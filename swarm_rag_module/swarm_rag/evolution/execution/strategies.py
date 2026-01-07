
from typing import Callable, ClassVar, Dict, List, Union
import random
import numpy as np

from ..types.config import EvolutionContext
from ..types.expressions import ExpressionEvolution, ExpressionNode
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

    @staticmethod
    @GeneticRegistry.register_crossover(GeneticKey.SUBTREE_CROSSOVER)
    def subtree_crossover(parent1: Genome, parent2: Genome, ctx: EvolutionContext) -> Genome:
        """
        GP-style Subtree Crossover.
        1. Mixes scalar parameters uniformly.
        2. For strategy trees, attempts to swap random subtrees between parents.
        """
        child = parent1.copy()
        child.mutation_rate = (parent1.mutation_rate + parent2.mutation_rate) / 2.0

        # 1. Uniform Parameter Mix
        for key in child.params:
            if random.random() > 0.5:
                child.params[key] = parent2.params[key]
        
        for key in child.group_ratios:
            if key in parent2.group_ratios and random.random() > 0.5:
                child.group_ratios[key] = parent2.group_ratios[key]

        # 2. Subtree Crossover for Expressions
        for key in child.strategies:
            p1_tree = parent1.strategies[key]
            p2_tree = parent2.strategies[key]
            
            # Chance to perform subtree swap vs just inheriting whole tree
            if random.random() < 0.7: # 70% chance to try mixing
                try:
                    # We need deep copies to avoid modifying parents
                    new_tree = p1_tree.copy()
                    donor_tree = p2_tree.copy()
                    
                    # Get all nodes (flatten)
                    p1_nodes = ExpressionEvolution.get_all_nodes(new_tree)
                    p2_nodes = ExpressionEvolution.get_all_nodes(donor_tree)
                    
                    if p1_nodes and p2_nodes:
                        # Pick crossover points
                        target_node = random.choice(p1_nodes)
                        source_node = random.choice(p2_nodes)
                        
                        # Swap content (type, value, children)
                        # We do this by modifying target_node in-place to become source_node
                        target_node.type = source_node.type
                        target_node.value = source_node.value
                        target_node.children = [c.copy() for c in source_node.children]
                        
                        child.strategies[key] = new_tree
                    else:
                        # Fallback
                        child.strategies[key] = p1_tree.copy()
                except Exception:
                    # Safety fallback
                    child.strategies[key] = p1_tree.copy()
            else:
                 # Just pick one parent's tree
                 if random.random() > 0.5:
                     child.strategies[key] = p2_tree.copy()
                 else:
                     child.strategies[key] = p1_tree.copy()

        child.clear_cache()
        return child

    @staticmethod
    @GeneticRegistry.register_crossover(GeneticKey.ROOT_MIX_CROSSOVER)
    def root_mix_crossover(parent1: Genome, parent2: Genome, ctx: EvolutionContext) -> Genome:
        """
        Swaps top-level branches of strategy trees.
        This is less destructive than random subtree crossover as it preserves
        the high-level logic (the operator) if both share it, or swaps whole approaches.
        """
        child = parent1.copy()
        child.mutation_rate = (parent1.mutation_rate + parent2.mutation_rate) / 2.0

        # 1. Uniform Parameter Mix
        for key in child.params:
            if random.random() > 0.5:
                child.params[key] = parent2.params[key]
        
        for key in child.group_ratios:
            if key in parent2.group_ratios and random.random() > 0.5:
                child.group_ratios[key] = parent2.group_ratios[key]

        # 2. Root Mix Crossover
        for key in child.strategies:
            p1_tree = parent1.strategies[key]
            p2_tree = parent2.strategies[key]

            # 70% chance to mix, 30% chance to clone one parent
            if random.random() < 0.7:
                # If both are operators with children (e.g. A * B, C + D)
                if p1_tree.type == 'op' and p2_tree.type == 'op' and p1_tree.children and p2_tree.children:
                     # Create new root using Parent 1's operator
                     new_tree = ExpressionNode(type='op', value=p1_tree.value, children=[])
                     
                     # Take one child from P1 and one from P2
                     # (Assumes binary operators for simplicity, or takes first child)
                     c1 = p1_tree.children[0].copy()
                     # If P2 has children, take one, otherwise take P2 itself
                     c2 = p2_tree.children[-1].copy() if len(p2_tree.children) > 1 else p2_tree.children[0].copy()
                     
                     # Randomly swap order
                     if random.random() > 0.5:
                         new_tree.children = [c1, c2]
                     else:
                         new_tree.children = [c2, c1]
                     
                     child.strategies[key] = new_tree
                else:
                    # If structures don't match well, just swap the whole tree
                    child.strategies[key] = p2_tree.copy()
            else:
                 if random.random() > 0.5:
                     child.strategies[key] = p2_tree.copy()
                 else:
                     child.strategies[key] = p1_tree.copy()

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
            genome.normalize_ratios()
            population.append(genome)

        return population

    @staticmethod
    @GeneticRegistry.register_creation(GeneticKey.SEEDED_INITIALIZATION)
    def seeded_initialization(ctx: EvolutionContext, count: int) -> List[Genome]:
        """
        Injects known effective strategies (Vector, Pheromone, Hybrid) 
        and fills the rest with random genomes.
        """
        population = []
        n_groups = ctx.config["n_agent_groups"]
        
        # --- Helper to create a seed genome ---
        def create_seed(name: str, mov_expr: str, dep_expr: str, rank_expr: str, params: dict = None):
            strategies = {}
            # Parse simple string expressions into ExpressionNodes (Manual Construction)
            # NOTE: For complex expressions, a parser would be better, but we construct simple ones manually here.
            
            def make_node(val):
                if val in ["*", "+", "-", "/"]: return ExpressionNode("op", val)
                if val in ["semantic_similarity", "pheromone_repulsion", "flat", "semantic", "semantic_rank", "percentage_visited"]:
                     return ExpressionNode("feature", val)
                return ExpressionNode("const", float(val))

            # Helper to build simple tree from strict format: "A * B" or just "A"
            def build_simple(expr_str):
                parts = expr_str.split()
                if len(parts) == 3: # A * B
                    return ExpressionNode("op", parts[1], [make_node(parts[0]), make_node(parts[2])])
                return make_node(parts[0])

            strategies["ranking"] = build_simple(rank_expr)
            
            group_ratios = {}
            for g in range(n_groups):
                group_ratios[f"g{g}"] = 1.0 / n_groups
                strategies[f"g{g}_movement"] = build_simple(mov_expr)
                strategies[f"g{g}_deposit"] = build_simple(dep_expr)

            p = DEFAULT_PARAMS.copy()
            if params: p.update(params)
            
            return Genome(id=name, params=p, group_ratios=group_ratios, strategies=strategies, mutation_rate=0.1)

        # 1. Pure Vector
        population.append(create_seed(
            "gen0_seed_vector", 
            "semantic_similarity", "semantic", "semantic_rank",
            {"steps": 4}
        ))
        
        # 2. Hybrid (Vector * Pheromone)
        if count > 1:
            population.append(create_seed(
                "gen0_seed_hybrid", 
                "semantic_similarity * pheromone_repulsion", "semantic", "semantic_rank",
                {"steps": 5}
            ))

        # 3. Pure Pheromone (Exploration)
        if count > 2:
            population.append(create_seed(
                "gen0_seed_ant", 
                "pheromone_repulsion", "flat", "percentage_visited",
                {"steps": 8, "decay": 0.95}
            ))

        # Fill remainder with standard random initialization
        remaining = count - len(population)
        if remaining > 0:
            random_pop = GeneticStrategies.standard_initialization(ctx, remaining)
            # Fix IDs to avoid collision
            for i, g in enumerate(random_pop):
                g.id = f"gen0_rand_{i}"
            population.extend(random_pop)

        return population

    # --- MUTATION ---

    @staticmethod
    @GeneticRegistry.register_mutation(GeneticKey.EXPRESSION_TREE_MUTATION)
    def expression_tree_mutation(genome: Genome, ctx: EvolutionContext) -> Genome:
        tau = 0.2
        genome.mutation_rate = genome.mutation_rate * np.exp(tau * np.random.normal(0, 1))
        genome.mutation_rate = max(0.01, min(0.5, genome.mutation_rate))
        rate = genome.mutation_rate * ctx.global_mutation_multiplier

        # 1. Parameter Mutation (Smart Jitter)
        for key, val in genome.params.items():
            if random.random() < rate:
                # 80% chance: Fine-tuning (Small Gaussian jitter)
                if random.random() < 0.8:
                    if isinstance(val, int):
                        delta = int(round(random.gauss(0, 1.5))) # +/- 1 or 2 usually
                        new_val = max(1, val + delta)
                        genome.params[key] = new_val
                    elif isinstance(val, float):
                        # +/- 10% relative change
                        factor = random.gauss(1.0, 0.1)
                        new_val = val * factor
                        # Clamp to 0.001 - 1.0 (typical for most floats here)
                        genome.params[key] = max(0.001, min(0.999, new_val))
                
                # 20% chance: Exploration (Re-sample or Large Jump)
                else:
                     ranges = ctx.config.get('param_ranges', {})
                     if key in ranges:
                         min_v, max_v = ranges[key]
                         if isinstance(min_v, int):
                             genome.params[key] = random.randint(min_v, max_v)
                         else:
                             genome.params[key] = random.uniform(min_v, max_v)

        # 2. Group Ratio Mutation
        for key, val in genome.group_ratios.items():
            if random.random() < rate:
                # Jitter ratio
                genome.group_ratios[key] = max(0.05, min(1.0, val * random.uniform(0.8, 1.2)))

        # 3. Strategy Tree Mutation
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

                # Structural Mutations
                mut_choice = random.random()
                
                if mut_choice < 0.1 and tree.type == 'op':
                    # Hoist: Replace current node with one of its children (Simplification)
                    if tree.children:
                        genome.strategies[key] = random.choice(tree.children).copy()
                
                elif mut_choice < 0.2:
                    # Wrapper: Wrap current tree in a unary function (Complexity)
                    func = random.choice(['log', 'sigmoid', 'tanh'])
                    new_root = ExpressionNode(type='func', value=func, children=[tree.copy()])
                    genome.strategies[key] = new_root

                else:
                    # Standard Node/Subtree Mutation
                    mutated_tree = ExpressionEvolution.mutate_tree(
                        tree,
                        features=feature_list,
                        mutation_rate=rate,
                        inplace=True 
                    )
                    genome.strategies[key] = mutated_tree
                
                # Occasional Simplification/Pruning to prevent bloat
                if random.random() < 0.1:
                    genome.strategies[key] = ExpressionEvolution.simplify_tree(genome.strategies[key], max_size=30)

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

    @staticmethod
    @GeneticRegistry.register_mutation(GeneticKey.GUIDED_MUTATION)
    def guided_mutation(genome: Genome, ctx: EvolutionContext) -> Genome:
        """
        Smart mutation that encourages known good patterns:
        - Ensures 'semantic_similarity' is present in movement.
        - Ensures 'pheromone_repulsion' (diversity) is occasionally injected.
        - Prevents 'destructive' loss of key features.
        """
        tau = 0.2
        genome.mutation_rate = genome.mutation_rate * np.exp(tau * np.random.normal(0, 1))
        genome.mutation_rate = max(0.01, min(0.5, genome.mutation_rate))
        rate = genome.mutation_rate * ctx.global_mutation_multiplier

        # 1. Standard Parameter Jitter (Same as expression_tree_mutation)
        for key, val in genome.params.items():
            if random.random() < rate:
                if random.random() < 0.8: # Fine tuning
                    if isinstance(val, int):
                        delta = int(round(random.gauss(0, 1.5)))
                        genome.params[key] = max(1, val + delta)
                    elif isinstance(val, float):
                        genome.params[key] = max(0.001, min(0.999, val * random.gauss(1.0, 0.1)))
                else: # Resample
                     ranges = ctx.config.get('param_ranges', {})
                     if key in ranges:
                         min_v, max_v = ranges[key]
                         if isinstance(min_v, int):
                             genome.params[key] = random.randint(min_v, max_v)
                         else:
                             genome.params[key] = random.uniform(min_v, max_v)

        # 2. Guided Tree Mutation
        for key, tree in genome.strategies.items():
            if random.random() < rate:
                feature_list = ctx.expression_features.get(key)
                if not feature_list:
                    if "movement" in key: feature_list = ctx.expression_features.get("movement")
                    elif "deposit" in key: feature_list = ctx.expression_features.get("deposit")
                    elif "ranking" in key: feature_list = ctx.expression_features.get("ranking")
                
                # Check for critical features
                all_nodes = ExpressionEvolution.get_all_nodes(tree)
                has_semantic = any(n.value == 'semantic_similarity' for n in all_nodes)
                has_pheromone = any(n.value == 'pheromone_repulsion' for n in all_nodes)
                
                # A) Injection Logic (If missing, strong chance to add)
                injected = False
                if "movement" in key:
                    # If missing semantic, 50% chance to force inject it
                    if not has_semantic and random.random() < 0.5:
                        # Wrap: (Current + semantic_similarity) / 2
                        new_node = ExpressionNode("op", "+", [
                            tree.copy(), 
                            ExpressionNode("feature", "semantic_similarity")
                        ])
                        genome.strategies[key] = ExpressionNode("op", "/", [
                             new_node,
                             ExpressionNode("const", 2.0)
                        ])
                        injected = True
                    
                    # If missing diversity, 30% chance to inject
                    elif not has_pheromone and random.random() < 0.3:
                         # Wrap: Current * pheromone_repulsion
                        genome.strategies[key] = ExpressionNode("op", "*", [
                            tree.copy(),
                            ExpressionNode("feature", "pheromone_repulsion")
                        ])
                        injected = True
                
                # B) Standard Mutation (if not injected)
                if not injected:
                    # If we have critical features, we want to be CAREFUL not to delete them.
                    # We use standard mutation but might revert if it loses the critical feature.
                    original_tree = tree.copy()
                    
                    mutated_tree = ExpressionEvolution.mutate_tree(
                        tree, features=feature_list, mutation_rate=rate, inplace=True
                    )
                    
                    # Verification check
                    if has_semantic:
                         new_nodes = ExpressionEvolution.get_all_nodes(mutated_tree)
                         if not any(n.value == 'semantic_similarity' for n in new_nodes):
                             # Revert! We lost the most important signal.
                             if random.random() < 0.8: # 80% chance to revert
                                 genome.strategies[key] = original_tree
                                 continue
                    
                    genome.strategies[key] = mutated_tree

        genome.clear_cache()
        return genome
    
