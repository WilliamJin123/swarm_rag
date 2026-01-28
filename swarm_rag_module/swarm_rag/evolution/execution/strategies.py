
from typing import Callable, ClassVar, Dict, List, Union
import random
import math
import torch

from ..types.config import EvolutionContext, SwarmParamRanges
from ..types.expressions import ExpressionEvolution, ExpressionNode
from ...interfaces.registry import _MutationRegistry, _CrossoverRegistry, _SelectionRegistry, _CreationRegistry
from ...interfaces.enums import GeneticKey
from ..types.genome import Genome, DEFAULT_PARAMS, SwarmParams, FIXED_PARAMS
from ..seed_genomes import get_all_seed_genomes
from ..focused_mutation import apply_focused_mutation

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

    # --- HELPERS ---
    @staticmethod
    def _mix_params(child: Genome, parent2: Genome):
        """Helper to mix scalar parameters and group ratios uniformly."""
        for key in child.params:
            if key in parent2.params and random.random() > 0.5:
                child.params[key] = parent2.params[key]

        for key in child.group_ratios:
            if key in parent2.group_ratios and random.random() > 0.5:
                child.group_ratios[key] = parent2.group_ratios[key]

    @staticmethod
    def _mutate_params_standard(genome: Genome, ctx: EvolutionContext, rate: float):
        """Standard parameter mutation (Smart Jitter).

        Skips fixed parameters (FIXED_PARAMS) and uses tightened ranges
        from SwarmParamRanges (single source of truth) for evolvable parameters.
        """
        evolvable_ranges = SwarmParamRanges().to_evolvable_dict()
        for key, val in genome.params.items():
            # Skip fixed parameters - they should never be mutated
            if key in FIXED_PARAMS:
                # Ensure fixed params have correct values
                genome.params[key] = FIXED_PARAMS[key]
                continue

            if random.random() < rate:
                # 80% chance: Fine-tuning (Small Gaussian jitter)
                if random.random() < 0.8:
                    if isinstance(val, int):
                        delta = int(round(random.gauss(0, 1.5)))  # +/- 1 or 2 usually
                        new_val = max(1, val + delta)
                        # Clamp to evolvable range if defined
                        if key in evolvable_ranges:
                            min_v, max_v = evolvable_ranges[key]
                            new_val = max(min_v, min(max_v, new_val))
                        genome.params[key] = new_val
                    elif isinstance(val, float):
                        # +/- 10% relative change
                        factor = random.gauss(1.0, 0.1)
                        new_val = val * factor
                        # Clamp to evolvable range if defined
                        if key in evolvable_ranges:
                            min_v, max_v = evolvable_ranges[key]
                            new_val = max(min_v, min(max_v, new_val))
                        else:
                            new_val = max(0.001, min(0.999, new_val))
                        genome.params[key] = new_val

                # 20% chance: Exploration (Re-sample from range)
                else:
                    # Prefer SwarmParamRanges, fallback to config ranges
                    if key in evolvable_ranges:
                        min_v, max_v = evolvable_ranges[key]
                        if isinstance(min_v, int):
                            genome.params[key] = random.randint(min_v, max_v)
                        else:
                            genome.params[key] = random.uniform(min_v, max_v)
                    else:
                        ranges = ctx.config.genetic.param_ranges
                        if hasattr(ranges, key):
                            min_v, max_v = getattr(ranges, key)
                            if isinstance(min_v, int):
                                genome.params[key] = random.randint(min_v, max_v)
                            else:
                                genome.params[key] = random.uniform(min_v, max_v)

    @staticmethod
    def _mutate_ratios_standard(genome: Genome, rate: float):
        """Standard group ratio mutation (Smart Jitter)."""
        for key, val in genome.group_ratios.items():
            if random.random() < rate:
                # Jitter ratio
                genome.group_ratios[key] = max(0.05, min(1.0, val * random.uniform(0.8, 1.2)))

    @staticmethod
    def _randomize_all_params(ctx: EvolutionContext) -> SwarmParams:
        """Helper to fully randomize evolvable parameters within tightened ranges.

        Uses FIXED_PARAMS for fixed parameters and SwarmParamRanges (single source
        of truth) for evolvable parameters. Falls back to ctx.config.genetic.param_ranges
        for parameters not defined in either.
        """
        # Start with default params
        params = DEFAULT_PARAMS.copy()

        # Apply fixed parameters (never randomized)
        params.update(FIXED_PARAMS)

        # Randomize evolvable parameters from SwarmParamRanges (single source of truth)
        evolvable_ranges = SwarmParamRanges().to_evolvable_dict()
        for key, (min_v, max_v) in evolvable_ranges.items():
            if isinstance(min_v, int):
                params[key] = random.randint(min_v, max_v)
            else:
                params[key] = random.uniform(min_v, max_v)

        # Fallback for any params not in evolvable_ranges or FIXED_PARAMS
        ranges = ctx.config.genetic.param_ranges
        for key in params.keys():
            if key not in FIXED_PARAMS and key not in evolvable_ranges:
                if hasattr(ranges, key):
                    min_v, max_v = getattr(ranges, key)
                    if isinstance(min_v, int):
                        params[key] = random.randint(int(min_v), int(max_v))
                    else:
                        params[key] = random.uniform(min_v, max_v)

        return params

    @staticmethod
    def _randomize_ratios(ctx: EvolutionContext, n_groups: int) -> Dict[str, float]:
        """Helper to fully randomize group ratios."""
        # Default range for group ratios (not in SwarmParamRanges)
        min_r, max_r = 0.1, 1.0
        return {f"g{i}": random.uniform(min_r, max_r) for i in range(n_groups)}

    @staticmethod
    def _resolve_feature_list(key: str, ctx: EvolutionContext) -> List[str]:
        """
        Resolves the appropriate feature list for a strategy key.

        Handles patterns like 'ranking', 'gN_movement', 'gN_deposit'.

        Args:
            key: Strategy key (e.g., "g0_movement", "ranking")
            ctx: Evolution context containing expression_features

        Returns:
            List of valid feature names for this strategy type
        """
        # Direct match first
        feature_list = ctx.expression_features.get(key)
        if feature_list:
            return feature_list

        # Pattern match for group strategies
        if key.endswith("_movement") or "movement" in key:
            return ctx.expression_features.get("movement", [])
        elif key.endswith("_deposit") or "deposit" in key:
            return ctx.expression_features.get("deposit", [])
        elif key == "ranking" or "ranking" in key:
            return ctx.expression_features.get("ranking", [])

        # Fallback: return empty list
        return []

    # --- SELECTION ---

    @staticmethod
    @GeneticRegistry.register_selection(GeneticKey.TOURNAMENT)
    def tournament_selection(ctx: EvolutionContext, k: int = 1) -> List[Genome]:
        """
        Selects 'k' parents using Tournament logic.
        """
        tourn_size = ctx.config.genetic.selection_k
        pop_size = len(ctx.population)
        winners = []
        for _ in range(k):
            indices = torch.randint(0, pop_size, (tourn_size,))
            contestants = [ctx.population[i] for i in indices.tolist()]
            # Select winner by FITNESS, not index
            winner = max(contestants, key=lambda g: g.fitness)
            winners.append(winner)

        return winners

    @staticmethod
    @GeneticRegistry.register_selection(GeneticKey.BOLTZMANN)
    def boltzmann_selection(ctx: EvolutionContext, k: int = 1) -> List[Genome]:
        """
        Boltzmann (Softmax) Selection with Adaptive Temperature.
        - Probability P(i) ~ exp(Fitness(i) / T)
        - Adapt T based on population diversity:
            * Low diversity -> Increase T (Heat up to explore)
            * High diversity -> Decrease T (Cool down to exploit)
        
        Optimized to use torch.softmax and float32 for faster computation.
        """
        boltzmann_cfg = ctx.config.genetic.boltzmann

        # Initialize Temperature (if first run)
        if ctx.generation == 0 and ctx.current_temperature == 1.0:
            ctx.current_temperature = boltzmann_cfg.temperature

        # Prepare Scores as float32 tensor for speed (sufficient precision)
        scores = torch.tensor([g.fitness.quality_score for g in ctx.population], dtype=torch.float32)

        # T controls the "pressure".
        # T -> inf: Uniform random
        # T -> 0: Deterministic max
        T = ctx.current_temperature
        T = max(1e-4, T)

        # Use torch.softmax for numerical stability and vectorization
        # softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        # Here we want exp(scores / T) / sum(exp(scores / T))
        probs = torch.softmax(scores / T, dim=0)

        # Select
        selection_indices = torch.multinomial(probs, num_samples=k, replacement=True)
        selected = [ctx.population[i] for i in selection_indices.tolist()]

        # Update Temperature (Adaptive)
        if boltzmann_cfg.adaptive:
            mean_score = scores.mean().item()
            # Calculate Coefficient of Variation (CV) = std / mean
            # Using tensor operations for speed
            if mean_score > 1e-6:
                diversity_cv = (scores.std() / mean_score).item()
            else:
                diversity_cv = 0.0

            cooling_factor = boltzmann_cfg.alpha
            heating_factor = 1.0 / cooling_factor

            min_T = boltzmann_cfg.min_temp
            max_T = boltzmann_cfg.max_temp
            diversity_threshold = boltzmann_cfg.diversity_threshold

            # Heuristic: If relative diversity is low, we are stagnating -> Heat up
            if diversity_cv < diversity_threshold:
                ctx.current_temperature *= heating_factor
            else:
                # Otherwise -> Cool down (Annealing)
                ctx.current_temperature *= cooling_factor

            # Clamp temperature within bounds using standard python math for scalar clamping
            # (faster than wrapping in tensor for a single scalar value)
            ctx.current_temperature = max(min_T, min(max_T, ctx.current_temperature))

        return selected

    # --- CROSSOVER ---

    @staticmethod
    @GeneticRegistry.register_crossover(GeneticKey.UNIFORM_PARAMETER_MIX)
    def uniform_parameter_mix(parent1: Genome, parent2: Genome, ctx: EvolutionContext) -> Genome:
        """Mixes traits 50/50. """
        child = parent1.copy()
        child.mutation_rate = (parent1.mutation_rate + parent2.mutation_rate) / 2.0
        
        GeneticStrategies._mix_params(child, parent2)

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
        GeneticStrategies._mix_params(child, parent2)

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
        GeneticStrategies._mix_params(child, parent2)

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
        max_d = ctx.config.genetic.expr_max_depth
        n_groups = ctx.config.genetic.n_agent_groups

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

        base_rate = ctx.config.genetic.base_mutation_rate

        population = []
        for i in range(count):
            # Randomize Global Params using helper
            params = GeneticStrategies._randomize_all_params(ctx)

            # Assign Trees
            strategies = {}
            strategies["ranking"] = ranking_trees[i]
            
            # Randomize Group Ratios using helper
            group_ratios = GeneticStrategies._randomize_ratios(ctx, n_groups)
            
            for g_idx in range(n_groups):
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
        n_groups = ctx.config.genetic.n_agent_groups
        base_rate = ctx.config.genetic.base_mutation_rate
        
        population = []
        for i in range(count):
            # 1. Params using helper
            params = GeneticStrategies._randomize_all_params(ctx)

            # 2. Strategies (Generated on the fly per genome, shallow)
            strategies = {}
            
            # Ranking (Depth 2, Grow)
            strategies["ranking"] = ExpressionEvolution.random_tree(
                features=ctx.expression_features["ranking"], 
                max_depth=2, 
                method='grow'
            )

            group_ratios = GeneticStrategies._randomize_ratios(ctx, n_groups)
            for g_idx in range(n_groups):
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
        Warm-start initialization using known good genome configurations.

        Uses pre-defined seed genomes from seed_genomes.py which provide
        a strong baseline and reduce wasted generations discovering
        basic effective strategies.

        Process:
        1. Include all seed genomes (up to available count)
        2. Fill remainder with random genomes

        This approach is more effective than the old manual seed creation
        because the seeds are based on empirically validated configurations.
        """
        population = []

        # Get all pre-defined seed genomes
        seed_genomes = get_all_seed_genomes()

        # Include seed genomes up to count
        for i, seed in enumerate(seed_genomes):
            if i >= count:
                break
            # Update ID to indicate generation 0
            seed.id = f"gen0_{seed.id}"
            population.append(seed)

        # Fill remainder with standard random initialization
        remaining = count - len(population)
        if remaining > 0:
            random_pop = GeneticStrategies.standard_initialization(ctx, remaining)
            # Fix IDs to avoid collision
            for i, g in enumerate(random_pop):
                g.id = f"gen0_rand_{i}"
            population.extend(random_pop)

        return population

    @staticmethod
    @GeneticRegistry.register_creation(GeneticKey.BASELINE_SEEDED_INITIALIZATION)
    def baseline_seeded_initialization(ctx: EvolutionContext, count: int) -> List[Genome]:
        """
        Seeds the population with genomes matching the baseline from test_n_q.py.

        The baseline config that achieves >20%:
        - semantic_similarity_unnormalized (0.5 weight)
        - stark_centrality (0.2 weight)
        - pheromone_repulsion (0.25 weight)
        - random_jitter (0.05 weight)
        - n_agents=25, steps=5, decay=0.5, initial_pool_size=30

        This ensures evolution starts from a known-good solution.
        """
        population = []
        n_groups = ctx.config.genetic.n_agent_groups

        def make_expression_tree(features_weights: List[tuple]) -> ExpressionNode:
            """
            Build expression tree from (feature, weight) pairs.
            Result: (f1 * w1) + (f2 * w2) + ...
            """
            if not features_weights:
                return ExpressionNode("const", 1.0)

            terms = []
            for feature, weight in features_weights:
                # feature * weight
                term = ExpressionNode("op", "*", [
                    ExpressionNode("feature", feature),
                    ExpressionNode("const", weight)
                ])
                terms.append(term)

            # Combine with + operator
            if len(terms) == 1:
                return terms[0]

            result = terms[0]
            for term in terms[1:]:
                result = ExpressionNode("op", "+", [result, term])

            return result

        # Baseline params from test_n_q.py
        baseline_params = {
            "n_agents": 25,
            "steps": 5,
            "decay": 0.5,
            "initial_pool_size": 30,
            "start_subset": 10,
        }

        # Baseline movement strategy
        # 0.5 * semantic_similarity_unnormalized + 0.2 * stark_centrality +
        # 0.25 * pheromone_repulsion + 0.05 * random_jitter
        baseline_movement = [
            ("semantic_similarity_unnormalized", 0.5),
            ("stark_centrality", 0.2),
            ("pheromone_repulsion", 0.25),
            ("random_jitter", 0.05),
        ]

        # 1. Pure Baseline (exact match)
        strategies = {}
        strategies["ranking"] = ExpressionNode("op", "+", [
            ExpressionNode("op", "*", [
                ExpressionNode("feature", "semantic_rank"),
                ExpressionNode("const", 0.9)
            ]),
            ExpressionNode("op", "*", [
                ExpressionNode("feature", "percentage_visited"),
                ExpressionNode("const", 0.1)
            ])
        ])

        group_ratios = {}
        for g in range(n_groups):
            group_ratios[f"g{g}"] = 1.0 / n_groups
            strategies[f"g{g}_movement"] = make_expression_tree(baseline_movement)
            strategies[f"g{g}_deposit"] = ExpressionNode("feature", "flat")

        params = baseline_params.copy()
        population.append(Genome(
            id="gen0_baseline_exact",
            params=params,
            group_ratios=group_ratios,
            strategies=strategies,
            mutation_rate=0.15  # Lower mutation to preserve good genes
        ))

        # 2. Baseline variant with unnormalized semantic for deposit
        if count > 1:
            strategies2 = {}
            strategies2["ranking"] = ExpressionNode("feature", "semantic_rank")
            group_ratios2 = {}
            for g in range(n_groups):
                group_ratios2[f"g{g}"] = 1.0 / n_groups
                strategies2[f"g{g}_movement"] = make_expression_tree(baseline_movement)
                strategies2[f"g{g}_deposit"] = ExpressionNode("feature", "semantic_unnormalized")

            population.append(Genome(
                id="gen0_baseline_variant1",
                params=baseline_params.copy(),
                group_ratios=group_ratios2,
                strategies=strategies2,
                mutation_rate=0.15
            ))

        # 3. Baseline without stark_centrality (fallback for graphs without it)
        if count > 2:
            no_stark_movement = [
                ("semantic_similarity_unnormalized", 0.6),
                ("pheromone_repulsion", 0.3),
                ("random_jitter", 0.1),
            ]
            strategies3 = {}
            strategies3["ranking"] = ExpressionNode("feature", "semantic_rank")
            group_ratios3 = {}
            for g in range(n_groups):
                group_ratios3[f"g{g}"] = 1.0 / n_groups
                strategies3[f"g{g}_movement"] = make_expression_tree(no_stark_movement)
                strategies3[f"g{g}_deposit"] = ExpressionNode("feature", "flat")

            population.append(Genome(
                id="gen0_baseline_no_stark",
                params=baseline_params.copy(),
                group_ratios=group_ratios3,
                strategies=strategies3,
                mutation_rate=0.20
            ))

        # 4. Baseline with more exploration (higher pheromone weight)
        if count > 3:
            explore_movement = [
                ("semantic_similarity_unnormalized", 0.4),
                ("stark_centrality", 0.15),
                ("pheromone_repulsion", 0.35),
                ("random_jitter", 0.1),
            ]
            strategies4 = {}
            strategies4["ranking"] = ExpressionNode("op", "+", [
                ExpressionNode("op", "*", [
                    ExpressionNode("feature", "semantic_rank"),
                    ExpressionNode("const", 0.8)
                ]),
                ExpressionNode("op", "*", [
                    ExpressionNode("feature", "percentage_visited"),
                    ExpressionNode("const", 0.2)
                ])
            ])
            group_ratios4 = {}
            for g in range(n_groups):
                group_ratios4[f"g{g}"] = 1.0 / n_groups
                strategies4[f"g{g}_movement"] = make_expression_tree(explore_movement)
                strategies4[f"g{g}_deposit"] = ExpressionNode("feature", "exploration_bonus")

            params4 = baseline_params.copy()
            params4["steps"] = 7  # More steps for exploration
            params4["decay"] = 0.8  # Slower decay

            population.append(Genome(
                id="gen0_baseline_explore",
                params=params4,
                group_ratios=group_ratios4,
                strategies=strategies4,
                mutation_rate=0.20
            ))

        # 5. Baseline with more agents
        if count > 4:
            strategies5 = {}
            strategies5["ranking"] = ExpressionNode("feature", "semantic_rank")
            group_ratios5 = {}
            for g in range(n_groups):
                group_ratios5[f"g{g}"] = 1.0 / n_groups
                strategies5[f"g{g}_movement"] = make_expression_tree(baseline_movement)
                strategies5[f"g{g}_deposit"] = ExpressionNode("feature", "flat")

            params5 = baseline_params.copy()
            params5["n_agents"] = 30
            params5["initial_pool_size"] = 40

            population.append(Genome(
                id="gen0_baseline_more_agents",
                params=params5,
                group_ratios=group_ratios5,
                strategies=strategies5,
                mutation_rate=0.15
            ))

        # Fill remainder with standard random initialization
        remaining = count - len(population)
        if remaining > 0:
            random_pop = GeneticStrategies.standard_initialization(ctx, remaining)
            for i, g in enumerate(random_pop):
                g.id = f"gen0_rand_{i}"
            population.extend(random_pop)

        return population

    # --- MUTATION ---

    @staticmethod
    @GeneticRegistry.register_mutation(GeneticKey.EXPRESSION_TREE_MUTATION)
    def expression_tree_mutation(genome: Genome, ctx: EvolutionContext) -> Genome:
        tau = 0.2
        # Optimized: Use python math/random for scalar operations instead of torch to avoid kernel launch overhead
        log_mutation_factor = tau * random.gauss(0, 1)
        genome.mutation_rate *= math.exp(log_mutation_factor)
        genome.mutation_rate = max(0.01, min(0.5, genome.mutation_rate))
        rate = genome.mutation_rate * ctx.global_mutation_multiplier

        # 1. Parameter Mutation (Smart Jitter)
        GeneticStrategies._mutate_params_standard(genome, ctx, rate)

        # 2. Group Ratio Mutation
        GeneticStrategies._mutate_ratios_standard(genome, rate)

        # 3. Strategy Tree Mutation
        for key, tree in genome.strategies.items():
            if random.random() < rate:
                feature_list = GeneticStrategies._resolve_feature_list(key, ctx)

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
        # Boost rate
        genome.mutation_rate = 0.4 # Lock to high rate
        rate = genome.mutation_rate * ctx.global_mutation_multiplier
        ranges = ctx.config.genetic.param_ranges

        # Aggressive Parameter Resampling
        for key in genome.params.keys():
            if random.random() < rate:
                if hasattr(ranges, key):
                    # 50% chance to purely resample from global range (Big Jump)
                    if random.random() < 0.5:
                        min_v, max_v = getattr(ranges, key)
                        if isinstance(min_v, int):
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

        # Aggressive Group Ratio Mutation
        for key, val in genome.group_ratios.items():
            if random.random() < rate:
                # Larger Jitter for aggressive
                genome.group_ratios[key] = max(0.05, min(1.0, val * random.uniform(0.6, 1.4)))

        # Aggressive Tree Mutation
        for key, tree in genome.strategies.items():
            if random.random() < rate:
                feature_list = GeneticStrategies._resolve_feature_list(key, ctx)

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
        - Ensures semantic similarity features are present in movement.
        - Ensures 'pheromone_repulsion' (diversity) is occasionally injected.
        - Protects critical features from being lost during mutation.

        Critical features to protect:
        - semantic_similarity / semantic_similarity_unnormalized
        - stark_centrality (when present)
        - pheromone_repulsion
        """
        tau = 0.2
        # Optimized: Use python math/random for scalar operations instead of torch
        genome.mutation_rate *= math.exp(tau * random.gauss(0, 1))
        genome.mutation_rate = max(0.01, min(0.5, genome.mutation_rate))
        rate = genome.mutation_rate * ctx.global_mutation_multiplier

        # Critical feature names (protect these during mutation)
        SEMANTIC_FEATURES = {'semantic_similarity', 'semantic_similarity_unnormalized'}
        CENTRALITY_FEATURES = {'stark_centrality', 'node_centrality'}
        DIVERSITY_FEATURES = {'pheromone_repulsion'}

        # Standard Parameter Jitter (Same as expression_tree_mutation)
        GeneticStrategies._mutate_params_standard(genome, ctx, rate)

        # Group Ratio Mutation (Standard Jitter)
        GeneticStrategies._mutate_ratios_standard(genome, rate)

        # Guided Tree Mutation
        for key, tree in genome.strategies.items():
            if random.random() < rate:
                feature_list = GeneticStrategies._resolve_feature_list(key, ctx)

                # Check for critical features
                all_nodes = ExpressionEvolution.get_all_nodes(tree)
                node_values = {n.value for n in all_nodes}

                has_semantic = bool(node_values & SEMANTIC_FEATURES)
                has_centrality = bool(node_values & CENTRALITY_FEATURES)
                has_pheromone = bool(node_values & DIVERSITY_FEATURES)

                # A) Injection Logic (If missing critical features, inject them)
                injected = False
                if "movement" in key:
                    # If missing semantic signal, 50% chance to inject
                    if not has_semantic and random.random() < 0.5:
                        # Prefer unnormalized for baseline compatibility
                        semantic_feature = "semantic_similarity_unnormalized"
                        if semantic_feature not in feature_list:
                            semantic_feature = "semantic_similarity"

                        # Wrap: (Current + semantic * 0.5) / 1.5
                        new_node = ExpressionNode("op", "+", [
                            tree.copy(),
                            ExpressionNode("op", "*", [
                                ExpressionNode("feature", semantic_feature),
                                ExpressionNode("const", 0.5)
                            ])
                        ])
                        genome.strategies[key] = ExpressionNode("op", "/", [
                             new_node,
                             ExpressionNode("const", 1.5)
                        ])
                        injected = True

                    # If missing diversity, 30% chance to inject
                    elif not has_pheromone and random.random() < 0.3:
                        # Wrap: Current * (0.7 + 0.3 * pheromone_repulsion)
                        pheromone_term = ExpressionNode("op", "+", [
                            ExpressionNode("const", 0.7),
                            ExpressionNode("op", "*", [
                                ExpressionNode("const", 0.3),
                                ExpressionNode("feature", "pheromone_repulsion")
                            ])
                        ])
                        genome.strategies[key] = ExpressionNode("op", "*", [
                            tree.copy(),
                            pheromone_term
                        ])
                        injected = True

                    # If has centrality but missing semantic, 40% chance to add both
                    elif has_centrality and not has_semantic and random.random() < 0.4:
                        semantic_feature = "semantic_similarity_unnormalized"
                        if semantic_feature not in feature_list:
                            semantic_feature = "semantic_similarity"

                        # Add semantic weighted sum
                        genome.strategies[key] = ExpressionNode("op", "+", [
                            ExpressionNode("op", "*", [
                                tree.copy(),
                                ExpressionNode("const", 0.4)
                            ]),
                            ExpressionNode("op", "*", [
                                ExpressionNode("feature", semantic_feature),
                                ExpressionNode("const", 0.6)
                            ])
                        ])
                        injected = True

                # B) Standard Mutation (if not injected)
                if not injected:
                    # Save original tree for potential revert
                    original_tree = tree.copy()

                    mutated_tree = ExpressionEvolution.mutate_tree(
                        tree, features=feature_list, mutation_rate=rate, inplace=True
                    )

                    # Verification: Check if we lost critical features
                    new_nodes = ExpressionEvolution.get_all_nodes(mutated_tree)
                    new_values = {n.value for n in new_nodes}

                    # Revert if we lost semantic features (very important!)
                    if has_semantic and not (new_values & SEMANTIC_FEATURES):
                        if random.random() < 0.85:  # 85% chance to revert
                            genome.strategies[key] = original_tree
                            continue

                    # Revert if we lost centrality (moderately important)
                    if has_centrality and not (new_values & CENTRALITY_FEATURES):
                        if random.random() < 0.6:  # 60% chance to revert
                            genome.strategies[key] = original_tree
                            continue

                    genome.strategies[key] = mutated_tree

        genome.clear_cache()
        return genome

    @staticmethod
    @GeneticRegistry.register_mutation(GeneticKey.FOCUSED_MUTATION)
    def focused_mutation(genome: Genome, ctx: EvolutionContext) -> Genome:
        """
        Metric-aware mutation that targets the weakest metric.

        Analyzes the genome's fitness profile to identify which metric
        (recall, MRR, precision) is weakest, then focuses mutations on
        parameters most likely to improve that metric.

        This provides more directed evolution than random parameter
        mutation, potentially accelerating convergence.
        """
        # Get fitness if available
        fitness = genome.fitness if hasattr(genome, 'fitness') and genome.fitness else None

        # Apply focused mutation using the dedicated module
        genome = apply_focused_mutation(
            genome=genome,
            ctx=ctx,
            fitness=fitness,
            mutation_rate=genome.mutation_rate * ctx.global_mutation_multiplier,
        )

        # Also apply some tree mutation for exploration
        rate = genome.mutation_rate * ctx.global_mutation_multiplier * 0.5  # Reduced rate

        for key, tree in genome.strategies.items():
            if random.random() < rate:
                feature_list = GeneticStrategies._resolve_feature_list(key, ctx)
                mutated_tree = ExpressionEvolution.mutate_tree(
                    tree, features=feature_list, mutation_rate=rate, inplace=True
                )
                genome.strategies[key] = mutated_tree

        genome.clear_cache()
        return genome