"""
Population initialization strategies for the evolutionary algorithm.

Contains standard (ramped half-and-half), shallow growth, seeded, and
baseline-seeded initialization strategies.
"""
import logging
import random
from typing import List

from ...types.config import EvolutionContext
from ...types.expressions import ExpressionEvolution, ExpressionNode
from ....interfaces.enums import GeneticKey
from ...types.genome import Genome
from ...seed_genomes import get_all_seed_genomes

from .registry import GeneticRegistry
from .mutations import _randomize_all_params, _randomize_ratios

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registered initialization operators
# ---------------------------------------------------------------------------

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
        params = _randomize_all_params(ctx)

        # Assign Trees
        strategies = {}
        strategies["ranking"] = ranking_trees[i]

        # Randomize Group Ratios using helper
        group_ratios = _randomize_ratios(ctx, n_groups)

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
        params = _randomize_all_params(ctx)

        # 2. Strategies (Generated on the fly per genome, shallow)
        strategies = {}

        # Ranking (Depth 2, Grow)
        strategies["ranking"] = ExpressionEvolution.random_tree(
            features=ctx.expression_features["ranking"],
            max_depth=2,
            method='grow'
        )

        group_ratios = _randomize_ratios(ctx, n_groups)
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
        random_pop = standard_initialization(ctx, remaining)
        # Fix IDs to avoid collision
        for i, g in enumerate(random_pop):
            g.id = f"gen0_rand_{i}"
        population.extend(random_pop)

    return population


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
        random_pop = standard_initialization(ctx, remaining)
        for i, g in enumerate(random_pop):
            g.id = f"gen0_rand_{i}"
        population.extend(random_pop)

    return population
