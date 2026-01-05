import random
from typing import Dict, List, Callable

from ..types.expressions import ExpressionEvolution, ExpressionNode
from ..types.genome import DEFAULT_PARAMS, Genome, SwarmParams
from ..types.config import EvolutionConfigDict, EvolutionContext


class GenomeFactory:
    def __init__(self, config: EvolutionConfigDict, context: EvolutionContext):
        self.config = config
        self.context = context

    def create_population(self, count: int) -> List[Genome]:
        """Default batch creation logic."""
        count = self.config['population_size']
        ranges = self.config['param_ranges']
        max_d = self.config['expr_max_depth']
        n_groups = self.config["n_agent_groups"]

        strat_trees = {}
        for strat_type in ["movement", "deposit"]:
            features = self.evo_context.expression_features[strat_type]
            total_trees = count * n_groups
            
            flat_list = ExpressionEvolution.generate_ramped_half_and_half(
                features=features,
                population_size=total_trees,
                max_depth=max_d
            )
            strat_trees[strat_type] = flat_list

        ranking_features = self.evo_context.expression_features["ranking"]
        ranking_trees = ExpressionEvolution.generate_ramped_half_and_half(
            features=ranking_features,
            population_size=count,
            max_depth=max_d
        )

        base_rate = self.config['base_mutation_rate']

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

    def create_single(self, 
        genome_id: str, 
        strategies: Dict[str, ExpressionNode], 
        group_ratios: Dict[str, float]
    ) -> Genome:
        """
        Creates a single genome with randomized global parameters.
        """
        params = self.initialize_parameters()
        base_rate = self.config['base_mutation_rate']
        start_rate = max(0.01, min(0.5, random.gauss(base_rate, 0.05)))

        return Genome(
            id=genome_id,
            params=params,
            group_ratios=group_ratios,
            strategies=strategies,
            mutation_rate=start_rate
        )

    def initialize_parameters(self) -> SwarmParams:
        """Randomizes global Swarm parameters based on config ranges."""
        params = DEFAULT_PARAMS.copy()
        ranges = self.config['param_ranges']
        for key in ["n_agents", "steps", "decay", "initial_pool_size", "start_subset", "drop_inc"]:
            if key in ranges:
                min_v, max_v = ranges[key]
                if isinstance(min_v, int):
                    params[key] = random.randint(min_v, max_v)
                else:
                    params[key] = random.uniform(min_v, max_v)
        return params