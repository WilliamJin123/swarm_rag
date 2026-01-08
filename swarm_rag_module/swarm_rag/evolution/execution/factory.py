import random
from typing import Dict, List, Callable

from ..types.expressions import ExpressionEvolution, ExpressionNode
from ..types.genome import DEFAULT_PARAMS, Genome, SwarmParams
from ..types.config import EvolutionConfigDict, EvolutionContext
from .strategies import GeneticRegistry


class GenomeFactory:
    def __init__(self, context: EvolutionContext):
        self.context = context
        self.config = context.config

    def create_population(self, count: int) -> List[Genome]:
        """
        Creates the initial population using the configured creation strategy.
        """
        # Resolve count if needed (usually passed from engine which takes it from config)
        if count is None:
            count = self.config['population_size']

        strategy_name = self.config.get("creation_strategy", "standard_initialization")
        creation_fn = GeneticRegistry.get_creation(strategy_name)

        return creation_fn(self.context, count)

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
        ranges = self.config['swarmrag_param_ranges']
        for key in ["n_agents", "steps", "decay", "initial_pool_size", "start_subset", "drop_inc"]:
            if key in ranges:
                min_v, max_v = ranges[key]
                if isinstance(min_v, int):
                    params[key] = random.randint(min_v, max_v)
                else:
                    params[key] = random.uniform(min_v, max_v)
        return params