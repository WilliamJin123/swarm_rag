import random
from typing import Dict, List, Callable

from ..types.expressions import ExpressionEvolution, ExpressionNode
from ..types.genome import DEFAULT_PARAMS, Genome, SwarmParams
from ..types.config import EvolutionContext
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
            count = self.config.map_elites.batch_size

        strategy_name = self.config.genetic.creation_strategy
        creation_fn = GeneticRegistry.get_creation(strategy_name)

        return creation_fn(self.context, count)

    def create_single(
        self,
        genome_id: str,
        strategies: Dict[str, ExpressionNode],
        group_ratios: Dict[str, float]
    ) -> Genome:
        """
        Creates a single genome with randomized global parameters.
        """
        params = self.initialize_parameters()
        base_rate = self.config.genetic.base_mutation_rate
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
        ranges = self.config.genetic.param_ranges

        # Map config field names to params keys
        range_mapping = {
            "n_agents": "n_agents",
            "steps": "steps",
            "decay": "decay",
            "initial_pool_size": "initial_pool_size",
            "start_subset": "start_subset",
            "drop_zone_inc": "drop_zone_inc",
        }

        for range_key, param_key in range_mapping.items():
            if hasattr(ranges, range_key):
                min_v, max_v = getattr(ranges, range_key)
                if isinstance(min_v, int):
                    params[param_key] = random.randint(min_v, max_v)
                else:
                    params[param_key] = random.uniform(min_v, max_v)
        return params