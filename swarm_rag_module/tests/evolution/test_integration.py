# tests/evolution/test_integration.py
"""Integration tests for evolution speedup and convergence improvements."""
import pytest
from swarm_rag.evolution.types.genome import (
    Genome,
    FIXED_PARAMS,
    EVOLVABLE_PARAM_RANGES,
    create_random_genome,
)
from swarm_rag.evolution.types.config import (
    EvolutionConfig,
    EvolutionContext,
    GeneticConfig,
    MapElitesConfig,
    ResourceConfig,
)
from swarm_rag.evolution.execution.evaluator import DEFAULT_EARLY_EXIT_THRESHOLD
from swarm_rag.evolution.seed_genomes import SEED_GENOMES, get_all_seed_genomes
from swarm_rag.evolution.focused_mutation import (
    identify_weakest_metric,
    get_mutation_focus,
    apply_focused_mutation,
    METRIC_TO_PARAM_MAPPING,
)
from swarm_rag.evolution.types.fitness_results import FitnessResult
from swarm_rag.evolution.execution.strategies import GeneticRegistry, GeneticStrategies
from swarm_rag.interfaces.enums import GeneticKey


class TestEvolutionImprovementsIntegration:
    """Integration tests for all evolution improvements."""

    def test_fixed_params_applied_in_seeded_initialization(self):
        """Seeded initialization should use fixed params."""
        config = EvolutionConfig(
            genetic=GeneticConfig(creation_strategy="seeded_initialization")
        )
        ctx = EvolutionContext(config=config)
        ctx.expression_features = {
            "movement": ["semantic_similarity", "node_centrality"],
            "deposit": ["semantic_similarity", "flat"],
            "ranking": ["semantic_rank", "percentage_visited"],
        }

        creation_fn = GeneticRegistry.get_creation("seeded_initialization")
        population = creation_fn(ctx, count=10)

        # All seed genomes should have fixed param values
        for genome in population:
            for param, value in FIXED_PARAMS.items():
                assert genome.params.get(param) == value, (
                    f"Genome {genome.id} has wrong value for fixed param {param}"
                )

    def test_seed_genomes_within_evolvable_ranges(self):
        """Seed genomes should have evolvable params within tightened ranges."""
        for seed_genome in get_all_seed_genomes():
            for param, (low, high) in EVOLVABLE_PARAM_RANGES.items():
                if param in seed_genome.params:
                    val = seed_genome.params[param]
                    assert low <= val <= high, (
                        f"Seed {seed_genome.id} param {param}={val} out of [{low}, {high}]"
                    )

    def test_focused_mutation_strategy_registered(self):
        """FOCUSED_MUTATION should be registered and callable."""
        mutation_fn = GeneticRegistry.get_mutation(GeneticKey.FOCUSED_MUTATION)
        assert mutation_fn is not None
        assert callable(mutation_fn)

    def test_early_exit_threshold_configured(self):
        """Early exit threshold should be properly configured."""
        # Default threshold should be 0.30
        assert DEFAULT_EARLY_EXIT_THRESHOLD == 0.30

        # ResourceConfig should have the threshold
        config = ResourceConfig()
        assert config.early_exit_threshold == 0.30

        # Should be customizable
        config_custom = ResourceConfig(early_exit_threshold=0.45)
        assert config_custom.early_exit_threshold == 0.45

    def test_parallel_mutation_config_in_genetic_config(self):
        """GeneticConfig should have parallel_mutation_workers."""
        config = GeneticConfig()
        assert hasattr(config, "parallel_mutation_workers")
        assert config.parallel_mutation_workers >= 1

    def test_focused_mutation_affects_weak_metric_params(self):
        """Focused mutation should target params for weakest metric."""
        # Create fitness with low recall
        fitness = FitnessResult(
            quality_score=0.15,
            metrics={"recall_at_20": 0.15, "mrr": 0.50, "precision_at_20": 0.45},
        )

        focus = get_mutation_focus(fitness)

        # Should identify recall as weakest
        assert focus["weakest_metric"] == "recall_at_20"

        # Should suggest coverage-related params
        expected_params = METRIC_TO_PARAM_MAPPING["recall_at_20"]["params"]
        assert focus["params"] == expected_params

    def test_create_random_genome_respects_constraints(self):
        """create_random_genome should respect fixed and evolvable ranges."""
        for _ in range(10):
            genome = create_random_genome()

            # Fixed params should be exactly as defined
            for param, value in FIXED_PARAMS.items():
                assert genome.params[param] == value

            # Evolvable params should be within ranges
            for param, (low, high) in EVOLVABLE_PARAM_RANGES.items():
                val = genome.params[param]
                assert low <= val <= high

    def test_full_improvement_chain(self):
        """Test the full chain of improvements working together."""
        # 1. Create config with all improvements
        config = EvolutionConfig(
            genetic=GeneticConfig(
                creation_strategy="seeded_initialization",
                mutation_strategy="focused_mutation",
                parallel_mutation_workers=4,
            )
        )
        ctx = EvolutionContext(config=config)
        ctx.expression_features = {
            "movement": ["semantic_similarity", "node_centrality", "pheromone_repulsion"],
            "deposit": ["semantic_similarity", "flat"],
            "ranking": ["semantic_rank", "percentage_visited"],
        }

        # 2. Initialize population with seeds
        creation_fn = GeneticRegistry.get_creation("seeded_initialization")
        population = creation_fn(ctx, count=10)
        assert len(population) >= 5  # At least seed genomes

        # 3. Verify seeds have correct structure
        seed_count = sum(1 for g in population if g.id.startswith("gen0_seed"))
        assert seed_count >= 1, "Should have at least one seed genome"

        # 4. Get mutation function
        mutation_fn = GeneticRegistry.get_mutation(GeneticKey.FOCUSED_MUTATION)
        assert mutation_fn is not None

        # 5. Verify early exit threshold is configured
        assert DEFAULT_EARLY_EXIT_THRESHOLD == 0.30
