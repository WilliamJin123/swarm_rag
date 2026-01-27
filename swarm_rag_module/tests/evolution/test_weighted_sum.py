"""
Tests for Dual-Mode Evolution System (Weighted Sum Mode)

Tests cover:
1. WeightTensors dataclass operations
2. MutationSigmas self-adaptive behavior
3. HeuristicFeatureConfig serialization
4. Genome dual-mode support
5. WeightedSumCompiler functionality
6. WeightedSumMutator operations
7. WeightedSumSeeder population generation
8. GenomeFactory mode selection
9. PopulationEvaluator compiler selection
10. Checkpoint serialization roundtrip
"""

import pytest
import torch
import tempfile
import os

from swarm_rag.evolution.types.config import (
    WeightTensors,
    MutationSigmas,
    HeuristicFeatureConfig,
    EvolutionConfig,
    EvolutionContext,
    GeneticConfig,
    STARK_FEATURES,
    GENERAL_FEATURES,
)
from swarm_rag.evolution.types.genome import Genome
from swarm_rag.evolution.types.fitness_results import FitnessResult
from swarm_rag.evolution.execution.weighted_sum import (
    WeightedSumCompiler,
    WeightedSumMutator,
    WeightedSumSeeder,
    compute_movement_scores_batched,
    assign_scores_to_agents,
)
from swarm_rag.evolution.execution.factory import GenomeFactory
from swarm_rag.evolution.execution.strategies import GeneticRegistry
from swarm_rag.evolution.storage.run_manager import RunManager


# =============================================================================
# WeightTensors Tests
# =============================================================================

class TestWeightTensors:
    """Tests for WeightTensors dataclass."""

    def test_default_initialization(self):
        """Test default WeightTensors creation."""
        wt = WeightTensors()
        assert wt.n_groups == 1
        assert wt.total_params > 0
        assert isinstance(wt.movement_weights, torch.Tensor)

    def test_custom_initialization(self):
        """Test WeightTensors with custom values."""
        wt = WeightTensors(
            movement_weights=torch.randn(3, 4),
            movement_biases=torch.zeros(3),
            deposit_weights=torch.randn(3, 2),
            deposit_biases=torch.zeros(3),
            ranking_weights=torch.randn(2),
            ranking_bias=0.5,
        )
        assert wt.n_groups == 3
        assert wt.movement_weights.shape == (3, 4)
        assert wt.ranking_bias == 0.5

    def test_to_device(self):
        """Test moving tensors to device."""
        wt = WeightTensors()
        wt_cpu = wt.to_device("cpu")
        assert wt_cpu.movement_weights.device.type == "cpu"

    def test_copy(self):
        """Test deep copy of WeightTensors."""
        wt = WeightTensors(
            movement_weights=torch.tensor([[1.0, 2.0]]),
            movement_biases=torch.tensor([0.5]),
            deposit_weights=torch.tensor([[3.0]]),
            deposit_biases=torch.tensor([0.1]),
            ranking_weights=torch.tensor([0.9]),
            ranking_bias=0.2,
        )
        wt_copy = wt.copy()

        # Modify original
        wt.movement_weights[0, 0] = 999.0

        # Copy should be unaffected
        assert wt_copy.movement_weights[0, 0] == 1.0

    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict."""
        wt = WeightTensors(
            movement_weights=torch.randn(2, 3),
            movement_biases=torch.randn(2),
            deposit_weights=torch.randn(2, 2),
            deposit_biases=torch.randn(2),
            ranking_weights=torch.randn(2),
            ranking_bias=0.123,
        )

        d = wt.to_dict()
        # Restore on CPU to match original tensors for comparison
        wt_restored = WeightTensors.from_dict(d, device="cpu")

        assert wt_restored.n_groups == wt.n_groups
        assert torch.allclose(wt_restored.movement_weights, wt.movement_weights)
        assert wt_restored.ranking_bias == pytest.approx(wt.ranking_bias)


# =============================================================================
# MutationSigmas Tests
# =============================================================================

class TestMutationSigmas:
    """Tests for MutationSigmas dataclass."""

    def test_default_values(self):
        """Test default sigma values."""
        ms = MutationSigmas()
        assert ms.weight_sigma == 0.10
        assert ms.bias_sigma == 0.05
        assert ms.tau == 0.1

    def test_adapt_changes_values(self):
        """Test that adapt() changes sigma values."""
        ms = MutationSigmas()
        original_weight_sigma = ms.weight_sigma

        # Run adaptation multiple times to ensure statistical change
        changed = False
        for _ in range(10):
            ms_adapted = ms.adapt()
            if ms_adapted.weight_sigma != original_weight_sigma:
                changed = True
                break

        assert changed, "Adaptation should change sigma values"

    def test_adapt_respects_bounds(self):
        """Test that adapt() respects min/max bounds."""
        ms = MutationSigmas(min_sigma=0.01, max_sigma=0.50)

        for _ in range(100):
            ms = ms.adapt()
            assert ms.weight_sigma >= ms.min_sigma
            assert ms.weight_sigma <= ms.max_sigma

    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict."""
        ms = MutationSigmas(weight_sigma=0.15, bias_sigma=0.08)

        d = ms.to_dict()
        ms_restored = MutationSigmas.from_dict(d)

        assert ms_restored.weight_sigma == ms.weight_sigma
        assert ms_restored.bias_sigma == ms.bias_sigma


# =============================================================================
# HeuristicFeatureConfig Tests
# =============================================================================

class TestHeuristicFeatureConfig:
    """Tests for HeuristicFeatureConfig dataclass."""

    def test_default_features(self):
        """Test default feature lists."""
        hfc = HeuristicFeatureConfig()
        assert len(hfc.movement) > 0
        assert len(hfc.deposit) > 0
        assert len(hfc.ranking) > 0

    def test_preset_configs(self):
        """Test preset configurations."""
        assert "stark_centrality" in STARK_FEATURES.movement
        assert "random_jitter" in GENERAL_FEATURES.movement

    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict."""
        hfc = HeuristicFeatureConfig(
            movement=["feat1", "feat2"],
            deposit=["feat3"],
            ranking=["feat4"],
        )

        d = hfc.to_dict()
        hfc_restored = HeuristicFeatureConfig.from_dict(d)

        assert hfc_restored.movement == hfc.movement
        assert hfc_restored.deposit == hfc.deposit


# =============================================================================
# Genome Dual-Mode Tests
# =============================================================================

class TestGenomeDualMode:
    """Tests for Genome dual-mode support."""

    def test_default_mode_is_expression_tree(self):
        """Test backward compatibility - default mode is expression_tree."""
        g = Genome(id="test")
        assert g.mode == "expression_tree"
        assert g.weight_tensors is None

    def test_weighted_sum_mode(self):
        """Test weighted_sum mode initialization."""
        wt = WeightTensors()
        g = Genome(
            id="test_ws",
            mode="weighted_sum",
            weight_tensors=wt,
        )
        assert g.mode == "weighted_sum"
        assert g.weight_tensors is not None

    def test_complexity_expression_tree(self):
        """Test complexity calculation for expression_tree mode."""
        g = Genome(id="test", mode="expression_tree")
        # Empty strategies = 0 complexity
        assert g.complexity() == 0

    def test_complexity_weighted_sum(self):
        """Test complexity calculation for weighted_sum mode."""
        wt = WeightTensors(
            movement_weights=torch.randn(2, 3),
            movement_biases=torch.randn(2),
            deposit_weights=torch.randn(2, 2),
            deposit_biases=torch.randn(2),
            ranking_weights=torch.randn(2),
            ranking_bias=0.0,
        )
        g = Genome(id="test_ws", mode="weighted_sum", weight_tensors=wt)

        assert g.complexity() == wt.total_params

    def test_copy_preserves_mode(self):
        """Test that copy preserves genome mode."""
        wt = WeightTensors()
        g = Genome(
            id="original",
            mode="weighted_sum",
            weight_tensors=wt,
            group_ratios={"g0": 1.0},
        )

        g_copy = g.copy("copy")

        assert g_copy.mode == "weighted_sum"
        assert g_copy.weight_tensors is not None
        assert g_copy.weight_tensors is not g.weight_tensors

    def test_to_dict_serializes_weight_tensors(self):
        """Test that to_dict properly serializes weight_tensors."""
        wt = WeightTensors()
        g = Genome(id="test_ws", mode="weighted_sum", weight_tensors=wt)

        d = g.to_dict()

        assert "weight_tensors" in d
        assert d["weight_tensors"] is not None
        assert "movement_weights" in d["weight_tensors"]


# =============================================================================
# WeightedSumSeeder Tests
# =============================================================================

class TestWeightedSumSeeder:
    """Tests for WeightedSumSeeder."""

    @pytest.fixture
    def seeder(self):
        """Create seeder with test features."""
        return WeightedSumSeeder(HeuristicFeatureConfig())

    def test_create_seed_population(self, seeder):
        """Test seed population creation."""
        seeds = seeder.create_seed_population(5)

        assert len(seeds) == 5
        for g in seeds:
            assert g.mode == "weighted_sum"
            assert g.weight_tensors is not None

    def test_baseline_genome_created(self, seeder):
        """Test that baseline genome is in seeds."""
        seeds = seeder.create_seed_population(10)
        baseline = next((g for g in seeds if "baseline" in g.id), None)
        assert baseline is not None

    def test_all_seeds_have_valid_weights(self, seeder):
        """Test that all seeds have valid weight shapes."""
        seeds = seeder.create_seed_population(18)

        for g in seeds:
            wt = g.weight_tensors
            assert wt.movement_weights.shape[1] == len(seeder.feature_config.movement)
            assert wt.deposit_weights.shape[1] == len(seeder.feature_config.deposit)
            assert wt.ranking_weights.shape[0] == len(seeder.feature_config.ranking)


# =============================================================================
# WeightedSumMutator Tests
# =============================================================================

class TestWeightedSumMutator:
    """Tests for WeightedSumMutator."""

    @pytest.fixture
    def mutator_and_context(self):
        """Create mutator and context."""
        feature_config = HeuristicFeatureConfig()
        config = EvolutionConfig(
            genome_mode="weighted_sum",
            heuristic_features=feature_config,
        )
        ctx = EvolutionContext(config=config)
        mutator = WeightedSumMutator(feature_config)
        return mutator, ctx

    @pytest.fixture
    def sample_genome(self):
        """Create a sample weighted_sum genome."""
        seeder = WeightedSumSeeder(HeuristicFeatureConfig())
        return seeder.create_seed_population(1)[0]

    def test_mutation_changes_genome(self, mutator_and_context, sample_genome):
        """Test that mutation changes the genome."""
        mutator, ctx = mutator_and_context

        original_weights = sample_genome.weight_tensors.movement_weights.clone()

        # Run mutation multiple times to ensure change
        for _ in range(20):
            g_copy = sample_genome.copy("mutated")
            mutator.mutate(g_copy, ctx)

            if not torch.equal(original_weights, g_copy.weight_tensors.movement_weights):
                return  # Success

        pytest.fail("Mutation should change weights")

    def test_mutation_marks_unevaluated(self, mutator_and_context, sample_genome):
        """Test that mutation marks genome as unevaluated."""
        mutator, ctx = mutator_and_context

        sample_genome.evaluated = True
        mutator.mutate(sample_genome, ctx)

        assert not sample_genome.evaluated


# =============================================================================
# GenomeFactory Tests
# =============================================================================

class TestGenomeFactoryDualMode:
    """Tests for GenomeFactory dual-mode support."""

    def test_weighted_sum_mode_uses_correct_strategy(self):
        """Test that weighted_sum mode uses weighted_sum_seeded strategy."""
        config = EvolutionConfig(
            genome_mode="weighted_sum",
            heuristic_features=HeuristicFeatureConfig(),
        )
        ctx = EvolutionContext(config=config)
        factory = GenomeFactory(ctx)

        pop = factory.create_population(3)

        assert len(pop) == 3
        assert all(g.mode == "weighted_sum" for g in pop)

    def test_expression_tree_mode_backward_compatible(self):
        """Test that expression_tree mode still works."""
        config = EvolutionConfig(genome_mode="expression_tree")
        ctx = EvolutionContext(config=config)
        ctx.available_features = ["semantic_similarity_unnormalized"]
        ctx.expression_features = {
            "movement": ["semantic_similarity_unnormalized"],
            "deposit": ["semantic_similarity_unnormalized"],
            "ranking": ["semantic_similarity_unnormalized"],
        }
        factory = GenomeFactory(ctx)

        pop = factory.create_population(2)

        assert len(pop) == 2
        assert all(g.mode == "expression_tree" for g in pop)


# =============================================================================
# Batch Computation Tests
# =============================================================================

class TestBatchComputation:
    """Tests for GPU-optimized batch computation functions."""

    def test_compute_movement_scores_batched(self):
        """Test batched movement score computation."""
        n_candidates = 10
        n_features = 4
        n_groups = 3

        features = torch.randn(n_candidates, n_features)
        weights = torch.randn(n_groups, n_features)
        biases = torch.randn(n_groups)

        scores = compute_movement_scores_batched(features, weights, biases)

        assert scores.shape == (n_candidates, n_groups)

    def test_assign_scores_to_agents(self):
        """Test score assignment to agents."""
        n_candidates = 10
        n_groups = 3
        n_agents = 20

        scores_all_groups = torch.randn(n_candidates, n_groups)
        agent_group_ids = torch.randint(0, n_groups, (n_agents,))

        agent_scores = assign_scores_to_agents(scores_all_groups, agent_group_ids)

        assert agent_scores.shape == (n_candidates, n_agents)


# =============================================================================
# Strategy Registration Tests
# =============================================================================

class TestStrategyRegistration:
    """Tests for weighted sum strategy registration."""

    def test_self_adaptive_es_registered(self):
        """Test that self_adaptive_es mutation is registered."""
        assert "self_adaptive_es" in GeneticRegistry.all_mutation()

    def test_weighted_sum_seeded_registered(self):
        """Test that weighted_sum_seeded creation is registered."""
        assert "weighted_sum_seeded" in GeneticRegistry.all_creation()

    def test_weighted_sum_crossover_registered(self):
        """Test that weighted_sum_crossover is registered."""
        assert "weighted_sum_crossover" in GeneticRegistry.all_crossover()


# =============================================================================
# Checkpoint Serialization Tests
# =============================================================================

class TestCheckpointSerialization:
    """Tests for checkpoint serialization with dual-mode genomes."""

    def test_weighted_sum_genome_roundtrip(self):
        """Test saving and loading weighted_sum genome via checkpoint."""
        seeder = WeightedSumSeeder(HeuristicFeatureConfig())
        genome = seeder.create_seed_population(1)[0]

        # Set some fitness values
        genome.fitness = FitnessResult(quality_score=0.75)
        genome.metrics = {"Recall@20": 0.8}
        genome.evaluated = True

        with tempfile.TemporaryDirectory() as tmpdir:
            from swarm_rag.evolution.types.config import StorageConfig
            storage_config = StorageConfig(
                base_dir=tmpdir,
                dataset="test",
                run_id="test_run",
                async_checkpoints=False,  # Disable async for synchronous test
            )

            run_manager = RunManager(storage_config)
            run_manager.initialize_run()

            # Save checkpoint
            run_manager.save_checkpoint(
                population=[genome],
                best_genome=genome,
                generation=0,
            )

            # Load checkpoint
            state = run_manager.load_checkpoint()

            loaded_genome = state["population"][0]

            assert loaded_genome.mode == "weighted_sum"
            assert loaded_genome.weight_tensors is not None
            # Move to same device for comparison
            assert torch.allclose(
                loaded_genome.weight_tensors.movement_weights.cpu(),
                genome.weight_tensors.movement_weights.cpu(),
            )
            assert loaded_genome.fitness.quality_score == pytest.approx(0.75)


# =============================================================================
# Integration Tests
# =============================================================================

class TestDualModeIntegration:
    """Integration tests for dual-mode evolution."""

    def test_full_mutation_crossover_cycle(self):
        """Test full cycle: create -> mutate -> crossover."""
        feature_config = HeuristicFeatureConfig()
        config = EvolutionConfig(
            genome_mode="weighted_sum",
            heuristic_features=feature_config,
        )
        ctx = EvolutionContext(config=config)

        # Create population
        factory = GenomeFactory(ctx)
        pop = factory.create_population(4)

        assert len(pop) == 4

        # Mutate
        mutator = WeightedSumMutator(feature_config)
        mutated = mutator.mutate(pop[0].copy("mutated"), ctx)

        assert mutated.mode == "weighted_sum"
        assert not mutated.evaluated

        # Crossover
        crossover_fn = GeneticRegistry.get_crossover("weighted_sum_crossover")
        child = crossover_fn(pop[0], pop[1], ctx)

        assert child.mode == "weighted_sum"
        assert child.weight_tensors is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
