# tests/evolution/test_seed_genomes.py
"""Tests for warm-start seed genomes."""
import pytest
from swarm_rag.evolution.seed_genomes import SEED_GENOMES, create_seed_genome


def test_seed_genomes_exist():
    """At least 3 seed genomes should be defined."""
    assert len(SEED_GENOMES) >= 3


def test_seed_genome_has_required_fields():
    """Each seed genome should have required fields."""
    required_fields = [
        "n_agents",
        "steps",
        "decay",
        "initial_pool_size",
        "movement_tree",
        "deposit_tree",
    ]
    for seed in SEED_GENOMES:
        for field in required_fields:
            assert field in seed, f"Seed '{seed.get('name', 'unknown')}' missing {field}"


def test_create_seed_genome_returns_valid_genome():
    """create_seed_genome should return a valid Genome object."""
    from swarm_rag.evolution.types.genome import Genome

    genome = create_seed_genome(SEED_GENOMES[0])
    assert genome is not None
    assert isinstance(genome, Genome)
    assert genome.params["n_agents"] == SEED_GENOMES[0]["n_agents"]
    assert genome.evaluated is False


def test_seed_genomes_use_fixed_params():
    """Seed genomes should use fixed parameter values."""
    from swarm_rag.evolution.types.genome import FIXED_PARAMS

    genome = create_seed_genome(SEED_GENOMES[0])
    for key, value in FIXED_PARAMS.items():
        assert genome.params[key] == value, f"Seed genome has wrong value for fixed param {key}"


def test_seed_genomes_have_valid_evolvable_params():
    """Seed genomes should have evolvable params within valid ranges."""
    from swarm_rag.evolution.types.genome import EVOLVABLE_PARAM_RANGES

    for seed_config in SEED_GENOMES:
        genome = create_seed_genome(seed_config)
        for key, (low, high) in EVOLVABLE_PARAM_RANGES.items():
            if key in genome.params:
                val = genome.params[key]
                assert low <= val <= high, f"Seed '{seed_config.get('name')}' param {key}={val} out of range [{low}, {high}]"


def test_seed_genomes_have_unique_names():
    """Seed genomes should have unique names."""
    names = [seed.get("name") for seed in SEED_GENOMES]
    assert len(names) == len(set(names)), "Seed genomes must have unique names"


def test_create_seed_genome_has_strategies():
    """Created seed genomes should have populated strategies."""
    genome = create_seed_genome(SEED_GENOMES[0])
    assert genome.strategies, "Seed genome should have strategies"
    # Should have ranking strategy
    assert "ranking" in genome.strategies, "Seed genome should have ranking strategy"
