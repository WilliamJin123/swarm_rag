# tests/evolution/test_genome.py
"""Tests for streamlined hyperparameter search space."""
import pytest
from swarm_rag.evolution.types.genome import (
    Genome,
    FIXED_PARAMS,
    EVOLVABLE_PARAM_RANGES,
    create_random_genome
)


def test_fixed_params_not_in_evolvable_ranges():
    """Fixed parameters should not be in evolvable ranges."""
    for param in FIXED_PARAMS:
        assert param not in EVOLVABLE_PARAM_RANGES, f"{param} should not be evolvable"


def test_fixed_params_have_default_values():
    """Fixed parameters should have default values."""
    assert "drop_zone_inc" in FIXED_PARAMS
    assert FIXED_PARAMS["drop_zone_inc"] == 0.05
    assert "start_subset" in FIXED_PARAMS
    assert FIXED_PARAMS["start_subset"] == 10


def test_evolvable_ranges_are_tightened():
    """Evolvable parameter ranges should be tightened per brainstorm."""
    assert EVOLVABLE_PARAM_RANGES["n_agents"] == (15, 50)
    assert EVOLVABLE_PARAM_RANGES["steps"] == (3, 7)
    assert EVOLVABLE_PARAM_RANGES["decay"] == (0.3, 0.8)
    assert EVOLVABLE_PARAM_RANGES["initial_pool_size"] == (20, 60)


def test_create_random_genome_uses_fixed_params():
    """New genomes should use fixed parameter values."""
    genome = create_random_genome()
    assert genome.params["drop_zone_inc"] == FIXED_PARAMS["drop_zone_inc"]
    assert genome.params["start_subset"] == FIXED_PARAMS["start_subset"]


def test_create_random_genome_respects_evolvable_ranges():
    """New genomes should have evolvable params within tightened ranges."""
    genome = create_random_genome()

    n_agents_range = EVOLVABLE_PARAM_RANGES["n_agents"]
    assert n_agents_range[0] <= genome.params["n_agents"] <= n_agents_range[1]

    steps_range = EVOLVABLE_PARAM_RANGES["steps"]
    assert steps_range[0] <= genome.params["steps"] <= steps_range[1]

    decay_range = EVOLVABLE_PARAM_RANGES["decay"]
    assert decay_range[0] <= genome.params["decay"] <= decay_range[1]

    pool_range = EVOLVABLE_PARAM_RANGES["initial_pool_size"]
    assert pool_range[0] <= genome.params["initial_pool_size"] <= pool_range[1]
