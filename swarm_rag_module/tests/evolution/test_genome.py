# tests/evolution/test_genome.py
"""Tests for streamlined hyperparameter search space."""
import pytest
from swarm_rag.evolution.types.genome import (
    Genome,
    FIXED_PARAMS,
    create_random_genome
)
from swarm_rag.evolution.types.config import SwarmParamRanges


def test_fixed_params_not_in_evolvable_ranges():
    """Fixed parameters should not be in evolvable ranges."""
    evolvable_ranges = SwarmParamRanges().to_evolvable_dict()
    for param in FIXED_PARAMS:
        assert param not in evolvable_ranges, f"{param} should not be evolvable"


def test_fixed_params_have_default_values():
    """Fixed parameters should have default values."""
    assert "drop_zone_inc" in FIXED_PARAMS
    assert FIXED_PARAMS["drop_zone_inc"] == 0.05
    assert "start_subset" in FIXED_PARAMS
    assert FIXED_PARAMS["start_subset"] == 10


def test_evolvable_ranges_are_tightened():
    """Evolvable parameter ranges should be tightened per brainstorm."""
    ranges = SwarmParamRanges()
    assert ranges.n_agents == (10, 30)
    assert ranges.steps == (3, 5)
    assert ranges.decay == (0.3, 0.8)
    assert ranges.initial_pool_size == (15, 40)


def test_create_random_genome_uses_fixed_params():
    """New genomes should use fixed parameter values."""
    genome = create_random_genome()
    assert genome.params["drop_zone_inc"] == FIXED_PARAMS["drop_zone_inc"]
    assert genome.params["start_subset"] == FIXED_PARAMS["start_subset"]


def test_create_random_genome_respects_evolvable_ranges():
    """New genomes should have evolvable params within tightened ranges."""
    genome = create_random_genome()
    ranges = SwarmParamRanges()

    assert ranges.n_agents[0] <= genome.params["n_agents"] <= ranges.n_agents[1]
    assert ranges.steps[0] <= genome.params["steps"] <= ranges.steps[1]
    assert ranges.decay[0] <= genome.params["decay"] <= ranges.decay[1]
    assert ranges.initial_pool_size[0] <= genome.params["initial_pool_size"] <= ranges.initial_pool_size[1]
