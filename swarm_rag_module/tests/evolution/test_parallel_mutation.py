# tests/evolution/test_parallel_mutation.py
"""Tests for parallel LLM mutation configuration and execution."""
import pytest
from swarm_rag.evolution.types.config import EvolutionConfig, GeneticConfig


def test_config_has_parallel_mutation_workers():
    """Config should have parallel_mutation_workers setting."""
    config = EvolutionConfig()
    assert hasattr(config.genetic, "parallel_mutation_workers")
    assert config.genetic.parallel_mutation_workers >= 1


def test_default_parallel_workers():
    """Default should be 4 workers for parallel mutations."""
    config = EvolutionConfig()
    assert config.genetic.parallel_mutation_workers == 4


def test_parallel_workers_configurable():
    """Parallel workers should be configurable."""
    config = EvolutionConfig(
        genetic=GeneticConfig(parallel_mutation_workers=8)
    )
    assert config.genetic.parallel_mutation_workers == 8
