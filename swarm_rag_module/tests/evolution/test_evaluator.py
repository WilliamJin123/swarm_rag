# tests/evolution/test_evaluator.py
"""Tests for single-checkpoint early exit evaluation configuration."""
import pytest
from swarm_rag.evolution.execution.evaluator import (
    DEFAULT_EARLY_EXIT_THRESHOLD,
    PopulationEvaluator,
)
from swarm_rag.evolution.types.config import ResourceConfig


def test_default_early_exit_threshold():
    """Verify default early exit threshold is set correctly."""
    assert DEFAULT_EARLY_EXIT_THRESHOLD == 0.30


def test_resource_config_has_early_exit_threshold():
    """ResourceConfig should have early_exit_threshold with correct default."""
    config = ResourceConfig()
    assert hasattr(config, "early_exit_threshold")
    assert config.early_exit_threshold == 0.30


def test_early_exit_threshold_is_configurable():
    """early_exit_threshold should be configurable via ResourceConfig."""
    config = ResourceConfig(early_exit_threshold=0.45)
    assert config.early_exit_threshold == 0.45


def test_evaluator_accepts_early_exit_threshold():
    """PopulationEvaluator should accept early_exit_threshold parameter."""
    # Create a minimal mock retriever
    class MockRetriever:
        pass

    class MockEvaluator:
        k_values = [1, 5, 10, 20]

    class MockFitnessCalc:
        pass

    evaluator = PopulationEvaluator(
        retriever=MockRetriever(),
        evaluator=MockEvaluator(),
        fitness_calc=MockFitnessCalc(),
        early_exit_threshold=0.35,
    )
    assert evaluator.early_exit_threshold == 0.35


def test_evaluator_uses_default_threshold():
    """PopulationEvaluator should use default threshold when not specified."""
    class MockRetriever:
        pass

    class MockEvaluator:
        k_values = [1, 5, 10, 20]

    class MockFitnessCalc:
        pass

    evaluator = PopulationEvaluator(
        retriever=MockRetriever(),
        evaluator=MockEvaluator(),
        fitness_calc=MockFitnessCalc(),
    )
    assert evaluator.early_exit_threshold == DEFAULT_EARLY_EXIT_THRESHOLD


def test_stats_have_simplified_tier_exits():
    """EvaluationStats should track early_exit and full only."""
    class MockRetriever:
        pass

    class MockEvaluator:
        k_values = [1, 5, 10, 20]

    class MockFitnessCalc:
        pass

    evaluator = PopulationEvaluator(
        retriever=MockRetriever(),
        evaluator=MockEvaluator(),
        fitness_calc=MockFitnessCalc(),
    )
    # Stats should have simplified tier_exits
    assert "early_exit" in evaluator.stats.tier_exits
    assert "full" in evaluator.stats.tier_exits
    assert len(evaluator.stats.tier_exits) == 2
