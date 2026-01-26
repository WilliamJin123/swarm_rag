# tests/evolution/test_evaluator.py
"""Tests for aggressive early stopping tier configuration."""
import pytest
from swarm_rag.evolution.execution.evaluator import DEFAULT_TIERS, EvaluationTier


def test_aggressive_early_stopping_tiers():
    """Verify aggressive early stopping tier configuration."""
    assert len(DEFAULT_TIERS) == 4

    # Tier 1: Quick filter (3 queries, 0.15 threshold)
    assert DEFAULT_TIERS[0].queries == 3
    assert DEFAULT_TIERS[0].threshold == 0.15
    assert DEFAULT_TIERS[0].name == "quick_filter"

    # Tier 2: Filter poor performers (8 queries, 0.30 threshold)
    assert DEFAULT_TIERS[1].queries == 8
    assert DEFAULT_TIERS[1].threshold == 0.30
    assert DEFAULT_TIERS[1].name == "poor_filter"

    # Tier 3: Filter mediocre (20 queries, 0.45 threshold)
    assert DEFAULT_TIERS[2].queries == 20
    assert DEFAULT_TIERS[2].threshold == 0.45
    assert DEFAULT_TIERS[2].name == "mediocre_filter"

    # Tier 4: Full evaluation (no threshold)
    assert DEFAULT_TIERS[3].threshold is None
    assert DEFAULT_TIERS[3].name == "full"


def test_tier_thresholds_are_progressive():
    """Thresholds should increase with each tier."""
    thresholds = [t.threshold for t in DEFAULT_TIERS if t.threshold is not None]
    for i in range(1, len(thresholds)):
        assert thresholds[i] > thresholds[i - 1], "Thresholds must be progressive"


def test_tier_queries_are_progressive():
    """Query counts should increase with each tier."""
    query_counts = [t.queries for t in DEFAULT_TIERS]
    for i in range(1, len(query_counts)):
        assert query_counts[i] > query_counts[i - 1], "Query counts must be progressive"


def test_evaluation_tier_dataclass():
    """EvaluationTier should be a proper dataclass."""
    tier = EvaluationTier(queries=10, threshold=0.25, name="test")
    assert tier.queries == 10
    assert tier.threshold == 0.25
    assert tier.name == "test"
