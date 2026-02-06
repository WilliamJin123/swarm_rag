# tests/evolution/test_focused_mutation.py
"""Tests for focused metric-aware mutation."""
import pytest
from swarm_rag.evolution.execution.genetics.focused_mutation import (
    identify_weakest_metric,
    get_mutation_focus,
    METRIC_TO_PARAM_MAPPING,
)
from swarm_rag.evolution.types.genome import Genome
from swarm_rag.evolution.types.fitness_results import FitnessResult


def _make_fitness(recall: float, mrr: float, precision: float) -> FitnessResult:
    """Helper to create FitnessResult with metrics dict."""
    return FitnessResult(
        quality_score=recall,  # Use recall as quality for simplicity
        metrics={
            "recall_at_20": recall,
            "mrr": mrr,
            "precision_at_20": precision,
        }
    )


def test_identify_weakest_metric_recall():
    """Should identify recall as weakest when it's lowest."""
    fitness = _make_fitness(recall=0.15, mrr=0.40, precision=0.35)
    weakest = identify_weakest_metric(fitness)
    assert weakest == "recall_at_20"


def test_identify_weakest_metric_mrr():
    """Should identify MRR as weakest when it's lowest."""
    fitness = _make_fitness(recall=0.50, mrr=0.10, precision=0.45)
    weakest = identify_weakest_metric(fitness)
    assert weakest == "mrr"


def test_identify_weakest_metric_precision():
    """Should identify precision as weakest when it's lowest."""
    fitness = _make_fitness(recall=0.50, mrr=0.45, precision=0.08)
    weakest = identify_weakest_metric(fitness)
    assert weakest == "precision_at_20"


def test_metric_to_param_mapping_exists():
    """All key metrics should have parameter mappings."""
    assert "recall_at_20" in METRIC_TO_PARAM_MAPPING
    assert "mrr" in METRIC_TO_PARAM_MAPPING
    assert "precision_at_20" in METRIC_TO_PARAM_MAPPING


def test_get_mutation_focus_returns_params():
    """get_mutation_focus should return parameters to adjust."""
    fitness = _make_fitness(recall=0.15, mrr=0.40, precision=0.35)
    focus = get_mutation_focus(fitness)
    assert "params" in focus
    assert "direction" in focus
    assert isinstance(focus["params"], list)
    assert len(focus["params"]) > 0


def test_get_mutation_focus_recall_suggests_more_agents():
    """Low recall should suggest increasing n_agents or steps."""
    fitness = _make_fitness(recall=0.10, mrr=0.50, precision=0.45)
    focus = get_mutation_focus(fitness)
    # For low recall, we want more coverage - either more agents or more steps
    assert any(p in focus["params"] for p in ["n_agents", "steps", "initial_pool_size"])


def test_get_mutation_focus_mrr_suggests_decay():
    """Low MRR should suggest adjusting decay or ranking strategies."""
    fitness = _make_fitness(recall=0.50, mrr=0.10, precision=0.45)
    focus = get_mutation_focus(fitness)
    # For low MRR, we want better ranking - adjust decay or n_agents
    assert any(p in focus["params"] for p in ["decay", "n_agents"])


def test_identify_weakest_metric_missing_metrics():
    """Should handle fitness without metrics gracefully."""
    fitness = FitnessResult(quality_score=0.5)  # No metrics dict
    weakest = identify_weakest_metric(fitness)
    # Should default to recall when metrics not available
    assert weakest == "recall_at_20"
