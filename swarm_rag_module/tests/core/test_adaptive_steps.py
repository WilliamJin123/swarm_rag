# tests/core/test_adaptive_steps.py
"""Tests for adaptive step count with early convergence detection."""
import pytest
import torch
from swarm_rag.core.swarm_retriever import should_continue_stepping


def test_should_continue_early_steps():
    """Always continue for first min_steps."""
    positions = torch.tensor([1, 2, 3, 4, 5])
    prev_positions = torch.tensor([1, 2, 3, 4, 5])  # All same (converged)

    # Step 0 should continue even if converged
    assert should_continue_stepping(positions, prev_positions, step_idx=0, min_steps=2) is True
    # Step 1 should continue even if converged
    assert should_continue_stepping(positions, prev_positions, step_idx=1, min_steps=2) is True


def test_should_stop_when_converged():
    """Stop if 80%+ agents haven't moved after min_steps."""
    # 10 agents, 8 stuck (80%)
    positions = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    prev_positions = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 99, 100])  # 8 same

    assert should_continue_stepping(positions, prev_positions, step_idx=3, min_steps=2) is False


def test_should_continue_when_not_converged():
    """Continue if agents are still moving."""
    # 10 agents, only 5 stuck (50%)
    positions = torch.tensor([1, 2, 3, 4, 5, 11, 12, 13, 14, 15])
    prev_positions = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    assert should_continue_stepping(positions, prev_positions, step_idx=3, min_steps=2) is True


def test_convergence_threshold_configurable():
    """Convergence threshold should be configurable."""
    # 10 agents, 7 stuck (70%)
    positions = torch.tensor([1, 2, 3, 4, 5, 6, 7, 18, 19, 20])
    prev_positions = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # With default 80% threshold, should continue
    assert should_continue_stepping(positions, prev_positions, step_idx=3, min_steps=2) is True

    # With 70% threshold, should stop
    assert should_continue_stepping(
        positions, prev_positions, step_idx=3, min_steps=2, convergence_threshold=0.7
    ) is False
