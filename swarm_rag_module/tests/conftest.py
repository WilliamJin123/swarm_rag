"""
Shared test fixtures for the swarm_rag test suite.

Contains common test components used across multiple test files:
- ToyStochasticRetriever: Simple retriever simulation for testing
- IntEvaluator: Evaluator that works with integer IDs
- Storage fixtures: Temporary directories for test outputs
- Config fixtures: Standard test configurations
"""
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import List, Any, Dict

import pytest

from swarm_rag.evolution.types.config import EvolutionConfig, StorageConfig
from swarm_rag.evolution.storage import RunManager
from swarm_rag.evolution.execution.fitness import FitnessCalculator
from swarm_rag.eval.metrics import Evaluator as BaseEvaluator


class ToyStochasticRetriever:
    """
    A fully functional retriever simulation for testing.

    Scenario:
        - A 1D line graph: 0 -> 1 -> 2 -> ... -> 50.
        - Agents start at node 0.
        - Target is at node N (defined by the query).

    Genome Parameters Controlled:
        - 'n_agents': How many attempts we get.
        - 'step_prob': Probability of moving forward (simulating 'alpha').
        - 'max_steps': How long the agent survives (simulating 'decay').

    Goal:
        Evolution must find high 'step_prob' and sufficient 'max_steps' to reach the target.
    """
    def retrieve_batch(self, queries: List[str], max_workers: int = 1, **kwargs):
        results = []

        # Extract Genome Params (with defaults if mutation breaks things)
        n_agents = kwargs.get('n_agents', 1)
        # Map 'alpha' (0-1) to step_prob
        step_prob = kwargs.get('alpha', 0.5)
        # Map 'decay' (0-1) to max_steps (0-50)
        decay = kwargs.get('decay', 0.9)
        max_steps = int(decay * 50)

        for q in queries:
            target_node = int(q)
            found = False

            # Run the Simulation (Real Logic)
            for _ in range(n_agents):
                current_pos = 0
                for _ in range(max_steps):
                    if random.random() < step_prob:
                        current_pos += 1

                    if current_pos == target_node:
                        found = True
                        break
                if found:
                    break

            # Return result format expected by Evaluator
            if found:
                # Found the target!
                results.append([{'id': target_node, 'score': 1.0, 'content': 'Target Reached'}])
            else:
                # Failed (stuck at current_pos)
                results.append([{'id': -1, 'score': 0.0, 'content': 'Failed'}])

        return results


class IntEvaluator(BaseEvaluator):
    """Evaluator that works with integer IDs for testing."""

    def __init__(self, index_name: str = "test"):
        super().__init__(index_name=index_name)

    def calculate_metrics(self, retrieved_nodes, ground_truth_ids, latency_sec=0):
        # Strict ID matching
        hit = 0.0
        if retrieved_nodes and retrieved_nodes[0]['id'] == ground_truth_ids[0]:
            hit = 1.0
        return {
            "Recall@10": hit,
            "Recall@20": hit,
            "Hit@1": hit,
            "Hit@5": hit,
            "Hit@10": hit,
            "MRR": hit,
            "latency": 10.0
        }


@pytest.fixture
def temp_results_dir(tmp_path):
    """Provide a temporary directory for test results."""
    results_dir = tmp_path / "evo_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    yield str(results_dir)
    # Cleanup handled by pytest tmp_path


@pytest.fixture
def test_storage_config(temp_results_dir):
    """Returns a consistent storage config for tests."""
    return StorageConfig(
        base_dir=temp_results_dir,
        dataset="test",
        run_id="test_run",
        validation_frequency=1,
        checkpoint_frequency=1,
        plot_title="Test Evolution",
    )


@pytest.fixture
def test_evolution_config(test_storage_config):
    """Returns a standard evolution config for tests."""
    config = EvolutionConfig(storage=test_storage_config)

    # Standard Params
    config.n_generations = 3
    config.map_elites.batch_size = 10
    config.genetic.selection_k = 3

    # Toy Problem Search Space
    config.genetic.param_ranges.n_agents = (1, 5)
    config.genetic.param_ranges.decay = (0.1, 0.99)

    return config


@pytest.fixture
def toy_retriever():
    """Provide a ToyStochasticRetriever instance."""
    return ToyStochasticRetriever()


@pytest.fixture
def int_evaluator():
    """Provide an IntEvaluator instance."""
    return IntEvaluator(index_name="test")


@pytest.fixture
def simple_fitness_calculator():
    """Provide a simple fitness calculator for tests."""
    return FitnessCalculator.from_weights({"Recall@10": 1.0})


@pytest.fixture
def balanced_fitness_calculator():
    """Provide a balanced fitness calculator for tests."""
    return FitnessCalculator.from_weights({
        "Hit@1": 0.25,
        "Hit@5": 0.25,
        "MRR": 0.25,
        "Recall@20": 0.25,
    })


@pytest.fixture
def train_data():
    """Provide simple training data for tests."""
    queries = ["10", "15", "20"]
    ground_truth = [[10], [15], [20]]
    return queries, ground_truth


@pytest.fixture
def val_data():
    """Provide simple validation data for tests."""
    queries = ["12", "18"]
    ground_truth = [[12], [18]]
    return queries, ground_truth


@pytest.fixture
def run_manager(test_storage_config):
    """Provide a RunManager instance."""
    return RunManager(test_storage_config)
