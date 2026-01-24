
import os
from pathlib import Path
import shutil
import random
import tempfile
import pytest
from typing import List, Any, Dict

from swarm_rag.evolution.engine import EvolutionEngine
from swarm_rag.evolution.types.config import EvolutionConfig, StorageConfig
from swarm_rag.evolution.storage import RunManager
from swarm_rag.evolution.execution.evaluator import PopulationEvaluator
from swarm_rag.evolution.execution.fitness import FitnessCalculator
from swarm_rag.eval.metrics import Evaluator as BaseEvaluator

# Reuse the Toy components
class ToyStochasticRetriever:
    def retrieve_batch(self, queries: List[str], max_workers: int = 1, **kwargs):
        results = []
        n_agents = kwargs.get('n_agents', 1)
        step_prob = kwargs.get('alpha', 0.5)
        decay = kwargs.get('decay', 0.9)
        max_steps = int(decay * 50)

        for q in queries:
            target_node = int(q)
            found = False
            for _ in range(n_agents):
                current_pos = 0
                for _ in range(max_steps):
                    if random.random() < step_prob:
                        current_pos += 1
                    if current_pos == target_node:
                        found = True
                        break
                if found: break

            if found:
                results.append([{'id': target_node, 'score': 1.0, 'content': 'Target Reached'}])
            else:
                results.append([{'id': -1, 'score': 0.0, 'content': 'Failed'}])
        return results

class IntEvaluator(BaseEvaluator):
    def calculate_metrics(self, retrieved_nodes, ground_truth_ids, latency_sec=0):
        hit = 0.0
        if retrieved_nodes and retrieved_nodes[0]['id'] == ground_truth_ids[0]:
            hit = 1.0
        return {
            "Recall@10": hit,
            "Hit@10": hit,
            "MRR": hit,
            "latency": 10.0
        }

RESULTS_DIR = None  # Will be set in setup_module

def setup_module():
    global RESULTS_DIR
    RESULTS_DIR = tempfile.mkdtemp(prefix="evo_map_test_")

def teardown_module():
    global RESULTS_DIR
    if RESULTS_DIR and os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR, ignore_errors=True)

def test_map_elites_flow():
    print("\n\n=== STARTING MAP-ELITES SIMULATION ===")

    # 1. Configuration
    storage = StorageConfig(
        base_dir=RESULTS_DIR,
        dataset="test",
        run_id="map_test",
        checkpoint_frequency=1,
        validation_frequency=1,
        plot_title="MAP-Elites Test",
    )

    config = EvolutionConfig(
        n_generations=5,
        storage=storage,
    )
    config.map_elites.batch_size = 10
    config.map_elites.dimensions = ["complexity", "n_agents"]
    config.map_elites.bins = [5, 5]
    config.map_elites.ranges = [(0, 50), (1, 10)]
    config.map_elites.initial_fill = 20

    config.genetic.param_ranges.n_agents = (1, 10)
    config.genetic.param_ranges.decay = (0.1, 0.99)

    train_queries = ["10", "15"]
    train_gt = [[10], [15]]
    val_queries = ["12"]
    val_gt = [[12]]

    retriever = ToyStochasticRetriever()
    evaluator = IntEvaluator(index_name="toy_sim")
    fitness_calculator = FitnessCalculator(weights={"Recall@10": 1.0})

    # Create RunManager
    run_manager = RunManager(storage)

    # 4. Initialize Engine
    engine = EvolutionEngine(
        retriever=retriever,
        fitness_calculator=fitness_calculator,
        evaluator=evaluator,
        train_query_ids=train_queries,
        train_ground_truth=train_gt,
        val_query_ids=val_queries,
        val_ground_truth=val_gt,
        config=config,
        run_manager=run_manager,
    )
    engine.initialize_run()

    assert engine.map_elites_archive is not None, "Archive should be initialized"
    assert engine.map_elites_loop is not None, "Loop should be initialized"

    # 5. RUN EVOLUTION
    print("  > Starting optimization...")
    best_genome = engine.optimize()

    # 6. VERIFICATION

    # Check Archive Stats
    stats = engine.map_elites_archive.stats()
    print(f"\n  > Archive Stats: {stats}")

    assert stats["filled_cells"] > 0, "Archive should not be empty"
    assert stats["coverage"] > 0.0, "Coverage should be > 0%"
    assert best_genome is not None

    # Check that we actually used the archive (grid has items)
    assert len(engine.map_elites_archive.grid) > 0

    # Check Checkpoint existence
    assert os.path.exists(storage.latest_checkpoint_path), f"Final checkpoint missing at {storage.latest_checkpoint_path}"

    print("  MAP-Elites simulation passed!")

if __name__ == "__main__":
    setup_module()
    try:
        test_map_elites_flow()
    finally:
        teardown_module()
