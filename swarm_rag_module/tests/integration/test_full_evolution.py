
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

# --- 1. THE TOY SIMULATOR (Replaces SwarmRetriever with real logic) ---

class ToyStochasticRetriever:
    """
    A fully functional retriever simulation.
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
                if found: break

            # Return result format expected by Evaluator
            if found:
                # Found the target!
                results.append([{'id': target_node, 'score': 1.0, 'content': 'Target Reached'}])
            else:
                # Failed (stuck at current_pos)
                results.append([{'id': -1, 'score': 0.0, 'content': 'Failed'}])

        return results

# --- 2. THE TEST ---

# Create results directory if it doesn't exist
RESULTS_DIR = None  # Will be set in setup_module

def get_test_storage(results_dir: str) -> StorageConfig:
    """Returns a consistent storage config for tests."""
    return StorageConfig(
        base_dir=results_dir,
        dataset="test",
        run_id="sim_test",
        validation_frequency=1,
        checkpoint_frequency=1,
        plot_title="Full E2E Evolution Test",
    )

def get_test_config(storage: StorageConfig) -> EvolutionConfig:
    """Returns a consistent config for both tests."""
    config = EvolutionConfig(storage=storage)
    # Standard Params
    config.map_elites.batch_size = 10
    config.genetic.selection_k = 3

    # Toy Problem Search Space
    config.genetic.param_ranges.n_agents = (1, 5)
    config.genetic.param_ranges.decay = (0.1, 0.99)

    return config

def setup_module():
    global RESULTS_DIR
    RESULTS_DIR = tempfile.mkdtemp(prefix="evo_test_")

def teardown_module():
    global RESULTS_DIR
    if RESULTS_DIR and os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR, ignore_errors=True)

def test_evolution_solves_toy_problem():
    print("\n\n=== STARTING FULL SYSTEM SIMULATION ===")

    # 1. Configuration
    storage = get_test_storage(RESULTS_DIR)
    config = get_test_config(storage)
    config.n_generations = 3

    # 2. Data Setup
    # Targets are at distance 10, 15, 20.
    # Hard to reach if alpha (step_prob) is low!
    train_queries = ["10", "15", "20"]
    train_gt = [[10], [15], [20]]

    val_queries = ["12", "18"]
    val_gt = [[12], [18]]

    # 3. Components
    retriever = ToyStochasticRetriever()

    # Standard Evaluator (Real Class)
    # We subclass just to ensure calculate_metrics works with our integers
    class IntEvaluator(BaseEvaluator):
        def calculate_metrics(self, retrieved_nodes, ground_truth_ids, latency_sec=0):
            # Strict ID matching
            hit = 0.0
            if retrieved_nodes and retrieved_nodes[0]['id'] == ground_truth_ids[0]:
                hit = 1.0
            return {
                "Recall@10": hit,
                "Hit@10": hit,
                "MRR": hit,
                "latency": 10.0
            }
    evaluator = IntEvaluator(index_name="toy_sim")

    # Fitness: Prioritize Recall heavily
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

    # 5. RUN EVOLUTION
    print("  > Starting optimization...")
    best_genome = engine.optimize()

    # 6. VERIFICATION
    print(f"\n  > Best Genome Params: {best_genome.params}")
    print(f"  > Best Genome Fitness: {best_genome.fitness.quality_score}")

    assert best_genome.fitness.quality_score > 0.0, "Evolution failed to find ANY solution (random walk failed)"

    # B. Check Checkpoints
    assert os.path.exists(storage.latest_checkpoint_path), f"Final checkpoint missing at {storage.latest_checkpoint_path}"

    # C. Check Logs
    assert os.path.exists(storage.log_path), f"Log file missing at {storage.log_path}"
    with open(storage.log_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) >= 3, "Log should have at least 3 entries (Gen 0, 1, 2)"

    print("  Full system simulation passed!")

def test_resume_simulation():
    """
    Verifies we can load the result of the previous simulation and continue.
    """
    print("\n=== TESTING RESUME CAPABILITY ===")

    storage = get_test_storage(RESULTS_DIR)

    if not os.path.exists(storage.latest_checkpoint_path):
        pytest.skip("Run test_evolution_solves_toy_problem first")

    # 1. Config for RESUME
    # Extend generations from 3 to 5
    config = get_test_config(storage)
    config.n_generations = 5

    # 2. Load
    retriever = ToyStochasticRetriever()
    evaluator = BaseEvaluator(index_name="toy_sim")
    fitness_calc = FitnessCalculator(weights={"Recall@10": 1.0})

    run_manager = RunManager(storage)

    engine = EvolutionEngine.load_checkpoint(
        run_manager=run_manager,
        checkpoint_path=storage.latest_checkpoint_path,
        retriever=retriever,
        fitness_calculator=fitness_calc,
        evaluator=evaluator,
        train_query_ids=["10"], train_ground_truth=[[10]],  # Dummy data
        val_query_ids=["10"], val_ground_truth=[[10]],
        config=config,
    )

    # 3. Verify Load State
    # Previous test ran 3 gens (0, 1, 2). Saved state is Gen 2.
    print(f"  > Loaded Gen: {engine.evo_context.generation}")
    assert engine.evo_context.generation == 2, "Should resume from end of previous run"

    # 4. Continue
    engine.optimize()

    # 5. Verify Log Append
    with open(storage.log_path, 'r') as f:
        lines = f.readlines()
        # Should have Gen 0,1,2 (Run 1) + Gen 3,4 (Run 2) = 5 lines
        print(f"  > Total Log Lines: {len(lines)}")
        assert len(lines) >= 5, "Resume did not append correctly"

    print("  Resume simulation passed!")

if __name__ == "__main__":
    try:
        setup_module()
        test_evolution_solves_toy_problem()
        test_resume_simulation()
    finally:
        teardown_module()
