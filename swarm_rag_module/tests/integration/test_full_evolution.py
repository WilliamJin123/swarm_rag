import os
import shutil
import random
import numpy as np
import pytest
from typing import List, Any, Dict

from swarm_rag.evolution.engine import EvolutionEngine
from swarm_rag.evolution.types.config import DEFAULT_EVO_CONFIG
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

CKPT_FILE = "sim_test.pkl"
LOG_FILE = "sim_log.jsonl"
PLOT_FILE = "sim_plot.png"
PLOT_TITLE = "Full E2E Evolution Test"

def get_test_config():
    """Returns a consistent config for both tests to avoid default file leakage."""
    config = DEFAULT_EVO_CONFIG.copy()
    config.update({
        # Standard Params
        "population_size": 10,
        "selection_k": 3,
        "validation_frequency": 1,
        
        # KEY FIX: Always enforce these paths!
        "checkpoint_path": CKPT_FILE,
        "log_path": LOG_FILE,
        "plot_path": PLOT_FILE,
        "plot_title": PLOT_TITLE,
        
        # Toy Problem Search Space
        "param_ranges": {
            "n_agents": (1, 5),
            "alpha": (0.1, 0.99),
            "decay": (0.1, 0.99)
        }
    })
    return config

def setup_module():
    # Clean artifacts
    for f in [CKPT_FILE, LOG_FILE, PLOT_FILE]:
        if os.path.exists(f): os.remove(f)

def teardown_module():
    # Clean artifacts
    for f in [CKPT_FILE, LOG_FILE, PLOT_FILE]:
        if os.path.exists(f): os.remove(f)
        
    # Clean intermediate checkpoints
    base, ext = os.path.splitext(CKPT_FILE)
    for f in os.listdir("."):
        if f.startswith(base) and f.endswith(ext):
            os.remove(f)

def test_evolution_solves_toy_problem():
    print("\n\n=== STARTING FULL SYSTEM SIMULATION ===")
    
    # 1. Configuration
    config = get_test_config()
    config.update({
        "n_generations": 3,           # Short run
        "population_size": 10,        # Enough diversity
        "selection_k": 3,             # Tournament size
        "checkpoint_path": CKPT_FILE,
        "log_file": LOG_FILE,
        "plot_file": PLOT_FILE,
        "validation_frequency": 1,
        
        # Define the Search Space for our Toy Problem
        "param_ranges": {
            "n_agents": (1, 5),       # Try 1 to 5 agents
            "alpha": (0.1, 0.99),     # Step probability (we want high)
            "decay": (0.1, 0.99)      # Max steps (we need enough to reach target)
        }
    })

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
                "latency_ms": 10.0
            }

    evaluator = IntEvaluator(index_name="toy_sim")
    
    # Fitness: Prioritize Recall heavily
    fitness_calculator = FitnessCalculator(weights={"Recall@10": 1.0})

    # 4. Initialize Engine
    engine = EvolutionEngine(
        retriever=retriever,
        fitness_calculator=fitness_calculator,
        evaluator=evaluator,
        train_queries=train_queries,
        train_ground_truth=train_gt,
        val_queries=val_queries,
        val_ground_truth=val_gt,
        config=config
    )

    # 5. RUN EVOLUTION
    print("  > Starting optimization...")
    best_genome = engine.optimize()

    # 6. VERIFICATION
    

    print(f"\n  > Best Genome Params: {best_genome.params}")
    print(f"  > Best Genome Fitness: {best_genome.fitness.quality_score}")
    
    assert best_genome.fitness.quality_score > 0.0, "Evolution failed to find ANY solution (random walk failed)"
    
    # B. Check Checkpoints
    assert os.path.exists(os.path.join("evo_results", CKPT_FILE)), "Final checkpoint missing"
    
    # C. Check Logs
    log_f = os.path.join("evo_results", LOG_FILE)
    assert os.path.exists(log_f), "Log file missing"
    with open(log_f, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) >= 3, "Log should have at least 3 entries (Gen 0, 1, 2)"
        
    print("  ✓ Full system simulation passed!")

def test_resume_simulation():
    """
    Verifies we can load the result of the previous simulation and continue.
    """
    print("\n=== TESTING RESUME CAPABILITY ===")
    
    if not os.path.exists(os.path.join("evo_results",CKPT_FILE)):
        pytest.skip("Run test_evolution_solves_toy_problem first")

    # 1. Config for RESUME
    # Extend generations from 3 to 5
    config = get_test_config()
    config["n_generations"] = 5
    config["log_file"] = LOG_FILE # Append to same log
    
    # 2. Load
    retriever = ToyStochasticRetriever()
    evaluator = BaseEvaluator(index_name="toy_sim") # Mock/Real doesn't matter here, just class structure
    fitness_calc = FitnessCalculator(weights={"Recall@10": 1.0})
    
    engine = EvolutionEngine.load_checkpoint(
        checkpoint_path=CKPT_FILE,
        retriever=retriever,
        fitness_calculator=fitness_calc,
        evaluator=evaluator,
        train_queries=["10"], train_ground_truth=[[10]], # Dummy data
        val_queries=["10"], val_ground_truth=[[10]],
        config=config
    )
    
    # 3. Verify Load State
    # Previous test ran 3 gens (0, 1, 2). Saved state is Gen 2.
    print(f"  > Loaded Gen: {engine.evo_context.generation}")
    assert engine.evo_context.generation == 2, "Should resume from end of previous run"
    
    # 4. Continue
    engine.optimize()
    
    # 5. Verify Log Append
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()
        # Should have Gen 0,1,2 (Run 1) + Gen 3,4 (Run 2) = 5 lines
        print(f"  > Total Log Lines: {len(lines)}")
        assert len(lines) >= 5, "Resume did not append correctly"
        
    print("  ✓ Resume simulation passed!")

if __name__ == "__main__":
    try:
        setup_module()
        test_evolution_solves_toy_problem()
        test_resume_simulation()
    finally:
        pass