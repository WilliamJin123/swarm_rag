
# test_integration_complex.py
import os
import shutil
import pickle
import numpy as np
from typing import List, Dict, Any

from swarm_rag.evolution.engine import EvolutionEngine
from swarm_rag.evolution.types.genome import Genome
from swarm_rag.evolution.types.config import DEFAULT_EVO_CONFIG
from swarm_rag.evolution.execution.evaluator import PopulationEvaluator
from swarm_rag.eval.metrics import Evaluator as BaseEvaluator
from swarm_rag.evolution.execution.fitness import FitnessCalculator

# --- MOCKS ---

class MockRetriever:
    def retrieve_batch(self, queries, **kwargs):
        # Return results that simulate "unstable" performance
        # Query 1: High Score, Query 2: Low Score -> High Variance
        results = []
        for i, _ in enumerate(queries):
            if i % 2 == 0:
                results.append([{'id': 0, 'score': 1.0}]) # Perfect
            else:
                results.append([{'id': 99, 'score': 0.0}]) # Fail
        return results

class MockBaseEvaluator(BaseEvaluator):
    def calculate_metrics(self, retrieved_nodes, ground_truth_ids, latency_sec=0):
        """
        FIX: Argument names must match exactly what PopulationEvaluator passes.
        """
        # If result is correct (id 0), give 1.0, else 0.0
        score = 1.0 if retrieved_nodes and retrieved_nodes[0]['id'] == 0 else 0.0
        return {'Recall@20': score, 'latency_ms': 50.0}

# --- TESTS ---

# Create results directory if it doesn't exist
RESULTS_DIR = "evo_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CKPT_FILE = os.path.join(RESULTS_DIR, "complex_test.pkl")
LOG_FILE = os.path.join(RESULTS_DIR, "complex_log.jsonl")

def test_variance_calculation_flow():
    print("\n--- Testing Variance/Stability Flow ---")
    
    # 1. Setup
    retriever = MockRetriever()
    base_eval = MockBaseEvaluator(index_name="test")
    # Fitness that cares about Stability
    fitness_calc = FitnessCalculator(weights={'Recall@20': 1.0}) 
    
    pop_eval = PopulationEvaluator(retriever, base_eval, fitness_calc)
    
    # 2. Create a dummy genome
    g = Genome(id="unstable_agent")
    class MockStrategy:
        def size(self): return 10  # Return a fake complexity size
        def copy(self): return self
    g.group_ratios = {'g0': 1.0}
    g.strategies = {
        'g0_movement': MockStrategy(),
        'g0_deposit': MockStrategy()
    }
    # Manually compile empty cache to pass checks
    g._compiled_cache = {
        'g0_movement': lambda x: 0, 
        'g0_deposit': lambda x: 0
    }
    
    # 3. Evaluate on 2 queries (One Good, One Bad)
    queries = ["q1", "q2"]
    ground_truth = [[0], [0]]
    
    pop_eval.evaluate([g], queries, ground_truth)
    
    # 4. Check Metrics
    print(f"  Recall Mean: {g.metrics['Recall@20']} (Expected 0.5)")
    print(f"  Variance:    {g.metrics['variance']} (Expected 0.25)")
    print(f"  Stability:   {g.fitness.stability_score} (Expected 0.75)")
    
    assert g.metrics['Recall@20'] == 0.5, "Mean calculation wrong"
    assert g.metrics['variance'] == 0.25, "Variance calculation wrong"
    # Stability = 1.0 - Variance
    assert g.fitness.stability_score == 0.75, "Stability score logic failed"
    print("  ✓ Stability metrics flow confirmed")

def test_checkpoint_resume():
    print("\n--- Testing Checkpoint Save & Resume ---")
    
    # 1. Config for Short Run
    config = DEFAULT_EVO_CONFIG.copy()
    config['n_generations'] = 4
    config['checkpoint_path'] = CKPT_FILE
    config['log_path'] = LOG_FILE
    config['population_size'] = 4
    config['plot_path'] = os.path.join(RESULTS_DIR, "evo_plot_complex.png")
    
    # 2. Run Initial Engine (Gens 0-1)
    print("  Running initial batch (Gen 0-1)...")
    engine = EvolutionEngine(
        retriever=MockRetriever(),
        fitness_calculator=FitnessCalculator({'Recall@20': 1.0}),
        evaluator=MockBaseEvaluator("test"),
        train_query_ids=["q1"], train_ground_truth=[[0]],
        val_query_ids=["v1"], val_ground_truth=[[0]],
        config=config
    )
    
    config['n_generations'] = 2
    engine.optimize()
    
    assert os.path.exists(CKPT_FILE), "Checkpoint not created"
    
    # 3. Load Checkpoint
    print("  Loading checkpoint...")
    new_config = config.copy()
    new_config['n_generations'] = 4 # Extend to 4 gens
    
    loaded_engine = EvolutionEngine.load_checkpoint(
        checkpoint_path=CKPT_FILE,
        retriever=MockRetriever(), # Re-inject dependencies
        evaluator=MockBaseEvaluator("test"),
        fitness_calculator=FitnessCalculator({'Recall@20': 1.0}),
        train_query_ids=["q1"], train_ground_truth=[[0]],
        val_query_ids=["v1"], val_ground_truth=[[0]],
        config=new_config # Pass new config
    )
    
    # 4. Verify State
    print(f"  Resumed at Gen: {loaded_engine.evo_context.generation}")
    assert loaded_engine.evo_context.generation == 1, "Did not load the correct generation index"
    
    # 5. Continue Run
    print("  Continuing evolution (Gen 2-3)...")
    loaded_engine.optimize()
    
    # 6. Verify Log has 4 entries (0, 1, 2, 3)
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()
        print(f"  Total log lines: {len(lines)}")
        assert len(lines) >= 4, "Resume did not append to log correctly"
        
    print("  ✓ Checkpoint resume successful")

def cleanup():
    for f in [CKPT_FILE, LOG_FILE, os.path.join(RESULTS_DIR, "evo_plot_complex.png")]:
        if os.path.exists(f):
            os.remove(f)
    # Clean intermediate checkpoints
    base, ext = os.path.splitext(CKPT_FILE)
    for f in os.listdir(RESULTS_DIR):
        if f.startswith(os.path.basename(base)) and f.endswith(ext):
            os.remove(os.path.join(RESULTS_DIR, f))

if __name__ == "__main__":
    try:
        test_variance_calculation_flow()
        test_checkpoint_resume()
        print("\nALL COMPLEX INTEGRATION TESTS PASSED")
    finally:
        # cleanup()
        pass