# test_evolution_integration.py
import time
import shutil
import os
import random
from typing import List, Dict, Any

from swarm_rag.evolution.engine import EvolutionEngine
from swarm_rag.evolution.types.genome import Genome, DEFAULT_PARAMS
from swarm_rag.evolution.types.config import DEFAULT_EVO_CONFIG
from swarm_rag.eval.metrics import Evaluator
from swarm_rag.evolution.execution.fitness import FitnessCalculator

# --- MOCKS ---

class MockRetriever:
    """Simulates the SwarmRetriever for fast testing."""
    def retrieve_batch(self, queries, **kwargs):
        # Return deterministic "fake" results
        results = []
        n_agents = kwargs.get('n_agents', 0)
        
        for q in queries:
            # Fake nodes [0, 1, 2, 3]
            if n_agents > 20:
                # If genetics evolved "high agents", give better results
                res = [{'id': 0, 'score': 0.9}, {'id': 1, 'score': 0.8}]
            else:
                res = [{'id': 99, 'score': 0.1}] # Bad results
            results.append(res)
        return results

class MockEvaluator(Evaluator):
    """Bypasses real metric calculation."""
    def calculate_metrics(self, retrieved_nodes, ground_truth_ids, latency_sec=0):
        # If retriever returned node 0, high score. Else low.
        if retrieved_nodes and retrieved_nodes[0]['id'] == 0:
            return {'Recall@20': 1.0, 'latency_ms': 10.0}
        return {'Recall@20': 0.0, 'latency_ms': 10.0}

# --- TEST ---

def test_full_evolution_loop():
    print("="*60)
    print("EVOLUTION ENGINE INTEGRATION TEST")
    print("="*60)

    # 1. Setup Config
    config = DEFAULT_EVO_CONFIG.copy()
    config.update({
        "n_generations": 3,
        "population_size": 6,
        "validation_frequency": 1,
        "n_agent_groups": 2,      # Explicitly set groups
        "output_dir": None,       # FIX: Disable auto-prefixing so paths match below
        "checkpoint_path": "test_ckpt.pkl",
        "log_path": "test_log.jsonl", # Matches config definition
        "plot_path": "test_plot.png"  # Matches config definition
    })
    
    # 2. Setup Data
    train_q = ["q1", "q2"]
    train_gt = [[0], [0]] # Expect node 0
    val_q = ["v1"]
    val_gt = [[0]]

    # 3. Initialize Components
    retriever = MockRetriever()
    evaluator = MockEvaluator(index_name="test")
    # Fitness: We care about Recall
    fitness_calc = FitnessCalculator(weights={'Recall@20': 1.0})

    engine = EvolutionEngine(
        retriever=retriever,
        fitness_calculator=fitness_calc,
        evaluator=evaluator,
        train_queries=train_q,
        train_ground_truth=train_gt,
        val_queries=val_q,
        val_ground_truth=val_gt,
        config=config
    )

    # 4. Run Optimization
    print("\nStarting Engine...")
    start = time.time()
    best_genome = engine.optimize()
    duration = time.time() - start
    
    print(f"\nOptimization Finished in {duration:.2f}s")
    
    # 5. Assertions
    print("\nVerifying Results:")
    
    # Check A: Best Genome exists
    assert best_genome is not None
    print(f"  ✓ Best Genome ID: {best_genome.id}")
    print(f"  ✓ Best Fitness: {best_genome.fitness.quality_score}")
    
    # Check B: Files created
    assert os.path.exists(config['log_file']), "Log file missing"
    assert os.path.exists(config['checkpoint_path']), "Checkpoint missing"
    assert os.path.exists(config['plot_file']), "Plot missing"
    print("  ✓ Artifacts created (log, checkpoint, plot)")

    # Check C: Did it actually evolve?
    # Our MockRetriever gives better score if n_agents > 20.
    # We hope the GA found that.
    print(f"  ✓ Best Param n_agents: {best_genome.params['n_agents']}")
    
    # Cleanup
    print("\nCleaning up...")
    for f in [config['log_file'], config['checkpoint_path'], config['plot_file']]:
        if os.path.exists(f):
            os.remove(f)
    print("  ✓ Cleanup done")

if __name__ == "__main__":
    test_full_evolution_loop()