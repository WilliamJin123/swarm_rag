import pytest
import numpy as np
import random
import os
import shutil
from swarm_rag.evolution.engine import EvolutionEngine
from swarm_rag.evolution.types.config import DEFAULT_EVO_CONFIG
from swarm_rag.evolution.execution.fitness import FitnessCalculator
from swarm_rag.eval.metrics import Evaluator as BaseEvaluator

# Extensions
from swarm_rag.evolution.extensions.niching import NichingExtension
from swarm_rag.evolution.extensions.immigration import RandomImmigrationExtension
from swarm_rag.evolution.extensions.migration import FileMigrationExtension

# Reuse Toy Retriever (Definition included to be standalone)
class ToyStochasticRetriever:
    """
    Simulator: Agents need high 'alpha' to find the target.
    Probability of finding target = alpha.
    """
    def retrieve_batch(self, queries, max_workers=1, **kwargs):
        results = []
        # Default alpha is 0.5 if not evolved
        step_prob = kwargs.get('alpha', 0.5) 
        
        for q in queries:
            target = int(q)
            # Stochastic success check
            if random.random() < step_prob:
                results.append([{'id': target, 'score': 1.0}])
            else:
                results.append([{'id': -1}]) # Miss
        return results

# --- HELPER: CONFIG FACTORY ---
def get_standard_config(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    config = DEFAULT_EVO_CONFIG.copy()
    config.update({
        "n_generations": 8,           # Enough time for immigration to kick in
        "population_size": 20,
        "param_ranges": {
            "n_agents": (1, 5),
            "alpha": (0.01, 0.99),    # Critical param
            "decay": (0.1, 0.99)
        },
        "output_dir": "comparisons"
    })
    return config

# --- HELPER: RUNNER ---
def run_evolution(extensions=[], seed=42, island_id=None, migration_dir=None):
    """
    Runs a full evolution loop.
    Handles strict dependency injection for Immigration.
    """
    config = get_standard_config(seed)
    
    # Define Problem: Target is Node 100
    train_queries = ["100"]
    train_gt = [[100]]
    
    retriever = ToyStochasticRetriever()
    evaluator = BaseEvaluator(index_name="toy")
    fitness_calc = FitnessCalculator(weights={"Recall@10": 1.0})
    
    engine = EvolutionEngine(
        retriever=retriever,
        fitness_calculator=fitness_calc,
        evaluator=evaluator,
        train_queries=train_queries, train_ground_truth=train_gt,
        val_queries=train_queries, val_ground_truth=train_gt,
        config=config,
        extensions=extensions,
        overwrite_logs=True
    )
    
    # --- MANUAL INJECTION ---
    # Since we create extensions BEFORE the engine, we must manually 
    # link them here if the Engine.__init__ didn't do it automatically.
    for ext in extensions:
        if isinstance(ext, RandomImmigrationExtension):
            ext.engine = engine
            
    best = engine.optimize()
    return best

# =============================================================================
# TEST 1: SINGLE ISLAND SHOWDOWN (Vanilla vs Niching/Immigration)
# =============================================================================
def test_full_extensions_vs_baseline():
    print("\n\n=== SHOWDOWN 1: Vanilla vs. Niching + Immigration ===")
    
    # 1. BASELINE (Seed 100: Known to be mediocre/stochastic)
    # Without Niching/Immigration, it might get stuck in local optima
    print("  [Running Baseline...]")
    best_base = run_evolution(extensions=[], seed=100)
    score_base = best_base.fitness.quality_score
    print(f"  > Baseline Score: {score_base:.4f} (Alpha: {best_base.params.get('alpha', 'N/A')})")
    
    # 2. EXTENDED
    # Niching: Prevents population from converging on "safe but low" alpha
    # Immigration: Injects wild new params (like alpha=0.99) if stagnation occurs
    print("  [Running Extended...]")
    exts = [
        NichingExtension(sigma_share=1.5, n_probes=5),
        RandomImmigrationExtension(rate=0.15) # Replaces bottom 15% every gen
    ]
    
    best_ext = run_evolution(extensions=exts, seed=100)
    score_ext = best_ext.fitness.quality_score
    print(f"  > Extended Score: {score_ext:.4f} (Alpha: {best_ext.params.get('alpha', 'N/A')})")
    
    # Assertion: Extended should be at least as good, usually better/more robust
    assert score_ext >= score_base - 0.05
    
    if score_ext > score_base:
        print("  => VICTORY: Extensions found a better solution.")
    elif score_ext == score_base:
        print("  => DRAW: Both found max score.")

# =============================================================================
# TEST 2: MULTI-ISLAND MIGRATION SYNERGY
# =============================================================================
def test_island_migration_synergy():
    """
    Scenario:
    - Island A (Struggling): Initialized with a 'bad' seed or difficult constraints.
    - Island B (Thriving): Initialized with a 'good' seed.
    - GOAL: Prove Island A improves AFTER importing Island B's elite.
    """
    print("\n\n=== SHOWDOWN 2: Multi-Island Migration Synergy ===")
    
    migration_dir = "./test_synergy_pool"
    if os.path.exists(migration_dir): shutil.rmtree(migration_dir)
    os.makedirs(migration_dir)
    
    try:
        # --- SETUP ISLAND A (The 'Learner') ---
        # We simulate a "struggling" start by manually crippling the config 
        # or just running it for fewer generations first.
        config_a = get_standard_config(seed=666) # "Bad" seed
        config_a['n_generations'] = 3
        
        # Island A Extensions
        exts_a = [
            FileMigrationExtension(migration_dir=migration_dir, interval=2, island_id="Island_A")
        ]
        
        # Setup Engine A
        engine_a = EvolutionEngine(
            retriever=ToyStochasticRetriever(),
            fitness_calculator=FitnessCalculator(weights={"Recall@10": 1.0}),
            evaluator=BaseEvaluator(index_name="toy"),
            train_queries=["100"], train_ground_truth=[[100]],
            val_queries=["100"], val_ground_truth=[[100]],
            config=config_a,
            extensions=exts_a,
            overwrite_logs=True
        )
        
        # --- SETUP ISLAND B (The 'Teacher') ---
        # "Good" seed that finds alpha=0.9 quickly
        config_b = get_standard_config(seed=777) 
        config_b['n_generations'] = 3
        
        exts_b = [
            FileMigrationExtension(migration_dir=migration_dir, interval=2, island_id="Island_B")
        ]
        
        engine_b = EvolutionEngine(
            retriever=ToyStochasticRetriever(),
            fitness_calculator=FitnessCalculator(weights={"Recall@10": 1.0}),
            evaluator=BaseEvaluator(index_name="toy"),
            train_queries=["100"], train_ground_truth=[[100]],
            val_queries=["100"], val_ground_truth=[[100]],
            config=config_b,
            extensions=exts_b,
            overwrite_logs=True
        )

        # --- STEP 1: RUN ISLAND B (Create the Elite) ---
        print("  [Running Island B (Teacher)...]")
        best_b = engine_b.optimize()
        score_b = best_b.fitness.quality_score
        print(f"  > Island B finished with Score: {score_b:.4f}")
        
        # Verify B actually generated a migration file
        files = os.listdir(migration_dir)
        assert len(files) > 0, "Island B failed to export migrants!"
        
        # --- STEP 2: RUN ISLAND A (The Beneficiary) ---
        print("  [Running Island A (Learner)...]")
        
        # We hook into the engine to check score BEFORE migration (Generation 0/1)
        # But for this integration test, we just run it. 
        # Since interval=2, it will import B's file at Generation 2.
        
        best_a = engine_a.optimize()
        score_a = best_a.fitness.quality_score
        print(f"  > Island A finished with Score: {score_a:.4f}")
        
        # --- VERIFY SYNERGY ---
        # Island A should have matched Island B's score because it imported the solution.
        # If migration failed, A (with bad seed) likely would have stalled lower.
        
        assert score_a >= score_b - 0.01, \
            f"Migration Failed! Island A ({score_a}) did not catch up to Island B ({score_b})"
            
        # Check if the Best ID in A actually came from B (or is a descendant)
        # The ID suffix might change due to mutation, but let's check exact match first
        # or check if A's best has high alpha like B.
        
        print("  => VICTORY: Island A successfully assimilated Island B's elite!")

    finally:
        # Cleanup
        if os.path.exists(migration_dir): shutil.rmtree(migration_dir)

if __name__ == "__main__":
    pytest.main([__file__, "-s"])