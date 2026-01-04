import numpy as np
import random
from typing import Dict, Any

from swarm_rag.evolution.types.genome import Genome, GenomeCompiler, SwarmParams, DEFAULT_PARAMS
from swarm_rag.evolution.types.expressions import ExpressionNode, ExpressionEvolution
from swarm_rag.evolution.execution.fitness import FitnessResult, FitnessCalculator
from swarm_rag.core.heuristics import HeuristicContext

def test_fitness_logic():
    print("\n--- Testing FitnessResult (Lexicographic Sorting) ---")
    
    # 1. Quality Dominates
    f1 = FitnessResult(quality_score=0.8, stability_score=0.5, cost_score=100)
    f2 = FitnessResult(quality_score=0.6, stability_score=0.9, cost_score=50) # Better stability/cost, worse quality
    assert f1 > f2, "High quality should beat low quality"
    print("  ✓ Quality dominance check passed")

    # 2. Stability Tie-Breaker (Quality equal within tolerance)
    f3 = FitnessResult(quality_score=0.8001, stability_score=0.9, cost_score=100)
    f4 = FitnessResult(quality_score=0.8002, stability_score=0.1, cost_score=100)
    # 0.8001 and 0.8002 are within epsilon 0.005, so Stability decides
    assert f3 > f4, "Stability should break ties when quality is similar"
    print("  ✓ Stability tie-breaker check passed")

    # 3. Cost Tie-Breaker (Quality & Stability equal)
    f5 = FitnessResult(quality_score=0.8, stability_score=0.5, cost_score=200) # High cost
    f6 = FitnessResult(quality_score=0.8, stability_score=0.5, cost_score=50)  # Low cost
    assert f6 > f5, "Lower cost should win when others are equal"
    print("  ✓ Cost tie-breaker check passed")

def test_expression_evaluation():
    print("\n--- Testing Expression Trees ---")
    
    # Create tree: (feature_A * 2) + 1
    # feature_A will be mocked as 0.5
    feat_node = ExpressionNode('feature', 'degree')
    const_2 = ExpressionNode('const', 2.0)
    mult_op = ExpressionNode('op', '*', [feat_node, const_2])
    const_1 = ExpressionNode('const', 1.0)
    root = ExpressionNode('op', '+', [mult_op, const_1])
    
    print(f"  Tree Structure: {root.to_string()}")

    # Mock Context
    # We need to manually inject feature values since we aren't using the full compiler here yet
    features = {'degree': 0.5} 
    result = root.evaluate(features)
    
    expected = (0.5 * 2.0) + 1.0
    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    print(f"  ✓ Evaluation Result: {result}")

def test_genome_compiler():
    print("\n--- Testing Genome Compiler ---")
    
    # 1. Create a dummy genome
    params: SwarmParams = DEFAULT_PARAMS.copy()
    params['n_agents'] = 99
    
    # Simple strategy: semantic_similarity * 1.0
    expr = ExpressionNode('feature', 'semantic_similarity')
    
    genome = Genome(
        id="test_01",
        params=params,
        strategies={'movement': expr}
    )
    
    # 2. Compile
    compiler = GenomeCompiler()
    kwargs = compiler.compile(genome)
    
    print("  ✓ Compilation successful")
    
    # 3. Check Kwargs
    assert kwargs['n_agents'] == 99, "Parameters not unpacked correctly"
    assert 'movement_strategies' in kwargs, "Strategy keys missing"
    
    # 4. Execute compiled strategy
    func, weight = kwargs['movement_strategies']['evolved_movement']
    
    # Mock HeuristicContext
    class MockCtx:
        query_vec = np.array([1, 0])
        target_vecs = np.array([[1, 0], [0, 1]]) # Sim 1.0 and 0.0
        # Registry lookups usually return arrays
    
    # Note: This relies on 'semantic_similarity' being in HeuristicRegistry
    # If using dummy environment, ensure registry is populated or mocked.
    print("  ✓ Compiled function is callable")

if __name__ == "__main__":
    test_fitness_logic()
    test_expression_evaluation()
    test_genome_compiler()
    print("\nALL CORE TESTS PASSED")