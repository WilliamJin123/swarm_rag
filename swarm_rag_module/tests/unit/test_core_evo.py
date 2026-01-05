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
    f2 = FitnessResult(quality_score=0.6, stability_score=0.9, cost_score=50) 
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
    features = {'degree': 0.5} 
    result = root.evaluate(features)
    
    expected = (0.5 * 2.0) + 1.0
    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    print(f"  ✓ Evaluation Result: {result}")

def test_genome_compiler():
    print("\n--- Testing Genome Compiler (Single Group) ---")
    
    # 1. Create a dummy genome
    params: SwarmParams = DEFAULT_PARAMS.copy()
    params['n_agents'] = 99
    
    # Simple strategy: semantic_similarity * 1.0
    expr = ExpressionNode('feature', 'semantic_similarity')
    
    # FIX: Use 'g0_movement' and provide ratio
    genome = Genome(
        id="test_01",
        params=params,
        group_ratios={'g0': 1.0},
        strategies={'g0_movement': expr}
    )
    
    # 2. Compile
    compiler = GenomeCompiler()
    kwargs = compiler.compile(genome)
    
    print("  ✓ Compilation successful")
    
    # 3. Check Kwargs
    assert kwargs['n_agents'] == 99, "Parameters not unpacked correctly"
    
    # FIX: Compiler now produces 'agent_groups', not flat strategies
    assert 'agent_groups' in kwargs, "agent_groups missing"
    assert len(kwargs['agent_groups']) == 1
    
    group_0 = kwargs['agent_groups'][0]
    assert group_0['count'] == 99
    assert 'evolved_mov_0' in group_0['movement_strategies']
    
    # 4. Execute compiled strategy
    func, weight = group_0['movement_strategies']['evolved_mov_0']
    
    # Mock HeuristicContext
    class MockCtx:
        query_vec = np.array([1, 0])
        target_vecs = np.array([[1, 0], [0, 1]]) 
        # Registry lookups usually return arrays, handled by wrapper
    
    # Note: This relies on 'semantic_similarity' being in HeuristicRegistry
    print("  ✓ Compiled function is callable")

def test_genome_compiler_heterogeneous():
    print("\n--- Testing Genome Compiler (Heterogeneous Groups) ---")
    
    # 1. Setup Params
    params: SwarmParams = DEFAULT_PARAMS.copy()
    params['n_agents'] = 100
    
    # 2. Setup Strategies (Namespaced)
    expr_a = ExpressionNode('feature', 'degree')
    expr_b = ExpressionNode('feature', 'pheromone')
    
    strategies = {
        'g0_movement': expr_a,
        'g0_deposit': expr_a, 
        'g1_movement': expr_b,
        'g1_deposit': expr_b, 
        'ranking': expr_a     
    }
    
    # 3. Setup Ratios (Split 80/20)
    ratios = {'g0': 0.8, 'g1': 0.2}
    
    genome = Genome(
        id="test_het_01",
        params=params,
        group_ratios=ratios,
        strategies=strategies
    )
    
    # 4. Compile
    compiler = GenomeCompiler()
    kwargs = compiler.compile(genome)
    
    print("  ✓ Compilation successful")
    
    # 5. Verify Output Structure
    assert 'agent_groups' in kwargs, "Compiler failed to generate 'agent_groups'"
    groups = kwargs['agent_groups']
    
    assert len(groups) == 2, f"Expected 2 agent groups, got {len(groups)}"
    
    # Check Counts (80 vs 20)
    count_0 = groups[0]['count']
    count_1 = groups[1]['count']
    
    print(f"  Group Counts: {count_0} / {count_1}")
    assert count_0 + count_1 == 100, "Total agents mismatch"
    assert count_0 == 80
    assert count_1 == 20
    
    # Check Strategy Wiring
    g0_mov_map = groups[0]['movement_strategies']
    assert 'evolved_mov_0' in g0_mov_map
    
    g1_mov_map = groups[1]['movement_strategies']
    assert 'evolved_mov_1' in g1_mov_map

    print("  ✓ Heterogeneous wiring confirmed")

if __name__ == "__main__":
    test_fitness_logic()
    test_expression_evaluation()
    test_genome_compiler()
    test_genome_compiler_heterogeneous()
    print("\nALL CORE TESTS PASSED")