import random
import numpy as np
from typing import List
from dataclasses import dataclass

from swarm_rag.evolution.types.genome import Genome, SwarmParams, DEFAULT_PARAMS
from swarm_rag.evolution.types.config import EvolutionContext, EvolutionConfigDict, DEFAULT_EVO_CONFIG
from swarm_rag.evolution.types.expressions import ExpressionNode
from swarm_rag.evolution.execution.strategies import GeneticStrategies
from swarm_rag.evolution.execution.fitness import FitnessResult

# Helper to make dummy genomes
def make_genome(id, q_score, n_agents=10, ratio_g0=0.5) -> Genome:
    params = DEFAULT_PARAMS.copy()
    params['n_agents'] = n_agents
    
    # Create simple dummy strategy
    dummy_tree = ExpressionNode('const', 1.0)
    strategies = {
        'g0_movement': dummy_tree.copy(),
        'g0_deposit': dummy_tree.copy(),
        'ranking': dummy_tree.copy()
    }
    
    # Ratios
    ratios = {'g0': ratio_g0, 'g1': 1.0 - ratio_g0}
    
    g = Genome(
        id=str(id), 
        params=params, 
        group_ratios=ratios,
        strategies=strategies
    )
    g.fitness = FitnessResult(q_score, 0.5, 100)
    return g

def test_selection_ops():
    print("\n--- Testing Selection Strategies ---")
    
    # CHANGE 1: Increase "best" density (50% best, 50% bad)
    # This guarantees high probability of winning a tournament of size 5
    pop = [make_genome(f"bad_{i}", 0.1) for i in range(5)]
    for i in range(5):
        pop.append(make_genome(f"best_{i}", 0.9))
    
    ctx = EvolutionContext(
        population=pop,
        generation=0,
        config=DEFAULT_EVO_CONFIG
    )
    
    # 1. Tournament
    print("  Testing Tournament...")
    ctx.config['selection_k'] = 5
    wins = 0
    for _ in range(100):
        # We grab the first winner from the list
        selected = GeneticStrategies.tournament_selection(ctx, k=1)[0]
        if "best" in selected.id: wins += 1
    
    print(f"  ✓ Tournament selected 'best' {wins}/100 times")
    # With 50% best and K=5, prob is ~96.8%. This assertion is now safe.
    assert wins > 80 

    # 2. Truncation (Requires Sorted Pop)
    print("  Testing Truncation...")
    pop.sort(key=lambda g: g.fitness, reverse=True) # Sort Best First
    ctx.population = pop # Update context
    
    # Truncation logic (e.g. top 20%) should ALWAYS pick a 'best' 
    # since half the population is 'best'
    selected = GeneticStrategies.truncation_selection(ctx, k=1)[0]
    print(f"  ✓ Truncation selected: {selected.id}")
    assert "best" in selected.id

def test_crossover_ops():
    print("\n--- Testing Crossover (Uniform Mix) ---")
    
    # Parent 1: 100% g0
    parent1 = make_genome("p1", 0.5, ratio_g0=1.0)
    # Parent 2: 0% g0
    parent2 = make_genome("p2", 0.5, ratio_g0=0.0)
    
    ctx = EvolutionContext(population=[], generation=0, config=DEFAULT_EVO_CONFIG)
    
    # Generate statistics over many children
    ratio_sum = 0
    n_trials = 100
    
    for _ in range(n_trials):
        child = GeneticStrategies.uniform_parameter_mix(parent1, parent2, ctx)
        ratio_sum += child.group_ratios['g0']
        
    avg_ratio = ratio_sum / n_trials
    
    print(f"  Avg g0 Ratio: {avg_ratio:.2f} (Expected ~0.5)")
    
    assert 0.4 < avg_ratio < 0.6, "Crossover bias detected"

def test_mutation_ops():
    print("\n--- Testing Mutation ---")
    
    g = make_genome("mutant", 0.5, n_agents=20)
    # Initial ratio 0.5
    
    ctx = EvolutionContext(population=[], generation=0, config=DEFAULT_EVO_CONFIG)
    ctx.config['mutation_rate'] = 1.0 # Force mutation
    
    # Needed for strategy mutation
    ctx.expression_features = {'movement': ['degree'], 'deposit': ['degree'], 'ranking': ['degree']}
    
    # Run mutation
    mutated = GeneticStrategies.expression_tree_mutation(g, ctx)
    
    # Check Ratio Mutation
    new_ratio = mutated.group_ratios['g0']
    print(f"  Original Ratio: 0.5 -> Mutated: {new_ratio}")
    assert new_ratio != 0.5, "Group ratio did not mutate"
    
    # Check Parameter Mutation
    print(f"  Original n_agents: 20 -> Mutated: {mutated.params['n_agents']}")
    assert mutated.params['n_agents'] != 20, "Parameter did not mutate"
    
    print("  ✓ Mutation operators working")

if __name__ == "__main__":
    test_selection_ops()
    test_crossover_ops()
    test_mutation_ops()
    print("\nALL OPS TESTS PASSED")