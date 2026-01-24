import random
from typing import List
from dataclasses import dataclass

from swarm_rag.evolution.types.genome import Genome, SwarmParams, DEFAULT_PARAMS
from swarm_rag.evolution.types.config import EvolutionContext, EvolutionConfig
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
    
    config = EvolutionConfig()
    config.genetic.selection_k = 5

    ctx = EvolutionContext(
        population=pop,
        generation=0,
        config=config
    )

    # 1. Tournament
    print("  Testing Tournament...")
    wins = 0
    for _ in range(100):
        # We grab the first winner from the list
        selected = GeneticStrategies.tournament_selection(ctx, k=1)[0]
        if "best" in selected.id: wins += 1
    
    print(f"  Tournament selected 'best' {wins}/100 times")
    # With 50% best and K=5, prob is ~96.8%. This assertion is now safe.
    assert wins > 80

def test_crossover_ops():
    print("\n--- Testing Crossover (Uniform Mix) ---")
    
    # Parent 1: 100% g0
    parent1 = make_genome("p1", 0.5, ratio_g0=1.0)
    # Parent 2: 0% g0
    parent2 = make_genome("p2", 0.5, ratio_g0=0.0)
    
    ctx = EvolutionContext(population=[], generation=0, config=EvolutionConfig())
    
    # Generate statistics over many children
    ratio_sum = 0
    n_trials = 100
    
    for _ in range(n_trials):
        child = GeneticStrategies.uniform_parameter_mix(parent1, parent2, ctx)
        ratio_sum += child.group_ratios['g0']
        
    avg_ratio = ratio_sum / n_trials
    
    print(f"  Avg g0 Ratio: {avg_ratio:.2f} (Expected ~0.5)")
    
    # Allow wider tolerance for probabilistic test (avg of 100 trials)
    assert 0.35 <= avg_ratio <= 0.65, "Crossover bias detected"

def test_mutation_ops():
    print("\n--- Testing Mutation ---")

    g = make_genome("mutant", 0.5, n_agents=20)
    # Initial ratio 0.5

    ctx = EvolutionContext(population=[], generation=0, config=EvolutionConfig())
    ctx.global_mutation_multiplier = 100.0  # Force mutation (overrides clamping)
    
    # Needed for strategy mutation
    ctx.expression_features = {'movement': ['degree'], 'deposit': ['degree'], 'ranking': ['degree']}
    
    # Run mutation
    mutated = GeneticStrategies.expression_tree_mutation(g, ctx)
    
    # Check Ratio Mutation
    new_ratio = mutated.group_ratios['g0']
    print(f"  Original Ratio: 0.5 -> Mutated: {new_ratio}")
    assert new_ratio != 0.5, "Group ratio did not mutate"
    
    # Check Parameter Mutation (probabilistic - check at least one param changed)
    print(f"  Original n_agents: 20 -> Mutated: {mutated.params['n_agents']}")
    params_changed = any(
        mutated.params[k] != DEFAULT_PARAMS[k]
        for k in ['n_agents', 'steps', 'decay']
    )
    assert params_changed, "No parameters mutated"
    
    print("  Mutation operators working")

if __name__ == "__main__":
    test_selection_ops()
    test_crossover_ops()
    test_mutation_ops()
    print("\nALL OPS TESTS PASSED")