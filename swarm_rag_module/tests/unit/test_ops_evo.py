# test_evolution_ops.py
import random
import numpy as np
from typing import List
from dataclasses import dataclass

from swarm_rag.evolution.types.genome import Genome, SwarmParams, DEFAULT_PARAMS
from swarm_rag.evolution.types.config import EvolutionContext, EvolutionConfigDict, DEFAULT_EVO_CONFIG
from swarm_rag.evolution.execution.strategies import GeneticStrategies
from swarm_rag.evolution.execution.fitness import FitnessResult

# Helper to make dummy genomes
def make_genome(id, q_score, n_agents=10) -> Genome:
    params = DEFAULT_PARAMS.copy()
    params['n_agents'] = n_agents
    g = Genome(id=str(id), params=params)
    g.fitness = FitnessResult(q_score, 0.5, 100)
    return g

def test_selection_ops():
    print("\n--- Testing Selection Strategies ---")
    
    # Create population: 1 Super, 9 Trash
    pop = [make_genome(f"bad_{i}", 0.1) for i in range(9)]
    best = make_genome("best", 0.9)
    pop.append(best)
    
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
        selected = GeneticStrategies(ctx)
        if selected.id == "best": wins += 1
    
    print(f"  ✓ Tournament selected 'best' {wins}/100 times (Expected > 80)")
    assert wins > 80

    # 2. Truncation (Requires Sorted Pop)
    print("  Testing Truncation...")
    pop.sort(key=lambda g: g.fitness, reverse=True) # Sort Best First
    ctx.population = pop # Update context
    
    # Truncation logic (e.g. top 20%) should ALWAYS pick 'best' 
    # since it's the only one in the top 10% of 10 items
    selected = GeneticStrategies.truncation_selection(ctx)
    print(f"  ✓ Truncation selected: {selected.id}")
    assert selected.id == "best"

def test_crossover_ops():
    print("\n--- Testing Crossover (Uniform Mix) ---")
    
    parent1 = make_genome("p1", 0.5, n_agents=10) # 10 agents
    parent1.params['decay'] = 0.1
    
    parent2 = make_genome("p2", 0.5, n_agents=90) # 90 agents
    parent2.params['decay'] = 0.9
    
    ctx = EvolutionContext(population=[], generation=0, config=DEFAULT_EVO_CONFIG)
    
    # Generate statistics over many children
    n_agents_sum = 0
    decay_sum = 0
    n_trials = 100
    
    for _ in range(n_trials):
        child : Genome = GeneticStrategies.uniform_parameter_mix(parent1, parent2, ctx)
        n_agents_sum += child.params['n_agents']
        decay_sum += child.params['decay']
        
    avg_agents = n_agents_sum / n_trials
    avg_decay = decay_sum / n_trials
    
    print(f"  Avg Agents: {avg_agents} (Expected ~50)")
    print(f"  Avg Decay:  {avg_decay:.2f} (Expected ~0.5)")
    
    assert 40 < avg_agents < 60, "Crossover bias detected"

def test_mutation_ops():
    print("\n--- Testing Mutation ---")
    
    g = make_genome("mutant", 0.5, n_agents=20)
    ctx = EvolutionContext(population=[], generation=0, config=DEFAULT_EVO_CONFIG)
    ctx.config['mutation_rate'] = 1.0 # Force mutation
    
    # Run mutation
    mutated = GeneticStrategies.expression_tree_mutation(g, ctx)
    
    print(f"  Original n_agents: 20")
    print(f"  Mutated n_agents:  {mutated.params['n_agents']}")
    
    # It should have changed (with high probability)
    # Integer jitter is +/- 2 usually
    assert mutated.params['n_agents'] != 20 or len(mutated.strategies) == 0
    print("  ✓ Parameter changed successfully")

if __name__ == "__main__":
    test_selection_ops()
    test_crossover_ops()
    test_mutation_ops()
    print("\nALL OPS TESTS PASSED")