import pytest
import numpy as np
from typing import List

from swarm_rag.evolution.types.genome import Genome, DEFAULT_PARAMS
from swarm_rag.evolution.types.config import EvolutionContext, DEFAULT_EVO_CONFIG
from swarm_rag.evolution.extensions.niching import NichingExtension

# Helper to create a functional context
def create_test_context(pop_size=5) -> EvolutionContext:
    pop = []
    for i in range(pop_size):
        g = Genome(id=f"g{i}", params=DEFAULT_PARAMS.copy(), strategies={})
        g.fitness.quality_score = 100.0  # Start with high fitness
        pop.append(g)
    
    return EvolutionContext(
        config=DEFAULT_EVO_CONFIG,
        generation=0,
        available_features=[],
        expression_features={"movement": [], "ranking": [], "deposit": []},
        population=pop
    )

def test_niching_penalizes_identical_behaviors():
    """
    Scenario: 3 agents behave identically (Clones). 2 agents are unique.
    Result: The 3 clones should share fitness (score drops). Unique ones stay high.
    """
    ctx = create_test_context(pop_size=5)
    
    # --- SETUP BEHAVIORS ---
    # Agents 0, 1, 2: Identical Behavior (Always return 1.0)
    # Agents 3, 4: Distinct Behavior (Return 50.0 and 90.0)
    
    # We inject directly into _compiled_cache to simulate "executable strategies"
    # This avoids needing to build complex Expression Trees just for the test.
    for i in range(3):
        ctx.population[i]._compiled_cache['movement'] = lambda x: 1.0
        
    ctx.population[3]._compiled_cache['movement'] = lambda x: 50.0
    ctx.population[4]._compiled_cache['movement'] = lambda x: 90.0

    # --- EXECUTE ---
    # Sigma 2.0 is standard. Alpha 1.0 is linear penalty.
    ext = NichingExtension(sigma_share=2.0, alpha=1.0, n_probes=10)
    ext.on_after_evaluation(ctx)
    
    # --- VERIFY ---
    
    # 1. The Clones (0, 1, 2)
    # They are distance 0.0 from each other.
    # They should effectively split the fitness of "1" solution.
    # Expected penalty factor roughly /3 (since there are 3 of them).
    # Starting score 100.0 -> Should be < 50.0
    print("\nClone Fitnesses:", [g.fitness.quality_score for g in ctx.population[:3]])
    for i in range(3):
        assert ctx.population[i].fitness.quality_score < 60.0, f"Clone {i} was not penalized enough"

    # 2. The Unique Agents (3, 4)
    # They are far from the clones (1.0 vs 50.0) and far from each other (50.0 vs 90.0).
    # They should retain most of their fitness.
    print("Unique Fitnesses:", [g.fitness.quality_score for g in ctx.population[3:]])
    assert ctx.population[3].fitness.quality_score > 90.0, "Unique agent 3 was penalized too much"
    assert ctx.population[4].fitness.quality_score > 90.0, "Unique agent 4 was penalized too much"

def test_niching_handles_missing_strategies():
    """Ensures it doesn't crash if genomes have no strategies yet."""
    ctx = create_test_context(pop_size=3)
    # No strategies set at all
    
    ext = NichingExtension()
    ext.on_after_evaluation(ctx)
    
    # Should run without error, treating them all as "Zero Vector" behaviors (clones)
    # So they should all be penalized equally
    scores = [g.fitness.quality_score for g in ctx.population]
    assert all(s < 100.0 for s in scores), "Empty genomes should be treated as clones and penalized"