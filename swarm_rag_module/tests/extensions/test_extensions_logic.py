import os
import shutil
import pytest
import numpy as np
from typing import List
import pickle

from swarm_rag.evolution.types.genome import Genome, DEFAULT_PARAMS
from swarm_rag.evolution.types.config import EvolutionContext, DEFAULT_EVO_CONFIG
from swarm_rag.evolution.extensions.niching import NichingExtension
from swarm_rag.evolution.extensions.immigration import RandomImmigrationExtension
from swarm_rag.evolution.extensions.migration import FileMigrationExtension

class MockStrategy:
    """A dummy object that acts like a Strategy Tree for testing."""
    def __init__(self, val=1.0):
        self.val = val
        
    def copy(self):
        """Genome.copy() needs this method."""
        return MockStrategy(self.val)

    def evaluate(self, context):
        """Fallback evaluation if compiled cache is missed."""
        return self.val

# --- MOCK CONTEXT HELPER ---
def create_mock_context(pop_size=10) -> EvolutionContext:
    pop = []
    for i in range(pop_size):
        # FIX: We give the genome "Dummy" strategies. 
        # The Niching extension will verify these exist, avoiding the "missing" log.
        # We don't need real Trees, because we will inject the Compiled Function directly.
        dummy_strategies = {
            "movement": MockStrategy(),
            "ranking": MockStrategy(),
            "deposit": MockStrategy()
        }
        
        g = Genome(id=f"g{i}", params=DEFAULT_PARAMS.copy(), strategies=dummy_strategies)
        g.fitness.quality_score = 1.0 
        
        # FIX: Inject Compiled Functions so Niching has something to execute
        # This simulates a genome that has successfully compiled its strategies
        g._compiled_cache['movement'] = lambda x: 1.0
        g._compiled_cache['ranking'] = lambda x: 1.0
        g._compiled_cache['deposit'] = lambda x: 1.0
        
        pop.append(g)
    
    ctx = EvolutionContext(
        config=DEFAULT_EVO_CONFIG,
        generation=0,
        available_features=[],
        expression_features={"movement": [], "ranking": [], "deposit": []},
        population=pop
    )
    return ctx

# --- 1. TEST NICHING ---
def test_niching_penalizes_clones():
    """
    If we have 5 identical clones, their fitness should be heavily penalized.
    """
    ctx = create_mock_context(pop_size=5)
    
    # 5 Clones (Empty strategies = identical behavior vector of [0.0...])
    # Everyone has quality_score = 1.0
    
    ext = NichingExtension(sigma_share=2.0, alpha=1.0)
    ext.on_after_evaluation(ctx)
    
    # Expected: The niche count is high (everyone is neighbor). 
    # Fitness should drop significantly below 1.0
    for g in ctx.population:
        assert g.fitness.quality_score < 0.5, "Niching failed to penalize clones"

def test_niching_spares_unique():
    """
    If agents are diverse, they shouldn't punish each other much.
    """
    ctx = create_mock_context(pop_size=2)
    # Mock behavior cache to force them to be different
    # Agent 0 returns [0,0...], Agent 1 returns [10,10...]
    # We cheat by injecting directly into the profiler logic or just mocking the tree
    # But since our Niching uses 'get_compiled_strategy', let's mock the _compiled_cache
    
    # Mock functions that return different constant values
    ctx.population[0]._compiled_cache['movement'] = lambda x: 0.0
    ctx.population[1]._compiled_cache['movement'] = lambda x: 100.0 # Far away
    
    ext = NichingExtension(sigma_share=2.0)
    ext.on_after_evaluation(ctx)
    
    # Distance is huge, so penalty should be ~0 (Fitness stays ~1.0)
    assert ctx.population[0].fitness.quality_score > 0.95

# --- 2. TEST RANDOM IMMIGRATION ---
class MockEngine:
    def create_initial_genomes(self):
        # Return a fresh list of "Immigrants"
        return [Genome(id="immigrant", params=DEFAULT_PARAMS, strategies={}) for _ in range(10)]

def test_immigration_replaces_worst():
    ctx = create_mock_context(pop_size=10)
    
    # Set fitnesses manually: 0..8 are Good, 9 is Bad
    for i, g in enumerate(ctx.population):
        g.fitness.quality_score = float(i) # 0.0 to 9.0
    
    # Sort so we know who is worst (index 0 has score 0.0)
    ctx.population.sort(key=lambda g: g.fitness.quality_score, reverse=True)
    # Now: Index 0=9.0 (Best), Index 9=0.0 (Worst)
    
    engine = MockEngine()
    # Replace bottom 20% (2 agents)
    ext = RandomImmigrationExtension(rate=0.2, engine_ref=engine)
    
    # Verify pre-condition
    assert ctx.population[-1].id != "immigrant"
    
    ext.on_before_breeding(ctx)
    
    # Verify post-condition: Bottom 2 should be immigrants
    assert ctx.population[-1].id == "immigrant"
    assert ctx.population[-2].id == "immigrant"
    # Top guy should still be there
    assert ctx.population[0].id != "immigrant"

# --- 3. TEST FILE MIGRATION ---
def test_migration_io():
    test_dir = "./test_migration_pool"
    if os.path.exists(test_dir): shutil.rmtree(test_dir)
    
    ctx = create_mock_context(pop_size=10)
    ctx.generation = 5 # Trigger interval
    
    # Give them distinct scores
    for i, g in enumerate(ctx.population):
        g.fitness.quality_score = float(i)
        
    ext = FileMigrationExtension(migration_dir=test_dir, interval=5, island_id="test_island")
    
    # 1. Test Export
    ext.on_generation_end(ctx)
    
    files = os.listdir(test_dir)
    assert len(files) == 1
    assert "island_test_island" in files[0]
    
    # 2. Test Import (Simulate a neighbor file)
    neighbor_file = os.path.join(test_dir, "gen_5_island_neighbor.pkl")
    # Save a "Super Genome" in the neighbor file
    super_genome = Genome(id="super_alien", params=DEFAULT_PARAMS, strategies={})
    super_genome.fitness.quality_score = 9999.0
    
    with open(neighbor_file, "wb") as f:
        pickle.dump([super_genome], f)
    
    # Run hook again -> Should find neighbor file
    ext.on_generation_end(ctx)
    
    # The Super Genome should have replaced our worst agent
    # Since we sort Best->Worst, the worst is at the end
    assert ctx.population[-1].id == "super_alien"
    assert ctx.population[-1].fitness.quality_score == 9999.0

    # Cleanup
    if os.path.exists(test_dir): shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_niching_penalizes_clones()
    test_niching_spares_unique()
    test_immigration_replaces_worst()
    test_migration_io()

    print("TESTS SUCCEEDED")