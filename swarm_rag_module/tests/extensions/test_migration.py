import os
import shutil
import pytest
import pickle
from swarm_rag.evolution.types.genome import Genome, DEFAULT_PARAMS
from swarm_rag.evolution.types.config import EvolutionContext, DEFAULT_EVO_CONFIG
from swarm_rag.evolution.extensions.migration import FileMigrationExtension

TEST_POOL_DIR = "./test_migration_pool"

# --- MOCK STRATEGY (Required for Genome.copy) ---
class MockStrategy:
    def __init__(self, val=1.0):
        self.val = val
    def copy(self):
        return MockStrategy(self.val)
    def evaluate(self, ctx):
        return self.val

@pytest.fixture
def clean_pool():
    if os.path.exists(TEST_POOL_DIR): shutil.rmtree(TEST_POOL_DIR)
    os.makedirs(TEST_POOL_DIR)
    yield
    if os.path.exists(TEST_POOL_DIR): shutil.rmtree(TEST_POOL_DIR)

def create_island_context(island_name, best_score=100.0):
    pop = []
    for i in range(10):
        # Give them mock strategies so copy() doesn't crash
        strategies = {"movement": MockStrategy()}
        g = Genome(id=f"{island_name}_g{i}", params=DEFAULT_PARAMS.copy(), strategies=strategies)
        g.fitness.quality_score = float(best_score - i)
        pop.append(g)
        
    return EvolutionContext(
        config=DEFAULT_EVO_CONFIG,
        generation=5,
        available_features=[],
        expression_features={},
        population=pop
    )

def test_migration_exchange(clean_pool):
    # --- ISLAND A (High Performance: 100..91) ---
    ctx_a = create_island_context("IslandA", best_score=100.0)
    ext_a = FileMigrationExtension(migration_dir=TEST_POOL_DIR, interval=5, island_id="A")
    
    # --- ISLAND B (Low Performance: 50..41) ---
    ctx_b = create_island_context("IslandB", best_score=50.0)
    ext_b = FileMigrationExtension(migration_dir=TEST_POOL_DIR, interval=5, island_id="B")
    
    # 1. Island A exports (Best: 100.0)
    ext_a.on_generation_end(ctx_a)
    
    # Verify file content manually to isolate Export vs Import bugs
    files = os.listdir(TEST_POOL_DIR)
    assert len(files) == 1
    with open(os.path.join(TEST_POOL_DIR, files[0]), "rb") as f:
        saved_genomes = pickle.load(f)
        # CRITICAL CHECK: Did we save the score correctly?
        assert saved_genomes[0].fitness.quality_score == 100.0, \
            f"Export failed! Saved score is {saved_genomes[0].fitness.quality_score}, expected 100.0"

    # 2. Island B imports
    ext_b.on_generation_end(ctx_b)
    
    # --- VERIFY SWAP ---
    # Sort B to find the best
    ctx_b.population.sort(key=lambda g: g.fitness.quality_score, reverse=True)
    best_genome_b = ctx_b.population[0]
    
    print(f"\nIsland B Top 3 IDs: {[g.id for g in ctx_b.population[:3]]}")
    print(f"Island B Top 3 Scores: {[g.fitness.quality_score for g in ctx_b.population[:3]]}")
    
    assert best_genome_b.fitness.quality_score == 100.0
    assert "IslandA" in best_genome_b.id

if __name__ == "__main__":
    pytest.main([__file__, "-s"])