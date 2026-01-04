import os
import shutil
import pytest
import pickle
from swarm_rag.evolution.types.genome import Genome, DEFAULT_PARAMS
from swarm_rag.evolution.types.config import EvolutionContext, DEFAULT_EVO_CONFIG
from swarm_rag.evolution.extensions.migration import FileMigrationExtension

TEST_POOL_DIR = "./test_migration_pool"

@pytest.fixture
def clean_pool():
    if os.path.exists(TEST_POOL_DIR): shutil.rmtree(TEST_POOL_DIR)
    os.makedirs(TEST_POOL_DIR)
    yield
    if os.path.exists(TEST_POOL_DIR): shutil.rmtree(TEST_POOL_DIR)

def create_island_context(island_name, best_score=100.0):
    """Creates a population where the best agent has a specific score."""
    pop = []
    for i in range(10):
        g = Genome(id=f"{island_name}_g{i}", params=DEFAULT_PARAMS, strategies={})
        # Score decreases
        g.fitness.quality_score = best_score - i
        pop.append(g)
        
    return EvolutionContext(
        config=DEFAULT_EVO_CONFIG,
        generation=5, # Matches interval
        available_features=[],
        expression_features={},
        population=pop
    )

def test_migration_exchange(clean_pool):
    # --- ISLAND A (High Performance) ---
    ctx_a = create_island_context("IslandA", best_score=100.0)
    ext_a = FileMigrationExtension(migration_dir=TEST_POOL_DIR, interval=5, island_id="A")
    
    # --- ISLAND B (Low Performance) ---
    ctx_b = create_island_context("IslandB", best_score=50.0)
    ext_b = FileMigrationExtension(migration_dir=TEST_POOL_DIR, interval=5, island_id="B")
    
    # 1. Island A runs generation end -> Exports to file
    ext_a.on_generation_end(ctx_a)
    
    # Check file exists
    files = os.listdir(TEST_POOL_DIR)
    assert len(files) == 1
    assert "island_A" in files[0]
    
    # 2. Island B runs generation end -> Imports A's file
    ext_b.on_generation_end(ctx_b)
    
    # --- VERIFY SWAP ---
    # Island B should now contain Island A's top agent (Score 100.0)
    # Because Island B's best was only 50.0.
    
    # Find max score in B
    max_score_b = max(g.fitness.quality_score for g in ctx_b.population)
    best_id_b = max(ctx_b.population, key=lambda g: g.fitness.quality_score).id
    
    assert max_score_b == 100.0
    assert "IslandA" in best_id_b
    
    print(f"\nIsland B Best Agent ID: {best_id_b} (Score: {max_score_b})")
    print("Migration successful: Island A's elite migrated to Island B.")

if __name__ == "__main__":
    pytest.main([__file__, "-s"])