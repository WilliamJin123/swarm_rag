import pytest
from typing import List
from swarm_rag.evolution.types.genome import Genome, DEFAULT_PARAMS
from swarm_rag.evolution.types.config import EvolutionContext, DEFAULT_EVO_CONFIG
from swarm_rag.evolution.extensions.immigration import RandomImmigrationExtension

# --- MOCK FACTORY ---
# Replacing FakeEngine with a factory-style callable
class MockGenomeFactory:
    def create_population(self, count: int) -> List[Genome]:
        # Return unique "Immigrant" genomes
        return [
            Genome(id=f"immigrant_{i}", params=DEFAULT_PARAMS.copy(), strategies={}) 
            for i in range(count)
        ]

def test_immigration_replaces_worst_performers():
    # 1. Setup population with 10 agents
    pop = []
    for i in range(10):
        g = Genome(id=f"native_{i}", params=DEFAULT_PARAMS.copy(), strategies={})
        # Assign fitness: Agent 0 is BEST (10.0), Agent 9 is WORST (1.0)
        g.fitness.quality_score = 10.0 - i 
        pop.append(g)

    ctx = EvolutionContext(
        config=DEFAULT_EVO_CONFIG,
        generation=0,
        available_features=[],
        expression_features={},
        population=pop
    )

    # 2. Init Extension using the new genome_factory parameter
    factory = MockGenomeFactory()
    # Rate 0.2 means replace bottom 20% (2 agents)
    # We pass the factory METHOD directly
    ext = RandomImmigrationExtension(
        rate=0.2, 
        genome_factory=factory.create_population
    )
    
    # 3. Run Hook
    ext.on_before_breeding(ctx)
    
    # 4. Verify Effects
    
    # The population size should stay constant
    assert len(ctx.population) == 10
    
    # The population is sorted Best->Worst (10.0 -> 1.0)
    # Index 0 (Best) should still be "native_0"
    assert ctx.population[0].id == "native_0"
    
    # The bottom 2 (indices -1 and -2) should now be immigrants
    assert "immigrant" in ctx.population[-1].id
    assert "immigrant" in ctx.population[-2].id
    
    # The "native_9" (worst) and "native_8" (second worst) should be GONE
    ids = [g.id for g in ctx.population]
    assert "native_9" not in ids
    assert "native_8" not in ids
    
    print("\nFinal Population IDs:", ids)

if __name__ == "__main__":
    test_immigration_replaces_worst_performers()