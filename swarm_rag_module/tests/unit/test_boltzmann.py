import random
import pytest
from swarm_rag.evolution.types.config import EvolutionContext, EvolutionConfig
from swarm_rag.evolution.types.fitness_results import FitnessResult
from swarm_rag.evolution.execution.genetics.strategies import GeneticStrategies

class MockGenome:
    def __init__(self, quality):
        self.fitness = FitnessResult(quality_score=quality)
        self.id = f"g_{quality}"

def test_boltzmann_selection_probability():
    random.seed(42)  # Ensure determinism
    print("\n--- Testing Boltzmann Selection Probability ---")

    # 1. Setup Context with proper EvolutionConfig dataclass
    config = EvolutionConfig()
    config.genetic.boltzmann.temperature = 1.0  # Moderate temp
    config.genetic.boltzmann.adaptive = False   # Disable adaptive for this test

    ctx = EvolutionContext(config=config)
    ctx.current_temperature = 1.0
    
    # 2. Create Population: Closer values so probabilities are more distributed
    # T=1.0
    # e^10, e^11, e^12 -> Ratios approx 1 : 2.7 : 7.4
    g1 = MockGenome(10.0)
    g2 = MockGenome(11.0) 
    g3 = MockGenome(12.0) 
    
    ctx.population = [g1, g2, g3]
    
    # 3. Select many times to check distribution
    selected = GeneticStrategies.boltzmann_selection(ctx, k=1000)
    
    counts = {g.id: 0 for g in ctx.population}
    for g in selected:
        counts[g.id] += 1
        
    print(f"Selection Counts (T=1.0): {counts}")
    
    # g3 > g2 > g1
    assert counts[g3.id] > counts[g2.id] > counts[g1.id]
    
    # g3 should be dominant but not 100%
    # Approx 7.4 / (1+2.7+7.4) = 7.4/11.1 = 66%
    assert counts[g3.id] > 500 
    assert counts[g1.id] > 0 

def test_boltzmann_adaptive_heating():
    print("\n--- Testing Adaptive Heating (Low Diversity) ---")

    # 1. Setup Context with proper EvolutionConfig dataclass
    config = EvolutionConfig()
    config.genetic.boltzmann.temperature = 1.0
    config.genetic.boltzmann.adaptive = True
    config.genetic.boltzmann.alpha = 0.9

    ctx = EvolutionContext(config=config)
    ctx.current_temperature = 1.0
    
    # 2. Population with ZERO diversity
    ctx.population = [MockGenome(10.0) for _ in range(5)]
    
    # 3. Perform selection (trigger update)
    GeneticStrategies.boltzmann_selection(ctx, k=1)
    
    print(f"Temp after heating: {ctx.current_temperature}")
    
    # Should have increased: T_new = T_old / alpha = 1.0 / 0.9 = 1.11
    assert ctx.current_temperature > 1.0
    assert abs(ctx.current_temperature - (1.0 / 0.9)) < 0.001

def test_boltzmann_adaptive_cooling():
    print("\n--- Testing Adaptive Cooling (High Diversity) ---")

    # 1. Setup Context with proper EvolutionConfig dataclass
    config = EvolutionConfig()
    config.genetic.boltzmann.temperature = 1.0
    config.genetic.boltzmann.adaptive = True
    config.genetic.boltzmann.alpha = 0.9

    ctx = EvolutionContext(config=config)
    ctx.current_temperature = 1.0
    
    # 2. Population with HIGH diversity
    ctx.population = [MockGenome(0.0), MockGenome(50.0), MockGenome(100.0)]
    
    # 3. Perform selection
    GeneticStrategies.boltzmann_selection(ctx, k=1)
    
    print(f"Temp after cooling: {ctx.current_temperature}")
    
    # Should have decreased: T_new = T_old * alpha = 0.9
    assert ctx.current_temperature < 1.0
    assert abs(ctx.current_temperature - 0.9) < 0.001

if __name__ == "__main__":
    test_boltzmann_selection_probability()
    test_boltzmann_adaptive_heating()
    test_boltzmann_adaptive_cooling()
    print("\nALL BOLTZMANN TESTS PASSED")
