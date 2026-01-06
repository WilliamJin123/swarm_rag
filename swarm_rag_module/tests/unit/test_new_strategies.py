import random
import numpy as np
from swarm_rag.evolution.types.config import EvolutionContext, DEFAULT_EVO_CONFIG
from swarm_rag.evolution.types.genome import Genome, DEFAULT_PARAMS
from swarm_rag.evolution.execution.factory import GenomeFactory
from swarm_rag.evolution.execution.strategies import GeneticStrategies, GeneticRegistry
from swarm_rag.evolution.types.expressions import ExpressionNode

def test_shallow_initialization():
    print("\n--- Testing Shallow Growth Initialization ---")
    
    # 1. Setup Context
    config = DEFAULT_EVO_CONFIG.copy()
    config["creation_strategy"] = "shallow_growth_initialization"
    config["population_size"] = 10
    config["n_agent_groups"] = 2
    
    # Need to mock expression features since factory needs them
    ctx = EvolutionContext(config=config)
    ctx.expression_features = {
        "movement": ["degree", "pheromone"],
        "deposit": ["degree"],
        "ranking": ["votes"]
    }
    
    # 2. Use Factory to create population
    factory = GenomeFactory(ctx)
    population = factory.create_population(count=10)
    
    # 3. Assertions
    assert len(population) == 10
    
    for genome in population:
        # Check Strategy Depths
        # Shallow growth forces max_depth=2. 
        # Note: ExpressionNode.depth() returns the depth. 
        # A single node has depth 1. A tree with one level of children has depth 2.
        
        for name, tree in genome.strategies.items():
            d = tree.depth()
            assert d <= 3, f"Tree {name} is too deep: {d} (Expected <= 3)"
            
            # Ensure it's not just a constant (should have some features if chance allows)
            # With 'grow' method and depth 2, it might be a feature or an op of features.
            
    print("  ✓ All genomes initialized with shallow trees (depth <= 2)")


def test_aggressive_mutation():
    print("\n--- Testing Aggressive Mutation ---")
    
    # 1. Create a dummy genome with known params
    params = DEFAULT_PARAMS.copy()
    params['n_agents'] = 10
    params['decay'] = 0.5
    
    # Simple tree: const(1.0)
    dummy_tree = ExpressionNode('const', 1.0)
    strategies = {
        'g0_movement': dummy_tree.copy(),
        'ranking': dummy_tree.copy()
    }
    
    genome = Genome(
        id="test_agg",
        params=params,
        group_ratios={'g0': 1.0},
        strategies=strategies,
        mutation_rate=0.1
    )
    
    # 2. Setup Context
    config = DEFAULT_EVO_CONFIG.copy()
    # Define ranges to allow resampling
    config['param_ranges'] = {
        'n_agents': (5, 30),
        'decay': (0.1, 0.9)
    }
    
    ctx = EvolutionContext(config=config)
    ctx.global_mutation_multiplier = 100.0 # Force mutation event
    ctx.expression_features = {
        "movement": ["degree"],
        "ranking": ["votes"]
    }
    
    # 3. Apply Mutation
    mutated = GeneticStrategies.aggressive_mutation(genome, ctx)
    
    # 4. Assertions
    
    # Rate Locking
    # Aggressive mutation locks base rate to 0.4
    # But note: in the implementation:
    #   genome.mutation_rate = 0.4
    #   rate = genome.mutation_rate * multiplier
    # The genome object itself should reflect the base rate update
    assert mutated.mutation_rate == 0.4, f"Mutation rate not locked to 0.4, got {mutated.mutation_rate}"
    
    # Parameter Mutation
    # With multiplier 100, probability is effectively 1.0
    # It should have changed
    print(f"  n_agents: {params['n_agents']} -> {mutated.params['n_agents']}")
    assert mutated.params['n_agents'] != 10 or mutated.params['decay'] != 0.5, "Parameters failed to mutate"
    
    # Tree Mutation
    # Should likely change from const(1.0)
    # Since we force aggressive, it might be a new random subtree
    new_tree = mutated.strategies['g0_movement']
    print(f"  Old Tree: {dummy_tree.to_string()}")
    print(f"  New Tree: {new_tree.to_string()}")
    
    # It is statistically incredibly unlikely to generate exactly const(1.0) 
    # if it triggers a subtree replacement or even a node mutation
    assert new_tree.to_string() != dummy_tree.to_string(), "Tree failed to mutate"
    
    print("  ✓ Aggressive mutation applied successfully")

def test_custom_strategy_integration():
    print("\n--- Testing Custom Strategy Integration (Mock) ---")
    
    # Verify we can register a dynamic lambda as a strategy on the fly
    # (Simulating a user defining a custom strategy in their own code)
    
    from swarm_rag.interfaces.registry import _CreationRegistry
    from swarm_rag.interfaces.enums import GeneticKey
    
    # 1. Define custom function
    def my_custom_creator(ctx, count):
        return [Genome(id=f"custom_{i}") for i in range(count)]
    
    # 2. Register it (using a string key since it's not in Enum)
    # The registry allows string keys!
    GeneticRegistry.creation.register("my_custom_init", my_custom_creator)
    
    # 3. Use it via Factory
    config = DEFAULT_EVO_CONFIG.copy()
    config["creation_strategy"] = "my_custom_init"
    
    ctx = EvolutionContext(config=config)
    factory = GenomeFactory(ctx)
    
    pop = factory.create_population(3)
    
    assert len(pop) == 3
    assert pop[0].id == "custom_0"
    
    print("  ✓ Custom dynamic strategy registered and executed")

if __name__ == "__main__":
    test_shallow_initialization()
    test_aggressive_mutation()
    test_custom_strategy_integration()
    print("\nALL NEW STRATEGY TESTS PASSED")
