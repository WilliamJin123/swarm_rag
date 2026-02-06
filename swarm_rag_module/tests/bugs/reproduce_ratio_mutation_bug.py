import random
import unittest
from unittest.mock import MagicMock

from swarm_rag.evolution.types.genome import Genome
from swarm_rag.evolution.execution.genetics.strategies import GeneticStrategies
from swarm_rag.evolution.types.config import EvolutionContext
from swarm_rag.evolution.types.expressions import ExpressionNode

class TestRatioMutationBug(unittest.TestCase):
    def setUp(self):
        self.ctx = MagicMock(spec=EvolutionContext)
        self.ctx.global_mutation_multiplier = 100.0 # Force mutation
        self.ctx.config = {
            'swarmrag_param_ranges': {},
            'base_mutation_rate': 0.5
        }
        self.ctx.expression_features = {}

    def test_aggressive_mutation_ignores_ratios(self):
        # Create a genome with specific ratios
        g = Genome(id="test", mutation_rate=0.5)
        g.group_ratios = {"g0": 0.5, "g1": 0.5}
        g.params = {"n_agents": 10} # dummy
        
        # Snapshot before
        initial_ratios = g.group_ratios.copy()
        
        # Apply Aggressive Mutation
        # We need to mock random to ensure we hit the mutation branch if it existed
        # But since the code is missing, it won't matter what random does, it won't touch it.
        # However, to be scientifically rigorous, we should set random to allow mutation.
        
        # The bug is that the code loop is MISSING. 
        # So even with high mutation rate, it won't change.
        
        mutated = GeneticStrategies.aggressive_mutation(g, self.ctx)
        
        # We expect ratios to change now
        self.assertNotEqual(mutated.group_ratios, initial_ratios, "Aggressive Mutation FAILED to mutate ratios")
        
        print("\n[Verified] Aggressive Mutation successfully mutated ratios.")

    def test_guided_mutation_ignores_ratios(self):
        # Create a genome with specific ratios
        g = Genome(id="test", mutation_rate=0.5)
        g.group_ratios = {"g0": 0.5, "g1": 0.5}
        g.params = {"n_agents": 10}
        
        initial_ratios = g.group_ratios.copy()
        
        mutated = GeneticStrategies.guided_mutation(g, self.ctx)
        
        self.assertNotEqual(mutated.group_ratios, initial_ratios, "Guided Mutation FAILED to mutate ratios")
        
        print("\n[Verified] Guided Mutation successfully mutated ratios.")

if __name__ == '__main__':
    unittest.main()