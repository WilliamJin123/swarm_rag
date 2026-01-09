import unittest
from unittest.mock import MagicMock
import logging

from swarm_rag.evolution.llm.loop import LLMEvolutionLoop
from swarm_rag.evolution.llm.optimizer import MockLLMOptimizer
from swarm_rag.evolution.types.genome import Genome, DEFAULT_PARAMS
from swarm_rag.evolution.types.config import EvolutionContext
from swarm_rag.evolution.types.expressions import ExpressionNode

class TestLLMEvolutionLoop(unittest.TestCase):
    def setUp(self):
        # Setup basic context
        self.config = {
            "population_size": 10,
            "elite_fraction": 0.2,
            "selection_strategy": "tournament",
            "selection_k": 3,
            "llm_concurrency": 2,
            "n_generations": 5
        }
        self.context = EvolutionContext(
            config=self.config,
            generation=0,
            available_features=["a", "b"],
            expression_features={}
        )
        
        # Create a population
        self.population = []
        for i in range(10):
            g = Genome(id=f"gen_0_{i}")
            g.fitness.quality_score = 0.5 + (i * 0.01) # vary fitness
            g.params["n_agents"] = 20
            # Add a dummy strategy
            g.strategies["ranking"] = ExpressionNode("op", "+", [
                ExpressionNode("feature", "a"),
                ExpressionNode("const", 1.0)
            ])
            self.population.append(g)

    def test_loop_step(self):
        # Use Mock Optimizer
        optimizer = MockLLMOptimizer()
        
        # Override refine_genome to be deterministic for this test if needed,
        # but the default MockLLMOptimizer is fine (it randomly tweaks).
        
        loop = LLMEvolutionLoop(self.context, optimizer=optimizer)
        
        # Run one step
        next_gen = loop.step(self.population)
        
        # Assertions
        self.assertEqual(len(next_gen), 10, "Population size should remain constant")
        
        # Check Elitism
        # Top 2 (20% of 10) should be preserved exactly (same IDs)
        # Sort original by fitness desc
        sorted_pop = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        elites = sorted_pop[:2]
        
        # Next gen first 2 should be these elites
        self.assertEqual(next_gen[0].id, elites[0].id)
        self.assertEqual(next_gen[1].id, elites[1].id)
        
        # Check Refined Offspring
        # Remaining 8 should be new IDs (derived from parents)
        for i in range(2, 10):
            child = next_gen[i]
            self.assertTrue(child.id.startswith("gen_1_"), f"Child ID {child.id} should show next gen")
            self.assertNotEqual(child.id, elites[0].id)
            
            # The Mock optimizer randomly tweaks 'n_agents' or 'decay'.
            # It also sometimes adds/changes a strategy.
            # We can't strictly assert values because of randomness, 
            # but we can check if it ran without crashing.
            
    def test_apply_edits_parsing(self):
        """Test the specific logic of parsing strategy strings."""
        loop = LLMEvolutionLoop(self.context)
        
        g = Genome(id="test")
        edits = {
            "proposed_changes": {
                "params": {"n_agents": 99},
                "strategies": {
                    "ranking": "a * 2.0"
                }
            }
        }
        
        loop._apply_edits(g, edits)
        
        self.assertEqual(g.params["n_agents"], 99)
        self.assertTrue("ranking" in g.strategies)
        # Check if parsed correctly
        tree = g.strategies["ranking"]
        self.assertEqual(tree.type, "op")
        self.assertEqual(tree.value, "*")
        self.assertEqual(tree.children[0].value, "a")

if __name__ == '__main__':
    unittest.main()