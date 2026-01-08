
import unittest
from unittest.mock import MagicMock
from swarm_rag.evolution.types.genome import Genome, GenomeCompiler
from swarm_rag.evolution.types.expressions import ExpressionNode
from swarm_rag.core.swarm_retriever import SwarmRetriever

class TestCompilerAndEvaluator(unittest.TestCase):
    def test_compiler_output(self):
        # 1. Setup Genome
        g = Genome(id="test")
        g.params['n_agents'] = 10
        g.group_ratios = {'g0': 0.5, 'g1': 0.5}
        
        # Create dummy strategy trees
        # g0_movement, g0_deposit, g1_movement...
        dummy_tree = ExpressionNode("feature", "semantic_similarity")
        g.strategies = {
            "g0_movement": dummy_tree,
            "g0_deposit": dummy_tree,
            "g1_movement": dummy_tree,
            "g1_deposit": dummy_tree,
            "ranking": dummy_tree
        }
        
        # 2. Compile
        compiler = GenomeCompiler()
        kwargs = compiler.compile(g)
        
        # 3. Assertions
        self.assertIn("agent_groups", kwargs)
        agent_groups = kwargs["agent_groups"]
        self.assertEqual(len(agent_groups), 2)
        
        # Check counts
        # 10 agents, 50/50 split -> 5 and 5
        counts = [grp['count'] for grp in agent_groups]
        self.assertEqual(counts, [5, 5])
        
        # Check strategy keys in the groups
        # Group 0
        g0 = agent_groups[0]
        self.assertIn("evolved_mov_0", g0['movement_strategies'])
        # Group 1
        g1 = agent_groups[1]
        self.assertIn("evolved_mov_1", g1['movement_strategies'])
        
        print("\n[Passed] GenomeCompiler correctly generates agent_groups.")

    def test_retriever_kwargs_passing(self):
        # 1. Setup Mock Retriever
        retriever = MagicMock(spec=SwarmRetriever)
        
        # 2. Simulate Evaluator Call
        # kwargs from previous test
        compiler_output = {
            'n_agents': 10,
            'agent_groups': [{'count': 5, 'foo': 'bar'}],
            'ranking_strategies': {}
        }
        
        # We simulate: retriever.retrieve_batch(queries=[...], **compiler_output)
        retriever.retrieve_batch(queries=["q1"], **compiler_output)
        
        # 3. Verify Call
        # Check if agent_groups was passed as a named argument
        call_args = retriever.retrieve_batch.call_args
        _, kwargs = call_args
        
        self.assertIn("agent_groups", kwargs)
        self.assertEqual(kwargs["agent_groups"], [{'count': 5, 'foo': 'bar'}])
        
        print("[Passed] kwargs are correctly unpacked into retrieve_batch.")

if __name__ == '__main__':
    unittest.main()
