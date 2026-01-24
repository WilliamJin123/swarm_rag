import unittest
import os
import shutil
from unittest.mock import MagicMock, patch
from swarm_rag.evolution.engine import EvolutionEngine
from swarm_rag.evolution.execution.fitness_strategies import (
    LexicographicStrategy,
    ParetoStrategy,
    PhasedStrategy
)
from swarm_rag.evolution.types.config import EvolutionConfig

class TestEngineFitnessStrategy(unittest.TestCase):
    def setUp(self):
        # Create temp dir for logs
        self.test_dir = "test_evo_run"
        os.makedirs(self.test_dir, exist_ok=True)

        # Mock dependencies
        self.mock_retriever = MagicMock()
        self.mock_fitness_calc = MagicMock()
        self.mock_evaluator = MagicMock()
        self.train_q = []
        self.train_gt = []
        self.val_q = []
        self.val_gt = []

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _get_config(self, **overrides):
        from dataclasses import replace
        config = EvolutionConfig()
        config.checkpoint.log_path = os.path.join(self.test_dir, "test.jsonl")
        config.checkpoint.plot_path = os.path.join(self.test_dir, "test.png")
        config.checkpoint.checkpoint_path = os.path.join(self.test_dir, "ckpt.pkl")
        if 'fitness_strategy' in overrides:
            config.fitness_strategy = overrides['fitness_strategy']
        if 'phased_switch_gen' in overrides:
            config.phased_switch_gen = overrides['phased_switch_gen']
        return config

    def test_default_strategy(self):
        """Test that default config loads LexicographicStrategy"""
        config = self._get_config(fitness_strategy="lexicographic")
            
        engine = EvolutionEngine(
            retriever=self.mock_retriever,
            fitness_calculator=self.mock_fitness_calc,
            evaluator=self.mock_evaluator,
            train_query_ids=self.train_q,
            train_ground_truth=self.train_gt,
            val_query_ids=self.val_q,
            val_ground_truth=self.val_gt,
            config=config,
            overwrite_logs=True
        )
        
        self.assertIsInstance(engine.fitness_strategy, LexicographicStrategy)
        print("✓ Default strategy is Lexicographic")

    def test_pareto_strategy_config(self):
        """Test that 'pareto' config loads ParetoStrategy"""
        config = self._get_config(fitness_strategy="pareto")
        
        engine = EvolutionEngine(
            retriever=self.mock_retriever,
            fitness_calculator=self.mock_fitness_calc,
            evaluator=self.mock_evaluator,
            train_query_ids=self.train_q,
            train_ground_truth=self.train_gt,
            val_query_ids=self.val_q,
            val_ground_truth=self.val_gt,
            config=config,
            overwrite_logs=True
        )
        
        self.assertIsInstance(engine.fitness_strategy, ParetoStrategy)
        print("✓ Config 'pareto' loads ParetoStrategy")

    def test_phased_strategy_config(self):
        """Test that 'phased' config loads PhasedStrategy with correct switch gen"""
        config = self._get_config(fitness_strategy="phased", phased_switch_gen=42)
        
        engine = EvolutionEngine(
            retriever=self.mock_retriever,
            fitness_calculator=self.mock_fitness_calc,
            evaluator=self.mock_evaluator,
            train_query_ids=self.train_q,
            train_ground_truth=self.train_gt,
            val_query_ids=self.val_q,
            val_ground_truth=self.val_gt,
            config=config,
            overwrite_logs=True
        )
        
        self.assertIsInstance(engine.fitness_strategy, PhasedStrategy)
        self.assertEqual(engine.fitness_strategy.switch_gen, 42)
        print("✓ Config 'phased' loads PhasedStrategy with correct generation")

if __name__ == "__main__":
    unittest.main()
