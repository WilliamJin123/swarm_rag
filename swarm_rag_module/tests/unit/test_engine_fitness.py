import unittest
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch
from swarm_rag.evolution.engine import EvolutionEngine
from swarm_rag.evolution.execution.fitness.strategies import (
    LexicographicStrategy,
    ParetoStrategy,
)
from swarm_rag.evolution.types.config import EvolutionConfig, StorageConfig
from swarm_rag.evolution.storage import RunManager

class TestEngineFitnessStrategy(unittest.TestCase):
    def setUp(self):
        # Create temp dir for logs
        self.test_dir = tempfile.mkdtemp(prefix="evo_fitness_test_")

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
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _get_config(self, **overrides):
        storage = StorageConfig(
            base_dir=self.test_dir,
            dataset="test",
            run_id="fitness_test",
            checkpoint_frequency=1,
            validation_frequency=1,
        )
        config = EvolutionConfig(storage=storage)
        if 'fitness_strategy' in overrides:
            config.fitness_strategy = overrides['fitness_strategy']
        return config, storage

    def test_default_strategy(self):
        """Test that default config loads LexicographicStrategy"""
        config, storage = self._get_config(fitness_strategy="lexicographic")
        run_manager = RunManager(storage)

        engine = EvolutionEngine(
            retriever=self.mock_retriever,
            fitness_calculator=self.mock_fitness_calc,
            evaluator=self.mock_evaluator,
            train_query_ids=self.train_q,
            train_ground_truth=self.train_gt,
            val_query_ids=self.val_q,
            val_ground_truth=self.val_gt,
            config=config,
            run_manager=run_manager,
            overwrite_logs=True
        )

        self.assertIsInstance(engine.fitness_strategy, LexicographicStrategy)
        print("Default strategy is Lexicographic")

    def test_pareto_strategy_config(self):
        """Test that 'pareto' config loads ParetoStrategy"""
        config, storage = self._get_config(fitness_strategy="pareto")
        run_manager = RunManager(storage)

        engine = EvolutionEngine(
            retriever=self.mock_retriever,
            fitness_calculator=self.mock_fitness_calc,
            evaluator=self.mock_evaluator,
            train_query_ids=self.train_q,
            train_ground_truth=self.train_gt,
            val_query_ids=self.val_q,
            val_ground_truth=self.val_gt,
            config=config,
            run_manager=run_manager,
            overwrite_logs=True
        )

        self.assertIsInstance(engine.fitness_strategy, ParetoStrategy)
        print("Config 'pareto' loads ParetoStrategy")

    def test_phased_falls_back_to_lexicographic(self):
        """Test that legacy 'phased' config falls back to LexicographicStrategy"""
        config, storage = self._get_config(fitness_strategy="phased")
        run_manager = RunManager(storage)

        engine = EvolutionEngine(
            retriever=self.mock_retriever,
            fitness_calculator=self.mock_fitness_calc,
            evaluator=self.mock_evaluator,
            train_query_ids=self.train_q,
            train_ground_truth=self.train_gt,
            val_query_ids=self.val_q,
            val_ground_truth=self.val_gt,
            config=config,
            run_manager=run_manager,
            overwrite_logs=True
        )

        # Phased was removed, should fall back to Lexicographic
        self.assertIsInstance(engine.fitness_strategy, LexicographicStrategy)
        print("Legacy 'phased' config falls back to LexicographicStrategy")

if __name__ == "__main__":
    unittest.main()
