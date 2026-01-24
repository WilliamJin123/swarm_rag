"""
Base orchestrator class with shared logic for evolution algorithms.
"""
import os
import random
import pickle
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Any

import numpy as np

from ..types.genome import Genome
from ..types.config import EvolutionConfig, EvolutionContext
from ..execution.evaluator import PopulationEvaluator
from ..execution.tracker import ProgressTracker
from ..execution.fitness_strategies import FitnessStrategy
from ...utils import TqdmLoggingHandler


class BaseOrchestrator(ABC):
    """
    Abstract base class for evolution orchestrators.

    Handles common concerns:
    - Logging setup/cleanup
    - Checkpointing (save/load)
    - Validation runner

    Subclasses implement the specific optimization loop.
    """

    def __init__(
        self,
        context: EvolutionContext,
        evaluator: PopulationEvaluator,
        fitness_strategy: FitnessStrategy,
        tracker: ProgressTracker,
        val_query_ids: List[Any],
        val_ground_truth: List[List[Any]],
    ):
        """
        Initialize base orchestrator.

        Args:
            context: Evolution context with config and state
            evaluator: Population evaluator for fitness calculation
            fitness_strategy: Strategy for assigning fitness sort keys
            tracker: Progress tracker for logging
            val_query_ids: Validation query IDs
            val_ground_truth: Validation ground truth
        """
        self.context = context
        self.evo_config = context.config
        self.evaluator = evaluator
        self.fitness_strategy = fitness_strategy
        self.tracker = tracker
        self.val_query_ids = val_query_ids
        self.val_gt = val_ground_truth

        self.logger: Optional[logging.Logger] = None
        self.restored_best_genome: Optional[Genome] = None

    def setup_logging(self) -> logging.Logger:
        """
        Configures logging for the evolution run.

        Returns:
            Logger instance for evolution messages
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Clear existing handlers
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        # File handler for detailed logs
        log_path = self.evo_config.checkpoint.log_path
        # Fix replacement order and add timestamp for unique log files per run
        base_log = log_path.replace(".jsonl", "").replace(".json", "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"{base_log}_{timestamp}.log"
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        fh = logging.FileHandler(log_file_path)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(fh)

        # Tqdm-friendly console handler
        th = TqdmLoggingHandler()
        th.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(th)

        return logging.getLogger("evolution")

    def cleanup_logging(self):
        """Cleanup file handlers after evolution."""
        logger = logging.getLogger()
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)

    def run_validation(self, genome: Genome, generation: int) -> Optional[dict]:
        """
        Runs validation if conditions are met.

        Args:
            genome: Genome to validate
            generation: Current generation

        Returns:
            Validation stats dict if validation was run, else None
        """
        n_gen = self.evo_config.n_generations
        val_freq = self.evo_config.checkpoint.validation_frequency

        if (generation % val_freq == 0) or (generation == n_gen - 1):
            # Create a copy to not mess up training metrics
            val_candidate = genome.copy()
            val_candidate.evaluated = False

            # Evaluate on validation set
            self.evaluator.evaluate(
                [val_candidate], queries=self.val_query_ids, ground_truth=self.val_gt
            )

            return {
                "best_quality": val_candidate.fitness.quality_score,
                "recall": val_candidate.metrics.get("Recall@20", 0.0),
            }

        return None

    def save_checkpoint(
        self,
        population: List[Genome],
        best_genome: Genome,
        generation: int,
        extra_state: dict = None,
    ):
        """
        Saves evolution state to disk.

        Args:
            population: Current population
            best_genome: Best genome found so far
            generation: Current generation
            extra_state: Additional state to save (e.g., archive data)
        """
        state = {
            "generation": generation,
            "population": population,
            "best_genome": best_genome,
            "random_state": random.getstate(),
            "np_random_state": np.random.get_state(),
            "tracker_history": self.tracker.history,
            "orchestrator_type": self.__class__.__name__,
        }

        # Merge extra state if provided
        if extra_state:
            state.update(extra_state)

        ckpt_path = self.evo_config.checkpoint.checkpoint_path
        ckpt_dir = os.path.dirname(ckpt_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

        base, ext = os.path.splitext(ckpt_path)
        numbered_path = f"{base}_gen_{generation}{ext}"

        # Save numbered checkpoint
        with open(numbered_path, "wb") as f:
            pickle.dump(state, f)

        # Atomic update of "latest" checkpoint
        temp_latest = ckpt_path + ".tmp"
        with open(temp_latest, "wb") as f:
            pickle.dump(state, f)

        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        os.rename(temp_latest, ckpt_path)

        print(f"--> Checkpoint saved: {numbered_path}")

    def _find_best_in_population(self, population: List[Genome]) -> Optional[Genome]:
        """
        Finds the best genome in a population.

        Args:
            population: List of genomes to search

        Returns:
            Best genome or None if population is empty
        """
        if not population:
            return None

        evaluated = [g for g in population if g.fitness.quality_score > -float("inf")]
        if not evaluated:
            return None

        evaluated.sort(key=lambda g: g.fitness, reverse=True)
        return evaluated[0].copy()

    @abstractmethod
    def optimize(self, initial_population: List[Genome] = None) -> Genome:
        """
        Main optimization loop.

        Must be implemented by subclasses.

        Args:
            initial_population: Optional starting population

        Returns:
            Best genome found during evolution
        """
        pass
