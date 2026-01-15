"""
Evolution Engine - Facade for evolutionary optimization.

This module provides backwards-compatible access to evolution functionality
while delegating to specialized orchestrators for different algorithms.
"""
import os
import random
import pickle
import numpy as np
from typing import List, Any
import logging

from ..interfaces.enums import GeneticKey

from ..core.swarm_retriever import SwarmRetriever
from ..core.heuristics import HeuristicRegistry
from ..eval.metrics import Evaluator

from .types.genome import Genome, DEFAULT_PARAMS
from .types.config import EvolutionContext, EvolutionConfigDict, DEFAULT_EVO_CONFIG
from .types.expressions import ExpressionEvolution

from .execution.evaluator import PopulationEvaluator
from .execution.loop import EvolutionLoop
from .execution.fitness import FitnessCalculator
from .execution.tracker import ProgressTracker
from .execution.factory import GenomeFactory
from .execution.fitness_strategies import (
    FitnessStrategy,
    LexicographicStrategy,
    ParetoStrategy,
    PhasedStrategy
)
from .extensions.base import EvolutionExtension

# MAP-Elites Imports
from .map_elites.archive import MapElitesArchive
from .map_elites.descriptors import DescriptorCalculator
from .map_elites.loop import MapElitesLoop

# Orchestrators
from .orchestrators.standard_ga import StandardGAOrchestrator
from .orchestrators.map_elites import MAPElitesOrchestrator


class EvolutionEngine:
    """
    Facade for evolutionary optimization.

    Provides backwards-compatible API while delegating to specialized orchestrators:
    - StandardGAOrchestrator: Traditional genetic algorithm
    - MAPElitesOrchestrator: Quality-Diversity optimization

    Example:
        engine = EvolutionEngine(
            retriever=retriever,
            fitness_calculator=fitness_calc,
            evaluator=evaluator,
            train_query_ids=train_ids,
            train_ground_truth=train_gt,
            val_query_ids=val_ids,
            val_ground_truth=val_gt,
            config=config
        )
        best_genome = engine.optimize()
    """

    def __init__(
        self,
        retriever: SwarmRetriever,
        fitness_calculator: FitnessCalculator,
        evaluator: Evaluator,
        train_query_ids: List[Any],
        train_ground_truth: List[List[Any]],
        val_query_ids: List[Any],
        val_ground_truth: List[List[Any]],
        config: EvolutionConfigDict = None,
        genome_factory: 'GenomeFactory' = None,
        extensions: List['EvolutionExtension'] = None,
        overwrite_logs: bool = True
    ):
        """
        Initialize the evolution engine.

        Args:
            retriever: SwarmRetriever instance for genome evaluation
            fitness_calculator: Calculator for fitness scores
            evaluator: Metric evaluator
            train_query_ids: Training query IDs
            train_ground_truth: Training ground truth
            val_query_ids: Validation query IDs
            val_ground_truth: Validation ground truth
            config: Evolution configuration (defaults to DEFAULT_EVO_CONFIG)
            genome_factory: Optional pre-configured genome factory
            extensions: List of evolution extensions (hooks)
            overwrite_logs: Whether to overwrite existing log files
        """
        logger = logging.getLogger(__name__)
        config = config or DEFAULT_EVO_CONFIG.copy()

        self.train_query_ids = train_query_ids
        self.train_gt = train_ground_truth
        self.val_query_ids = val_query_ids
        self.val_gt = val_ground_truth

        # Initialize context and factory
        if genome_factory is None:
            self.evo_context = EvolutionContext(
                config=config,
                generation=0,
                available_features=list(HeuristicRegistry.all().keys()),
                expression_features={
                    "movement": list(HeuristicRegistry.all_movement().keys()),
                    "ranking": list(HeuristicRegistry.all_ranking().keys()),
                    "deposit": list(HeuristicRegistry.all_deposit().keys()),
                }
            )
            self.genome_factory = GenomeFactory(self.evo_context)
        else:
            self.genome_factory = genome_factory
            self.evo_context = genome_factory.context

        self.config = self.evo_context.config

        # Initialize population evaluator
        self.population_evaluator = PopulationEvaluator(
            retriever=retriever,
            evaluator=evaluator,
            fitness_calc=fitness_calculator,
            queries=train_query_ids,
            ground_truth=train_ground_truth,
            concurrent_evaluations=self.config["concurrent_evaluations"],
            max_workers_per_retrieval=self.config["max_workers_per_retrieval"]
        )

        # Initialize evolution loop
        self.loop = EvolutionLoop(self.evo_context)

        # Initialize LLM Provider if using LLM Mutation
        if self.config.get("mutation_strategy") == GeneticKey.LLM_MUTATION:
            from .llm.factory import LLMProviderFactory
            llm_provider = LLMProviderFactory.create(self.config)
            if llm_provider is not None:
                self.evo_context.llm_provider = llm_provider
                logger.info(
                    f"ENABLED: LLM-Guided Mutation Strategy "
                    f"(provider={self.config.get('llm_provider', 'cerebras')}, "
                    f"model={self.config.get('llm_model', 'zai-glm-4.7')})"
                )

        # Initialize progress tracker
        self.tracker = ProgressTracker(
            log_path=self.config["log_path"],
            plot_path=self.config["plot_path"],
            plot_title=self.config["plot_title"],
            overwrite=overwrite_logs
        )

        # Initialize fitness strategy
        strategy_name = self.config.get("fitness_strategy", "lexicographic")
        if strategy_name == "pareto":
            self.fitness_strategy = ParetoStrategy()
        elif strategy_name == "phased":
            switch_gen = self.config.get("phased_switch_gen", 10)
            self.fitness_strategy = PhasedStrategy(switch_gen=switch_gen)
        else:
            self.fitness_strategy = LexicographicStrategy()

        logger.info(f"Using Fitness Strategy: {self.fitness_strategy.__class__.__name__}")

        # Initialize extensions
        self.extensions = extensions or []
        for ext in self.extensions:
            if hasattr(ext, 'genome_factory') and ext.genome_factory is None:
                ext.genome_factory = self.genome_factory.create_population
            ext.on_init(self.evo_context)

        # Initialize MAP-Elites components if enabled
        self.map_elites_archive = None
        self.map_elites_loop = None
        if self.config.get("map_elites_enabled", False):
            logger.info("ENABLED: MAP-Elites Mode")
            descriptor_calc = DescriptorCalculator(
                dimensions=self.config["map_elites_dims"],
                ranges=self.config["map_elites_ranges"]
            )
            self.map_elites_archive = MapElitesArchive(
                descriptor_calc=descriptor_calc,
                bins=self.config["map_elites_bins"],
                ranges=self.config["map_elites_ranges"]
            )
            self.map_elites_loop = MapElitesLoop(self.evo_context)

        # Create appropriate orchestrator
        self._orchestrator = self._create_orchestrator()

        # For checkpoint restoration
        self.restored_best_genome = None

    def _create_orchestrator(self):
        """Create the appropriate orchestrator based on configuration."""
        if self.config.get("map_elites_enabled", False):
            return MAPElitesOrchestrator(
                context=self.evo_context,
                evaluator=self.population_evaluator,
                fitness_strategy=self.fitness_strategy,
                tracker=self.tracker,
                val_query_ids=self.val_query_ids,
                val_ground_truth=self.val_gt,
                archive=self.map_elites_archive,
                me_loop=self.map_elites_loop,
                genome_factory=self.genome_factory,
                extensions=self.extensions
            )
        else:
            return StandardGAOrchestrator(
                context=self.evo_context,
                evaluator=self.population_evaluator,
                fitness_strategy=self.fitness_strategy,
                tracker=self.tracker,
                val_query_ids=self.val_query_ids,
                val_ground_truth=self.val_gt,
                loop=self.loop,
                genome_factory=self.genome_factory,
                extensions=self.extensions
            )

    def optimize(self, initial_population: List[Genome] = None) -> Genome:
        """
        Run evolutionary optimization.

        Delegates to the appropriate orchestrator based on configuration.

        Args:
            initial_population: Optional starting population

        Returns:
            Best genome found during evolution
        """
        # Transfer any restored state to orchestrator
        if self.restored_best_genome:
            self._orchestrator.restored_best_genome = self.restored_best_genome

        return self._orchestrator.optimize(initial_population)

    def save_checkpoint(self, population: List[Genome], best_genome: Genome, generation: int):
        """
        Saves evolution state to disk.

        Delegates to orchestrator for consistent checkpoint format.
        """
        self._orchestrator.save_checkpoint(population, best_genome, generation)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        retriever: Any,
        fitness_calculator: Any,
        evaluator: Any,
        train_query_ids: List[Any],
        train_ground_truth: List[List[Any]],
        val_query_ids: List[Any],
        val_ground_truth: List[List[Any]],
        config: EvolutionConfigDict,
        extensions: List['EvolutionExtension'] = None,
    ) -> 'EvolutionEngine':
        """
        Factory method: Creates a NEW engine instance and restores state from disk.

        Args:
            checkpoint_path: Path to checkpoint file
            retriever: SwarmRetriever instance
            fitness_calculator: FitnessCalculator instance
            evaluator: Evaluator instance
            train_query_ids: Training query IDs
            train_ground_truth: Training ground truth
            val_query_ids: Validation query IDs
            val_ground_truth: Validation ground truth
            config: Evolution configuration
            extensions: Optional evolution extensions

        Returns:
            Initialized EvolutionEngine with restored state
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        print(f"--> Loading checkpoint from: {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)

        # Create fresh engine
        engine = cls(
            retriever=retriever,
            fitness_calculator=fitness_calculator,
            evaluator=evaluator,
            train_query_ids=train_query_ids,
            train_ground_truth=train_ground_truth,
            val_query_ids=val_query_ids,
            val_ground_truth=val_ground_truth,
            config=config,
            extensions=extensions,
            overwrite_logs=False
        )

        # Restore evolutionary state
        engine.evo_context.population = state['population']
        engine.evo_context.generation = state['generation']

        # Restore RNG states for reproducibility
        if 'random_state' in state:
            random.setstate(state['random_state'])
        if 'np_random_state' in state:
            np.random.set_state(state['np_random_state'])

        # Restore tracker history
        if 'tracker_history' in state:
            engine.tracker.history = state['tracker_history']

        # Restore best genome
        engine.restored_best_genome = state.get('best_genome')

        print(f"  State restored. Resuming from Generation {state['generation']}")
        return engine
