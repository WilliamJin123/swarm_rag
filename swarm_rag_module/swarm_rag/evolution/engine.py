"""
Evolution Engine - MAP-Elites based evolutionary optimization.

This module provides the main interface for running evolution using MAP-Elites
with optional LLM-guided mutations.
"""
import os
import random
import torch
from typing import List, Any, Optional
import logging

from ..interfaces.enums import GeneticKey

from ..core.swarm_retriever import SwarmRetriever
from ..core.heuristics import HeuristicRegistry
from ..eval.metrics import Evaluator

from .types.genome import Genome
from .types.config import (
    EvolutionConfig,
    EvolutionContext,
    DEFAULT_CONFIG,
    StorageConfig,
)
from .storage import RunManager

from .execution.evaluator import PopulationEvaluator
from .execution.fitness import FitnessCalculator
from .execution.tracker import ProgressTracker
from .execution.factory import GenomeFactory
from .execution.fitness_strategies import (
    LexicographicStrategy,
    ParetoStrategy,
)

# MAP-Elites Imports
from .map_elites.archive import (
    MapElitesArchive,
    ArchiveComparatorConfig,
    ArchiveComparisonMode,
)
from .map_elites.descriptors import DescriptorCalculator, DescriptorRegistry
from .map_elites.loop import MapElitesLoop

# Orchestrator
from .orchestrators.map_elites import MAPElitesOrchestrator

logger = logging.getLogger(__name__)


class EvolutionEngine:
    """
    MAP-Elites based evolutionary optimization engine.

    Uses quality-diversity optimization to evolve a diverse population of
    specialized retrieval strategies, with optional LLM-guided mutations.

    Example:
        config = EvolutionConfig(
            n_generations=100,
            map_elites=MapElitesConfig(bins=[20, 15]),
            llm=LLMConfig(enabled=True)
        )

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
        config: Optional[EvolutionConfig] = None,
        run_manager: Optional[RunManager] = None,
        genome_factory: "GenomeFactory" = None,
        overwrite_logs: bool = True,
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
            config: Evolution configuration (EvolutionConfig dataclass)
            run_manager: RunManager for storage (created if None)
            genome_factory: Optional pre-configured genome factory
            overwrite_logs: Whether to overwrite existing log files
        """
        self.evo_config = config if config is not None else DEFAULT_CONFIG

        # Initialize RunManager if not provided
        if run_manager is None:
            run_manager = RunManager(self.evo_config.storage)
        self.run_manager = run_manager

        self.train_query_ids = train_query_ids
        self.train_gt = train_ground_truth
        self.val_query_ids = val_query_ids
        self.val_gt = val_ground_truth

        # Initialize context and factory
        if genome_factory is None:
            self.evo_context = EvolutionContext(
                config=self.evo_config,
                generation=0,
                available_features=list(HeuristicRegistry.all().keys()),
                expression_features={
                    "movement": list(HeuristicRegistry.all_movement().keys()),
                    "ranking": list(HeuristicRegistry.all_ranking().keys()),
                    "deposit": list(HeuristicRegistry.all_deposit().keys()),
                },
            )
            self.genome_factory = GenomeFactory(self.evo_context)
        else:
            self.genome_factory = genome_factory
            self.evo_context = genome_factory.context

        # Check if LLM mutations are enabled (auto-enable decision tracking)
        mutation_strategy = self.evo_config.genetic.mutation_strategy
        llm_enabled = mutation_strategy == GeneticKey.LLM_MUTATION or self.evo_config.llm.enabled

        # Get device from context (centralized resolution)
        device = self.evo_context.resolved_device

        # Initialize population evaluator with decision tracking if LLM enabled
        self.population_evaluator = PopulationEvaluator(
            retriever=retriever,
            evaluator=evaluator,
            fitness_calc=fitness_calculator,
            queries=train_query_ids,
            ground_truth=train_ground_truth,
            concurrent_evaluations=self.evo_config.resources.concurrent_evaluations,
            max_workers_per_retrieval=self.evo_config.resources.max_workers_per_retrieval,
            track_decisions=llm_enabled,  # Auto-enable when LLM is used
            decision_sample_rate=1.0,  # 100% sampling for full LLM context
            early_exit_threshold=self.evo_config.resources.early_exit_threshold,
            device=device,  # Pass device for GPU-accelerated evaluation
            enable_shared_precompute=self.evo_config.resources.enable_shared_precompute,
            enable_cross_genome_metric_batch=self.evo_config.resources.enable_cross_genome_metric_batch,
        )

        # Initialize LLM Provider if using LLM Mutation
        if llm_enabled:
            self._initialize_llm_provider()

        # Initialize progress tracker
        self.tracker = ProgressTracker(
            log_path=self.run_manager.config.log_path,
            plot_path=self.run_manager.config.plot_path,
            plot_title=self.evo_config.storage.plot_title,
            overwrite=overwrite_logs,
        )

        # Initialize fitness strategy
        self.fitness_strategy = self._create_fitness_strategy()
        logger.info(f"Using Fitness Strategy: {self.fitness_strategy.__class__.__name__}")

        # Initialize MAP-Elites components (always enabled)
        logger.info("Initializing MAP-Elites archive and breeding loop")
        descriptor_calc = DescriptorRegistry.create_calculator(
            dimensions=self.evo_config.map_elites.dimensions,
            ranges=self.evo_config.map_elites.ranges,
        )
        comparator_config = ArchiveComparatorConfig(
            mode=ArchiveComparisonMode(self.evo_config.map_elites.comparison_mode)
        )
        self.map_elites_archive = MapElitesArchive(
            descriptor_calc=descriptor_calc,
            bins=self.evo_config.map_elites.bins,
            ranges=self.evo_config.map_elites.ranges,
            comparator_config=comparator_config,
        )
        self.map_elites_loop = MapElitesLoop(self.evo_context)

        # Create orchestrator
        self._orchestrator = MAPElitesOrchestrator(
            context=self.evo_context,
            evaluator=self.population_evaluator,
            fitness_strategy=self.fitness_strategy,
            tracker=self.tracker,
            run_manager=self.run_manager,
            val_query_ids=self.val_query_ids,
            val_ground_truth=self.val_gt,
            archive=self.map_elites_archive,
            me_loop=self.map_elites_loop,
            genome_factory=self.genome_factory,
        )

        # For checkpoint restoration
        self.restored_best_genome = None

    def _initialize_llm_provider(self):
        """Initialize LLM provider for LLM-guided mutations."""
        from .llm.factory import LLMProviderFactory

        llm_provider = LLMProviderFactory.create(self.evo_config)
        if llm_provider is not None:
            self.evo_context.llm_provider = llm_provider
            logger.info(
                f"ENABLED: LLM-Guided Mutation "
                f"(provider={self.evo_config.llm.provider}, "
                f"model={self.evo_config.llm.model})"
            )

    def _create_fitness_strategy(self):
        """Create the appropriate fitness strategy based on config."""
        strategy_name = self.evo_config.fitness_strategy

        if strategy_name == "pareto":
            return ParetoStrategy()
        else:
            # Default to lexicographic (including any legacy "phased" configs)
            return LexicographicStrategy()

    def optimize(self, initial_population: List[Genome] = None) -> Genome:
        """
        Run MAP-Elites evolutionary optimization.

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
        """Saves evolution state to disk via RunManager."""
        self._orchestrator.save_checkpoint(population, best_genome, generation)

    def initialize_run(self):
        """Initialize run directories and save config snapshot."""
        self.run_manager.initialize_run(self.evo_config)

    @classmethod
    def load_checkpoint(
        cls,
        run_manager: RunManager,
        checkpoint_path: Optional[str],
        retriever: Any,
        fitness_calculator: Any,
        evaluator: Any,
        train_query_ids: List[Any],
        train_ground_truth: List[List[Any]],
        val_query_ids: List[Any],
        val_ground_truth: List[List[Any]],
        config: EvolutionConfig,
    ) -> "EvolutionEngine":
        """
        Factory method: Creates a NEW engine instance and restores state from disk.

        Args:
            run_manager: RunManager for storage
            checkpoint_path: Path to checkpoint file (uses latest if None)
            retriever: SwarmRetriever instance
            fitness_calculator: FitnessCalculator instance
            evaluator: Evaluator instance
            train_query_ids: Training query IDs
            train_ground_truth: Training ground truth
            val_query_ids: Validation query IDs
            val_ground_truth: Validation ground truth
            config: Evolution configuration

        Returns:
            Initialized EvolutionEngine with restored state
        """
        # Load checkpoint via RunManager (handles device mapping)
        state = run_manager.load_checkpoint(checkpoint_path)
        print(f"--> Loaded checkpoint (generation {state['generation']})")

        # Create fresh engine with existing RunManager
        engine = cls(
            retriever=retriever,
            fitness_calculator=fitness_calculator,
            evaluator=evaluator,
            train_query_ids=train_query_ids,
            train_ground_truth=train_ground_truth,
            val_query_ids=val_query_ids,
            val_ground_truth=val_ground_truth,
            config=config,
            run_manager=run_manager,
            overwrite_logs=False,
        )

        # Restore evolutionary state
        engine.evo_context.population = state["population"]
        engine.evo_context.generation = state["generation"]

        # Restore RNG states for reproducibility
        if "random_state" in state:
            random.setstate(state["random_state"])
        if "torch_rng_state" in state:
            rng_state = state["torch_rng_state"]
            # Fix: Ensure RNG state is CPU tensor with uint8 dtype
            if isinstance(rng_state, torch.Tensor):
                rng_state = rng_state.cpu().to(torch.uint8)
            torch.set_rng_state(rng_state)

        # Restore tracker history
        if "tracker_history" in state:
            engine.tracker.history = state["tracker_history"]

        # Restore best genome
        engine.restored_best_genome = state.get("best_genome")

        print(f"  State restored. Resuming from Generation {state['generation']}")
        return engine
