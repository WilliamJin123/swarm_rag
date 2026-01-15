"""
Configuration validation and builder utilities for the evolution system.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

from .config import EvolutionConfigDict, DEFAULT_EVO_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

    def raise_if_invalid(self):
        """Raises ValueError if validation failed."""
        if not self.is_valid:
            raise ValueError(f"Invalid configuration: {'; '.join(self.errors)}")


class ConfigValidator:
    """Validates EvolutionConfigDict for consistency and completeness."""

    REQUIRED_FIELDS = [
        "n_generations",
        "population_size",
        "base_mutation_rate",
        "crossover_rate",
        "selection_strategy",
        "mutation_strategy",
        "crossover_strategy"
    ]

    @classmethod
    def validate(cls, config: EvolutionConfigDict) -> ValidationResult:
        """
        Validates an evolution configuration.

        Returns:
            ValidationResult with is_valid, errors, and warnings.
        """
        errors = []
        warnings = []

        # Required fields check
        for field in cls.REQUIRED_FIELDS:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Numeric bounds
        if config.get("n_generations", 1) < 1:
            errors.append("n_generations must be >= 1")

        if config.get("population_size", 1) < 2:
            errors.append("population_size must be >= 2")

        elite_frac = config.get("elite_fraction", 0.1)
        if not (0.0 <= elite_frac <= 1.0):
            errors.append("elite_fraction must be between 0.0 and 1.0")

        mutation_rate = config.get("base_mutation_rate", 0.2)
        if not (0.0 <= mutation_rate <= 1.0):
            errors.append("base_mutation_rate must be between 0.0 and 1.0")

        crossover_rate = config.get("crossover_rate", 0.6)
        if not (0.0 <= crossover_rate <= 1.0):
            errors.append("crossover_rate must be between 0.0 and 1.0")

        # MAP-Elites dimension consistency
        if config.get("map_elites_enabled"):
            dims = config.get("map_elites_dims", [])
            bins = config.get("map_elites_bins", [])
            ranges = config.get("map_elites_ranges", [])

            if len(dims) == 0:
                errors.append("map_elites_dims cannot be empty when MAP-Elites is enabled")

            if len(dims) != len(bins):
                errors.append(
                    f"map_elites_dims ({len(dims)}) must match map_elites_bins ({len(bins)}) in length"
                )

            if len(dims) != len(ranges):
                errors.append(
                    f"map_elites_dims ({len(dims)}) must match map_elites_ranges ({len(ranges)}) in length"
                )

            # Validate individual ranges
            for i, r in enumerate(ranges):
                if not isinstance(r, (list, tuple)) or len(r) != 2:
                    errors.append(f"map_elites_ranges[{i}] must be a (min, max) tuple")
                elif r[0] >= r[1]:
                    errors.append(f"map_elites_ranges[{i}]: min ({r[0]}) must be < max ({r[1]})")

            # Validate bins are positive
            for i, b in enumerate(bins):
                if b < 1:
                    errors.append(f"map_elites_bins[{i}] must be >= 1")

            initial_fill = config.get("map_elites_initial_fill", 100)
            if initial_fill < 1:
                errors.append("map_elites_initial_fill must be >= 1")

        # Boltzmann selection validation
        if config.get("selection_strategy") == "boltzmann":
            min_temp = config.get("boltzmann_min_temp", 0.1)
            max_temp = config.get("boltzmann_max_temp", 5.0)
            init_temp = config.get("boltzmann_temperature", 1.0)

            if min_temp <= 0:
                errors.append("boltzmann_min_temp must be > 0")

            if min_temp >= max_temp:
                errors.append("boltzmann_min_temp must be < boltzmann_max_temp")

            if not (min_temp <= init_temp <= max_temp):
                warnings.append(
                    f"boltzmann_temperature ({init_temp}) is outside bounds [{min_temp}, {max_temp}]"
                )

            alpha = config.get("boltzmann_alpha", 0.95)
            if not (0.0 < alpha < 1.0):
                errors.append("boltzmann_alpha must be between 0 and 1 (exclusive)")

        # LLM mutation validation
        if config.get("mutation_strategy") == "llm_mutation":
            if not config.get("llm_provider"):
                warnings.append("LLM mutation enabled but no llm_provider specified, using default")
            if not config.get("llm_model"):
                warnings.append("LLM mutation enabled but no llm_model specified, using default")

        # SwarmRAG param ranges validation
        param_ranges = config.get("swarmrag_param_ranges", {})
        for param_name, range_tuple in param_ranges.items():
            if not isinstance(range_tuple, (list, tuple)) or len(range_tuple) != 2:
                errors.append(f"swarmrag_param_ranges['{param_name}'] must be a (min, max) tuple")
            elif range_tuple[0] > range_tuple[1]:
                errors.append(
                    f"swarmrag_param_ranges['{param_name}']: min ({range_tuple[0]}) must be <= max ({range_tuple[1]})"
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class ConfigBuilder:
    """
    Fluent builder for creating validated evolution configurations.

    Example:
        config = (ConfigBuilder()
            .with_generations(50)
            .with_population(100, elite_fraction=0.15)
            .with_boltzmann_selection(temperature=1.5, adaptive=True)
            .with_map_elites(
                dims=["complexity", "n_agents"],
                bins=[10, 10],
                ranges=[(0, 100), (5, 30)]
            )
            .build())
    """

    def __init__(self, base_config: EvolutionConfigDict = None):
        """
        Initialize builder with optional base configuration.

        Args:
            base_config: Starting configuration (defaults to DEFAULT_EVO_CONFIG)
        """
        self._config: EvolutionConfigDict = (base_config or DEFAULT_EVO_CONFIG).copy()

    def with_generations(self, n: int) -> 'ConfigBuilder':
        """Set number of generations."""
        self._config["n_generations"] = n
        return self

    def with_population(
        self,
        size: int,
        elite_fraction: float = None
    ) -> 'ConfigBuilder':
        """Set population size and optionally elite fraction."""
        self._config["population_size"] = size
        if elite_fraction is not None:
            self._config["elite_fraction"] = elite_fraction
        return self

    def with_mutation(
        self,
        strategy: str = None,
        rate: float = None,
        max_expr_size: int = None,
        max_depth: int = None
    ) -> 'ConfigBuilder':
        """Configure mutation parameters."""
        if strategy is not None:
            self._config["mutation_strategy"] = strategy
        if rate is not None:
            self._config["base_mutation_rate"] = rate
        if max_expr_size is not None:
            self._config["mutation_max_expr_size"] = max_expr_size
        if max_depth is not None:
            self._config["expr_max_depth"] = max_depth
        return self

    def with_crossover(
        self,
        strategy: str = None,
        rate: float = None
    ) -> 'ConfigBuilder':
        """Configure crossover parameters."""
        if strategy is not None:
            self._config["crossover_strategy"] = strategy
        if rate is not None:
            self._config["crossover_rate"] = rate
        return self

    def with_selection(self, strategy: str, **kwargs) -> 'ConfigBuilder':
        """
        Configure selection strategy with strategy-specific parameters.

        For tournament: selection_k
        For boltzmann: use with_boltzmann_selection() instead
        """
        self._config["selection_strategy"] = strategy
        if "selection_k" in kwargs:
            self._config["selection_k"] = kwargs["selection_k"]
        return self

    def with_boltzmann_selection(
        self,
        temperature: float = 1.0,
        adaptive: bool = True,
        alpha: float = 0.95,
        min_temp: float = 0.1,
        max_temp: float = 5.0,
        diversity_threshold: float = 0.05
    ) -> 'ConfigBuilder':
        """Configure Boltzmann selection with all parameters."""
        self._config["selection_strategy"] = "boltzmann"
        self._config["boltzmann_temperature"] = temperature
        self._config["boltzmann_adaptive"] = adaptive
        self._config["boltzmann_alpha"] = alpha
        self._config["boltzmann_min_temp"] = min_temp
        self._config["boltzmann_max_temp"] = max_temp
        self._config["boltzmann_diversity_threshold"] = diversity_threshold
        return self

    def with_map_elites(
        self,
        dims: List[str],
        bins: List[int],
        ranges: List[Tuple[float, float]],
        initial_fill: int = 100
    ) -> 'ConfigBuilder':
        """
        Enable and configure MAP-Elites.

        Args:
            dims: Names of behavioral descriptors (e.g., ["complexity", "n_agents"])
            bins: Number of bins per dimension (e.g., [10, 10])
            ranges: (min, max) for each dimension (e.g., [(0, 100), (5, 30)])
            initial_fill: Initial random population size to seed archive
        """
        self._config["map_elites_enabled"] = True
        self._config["map_elites_dims"] = dims
        self._config["map_elites_bins"] = bins
        self._config["map_elites_ranges"] = ranges
        self._config["map_elites_initial_fill"] = initial_fill
        return self

    def without_map_elites(self) -> 'ConfigBuilder':
        """Disable MAP-Elites (use standard GA)."""
        self._config["map_elites_enabled"] = False
        return self

    def with_llm_mutation(
        self,
        provider: str = "cerebras",
        model: str = "llama-3.3-70b-versatile",
        env_path: str = ".env",
        concurrency: int = 50
    ) -> 'ConfigBuilder':
        """Enable LLM-guided mutation."""
        self._config["mutation_strategy"] = "llm_mutation"
        self._config["llm_provider"] = provider
        self._config["llm_model"] = model
        self._config["llm_env_path"] = env_path
        self._config["llm_concurrency"] = concurrency
        return self

    def with_fitness_strategy(
        self,
        strategy: str,
        phased_switch_gen: int = None
    ) -> 'ConfigBuilder':
        """
        Set fitness assignment strategy.

        Args:
            strategy: One of "lexicographic", "pareto", "phased"
            phased_switch_gen: Generation to switch strategies (for "phased" only)
        """
        self._config["fitness_strategy"] = strategy
        if phased_switch_gen is not None:
            self._config["phased_switch_gen"] = phased_switch_gen
        return self

    def with_swarmrag_ranges(
        self,
        n_agents: Tuple[int, int] = None,
        steps: Tuple[int, int] = None,
        decay: Tuple[float, float] = None,
        initial_pool_size: Tuple[int, int] = None,
        start_subset: Tuple[int, int] = None,
        drop_zone_inc: Tuple[float, float] = None
    ) -> 'ConfigBuilder':
        """Configure SwarmRAG parameter search ranges."""
        ranges = self._config.get("swarmrag_param_ranges", {}).copy()
        if n_agents is not None:
            ranges["n_agents"] = n_agents
        if steps is not None:
            ranges["steps"] = steps
        if decay is not None:
            ranges["decay"] = decay
        if initial_pool_size is not None:
            ranges["initial_pool_size"] = initial_pool_size
        if start_subset is not None:
            ranges["start_subset"] = start_subset
        if drop_zone_inc is not None:
            ranges["drop_zone_inc"] = drop_zone_inc
        self._config["swarmrag_param_ranges"] = ranges
        return self

    def with_logging(
        self,
        log_path: str = None,
        plot_path: str = None,
        plot_title: str = None,
        checkpoint_path: str = None,
        checkpoint_frequency: int = None,
        validation_frequency: int = None
    ) -> 'ConfigBuilder':
        """Configure logging and checkpointing."""
        if log_path is not None:
            self._config["log_path"] = log_path
        if plot_path is not None:
            self._config["plot_path"] = plot_path
        if plot_title is not None:
            self._config["plot_title"] = plot_title
        if checkpoint_path is not None:
            self._config["checkpoint_path"] = checkpoint_path
        if checkpoint_frequency is not None:
            self._config["checkpoint_frequency"] = checkpoint_frequency
        if validation_frequency is not None:
            self._config["validation_frequency"] = validation_frequency
        return self

    def with_concurrency(
        self,
        concurrent_evaluations: int = None,
        max_workers_per_retrieval: int = None
    ) -> 'ConfigBuilder':
        """Configure parallel evaluation settings."""
        if concurrent_evaluations is not None:
            self._config["concurrent_evaluations"] = concurrent_evaluations
        if max_workers_per_retrieval is not None:
            self._config["max_workers_per_retrieval"] = max_workers_per_retrieval
        return self

    def with_agent_groups(self, n_groups: int) -> 'ConfigBuilder':
        """Set number of agent groups for heterogeneous swarms."""
        self._config["n_agent_groups"] = n_groups
        return self

    def build(self, validate: bool = True) -> EvolutionConfigDict:
        """
        Build and optionally validate the configuration.

        Args:
            validate: If True, validates config and raises on errors

        Returns:
            The built configuration dictionary

        Raises:
            ValueError: If validation is enabled and config is invalid
        """
        if validate:
            result = ConfigValidator.validate(self._config)

            # Log warnings
            for warning in result.warnings:
                logger.warning(f"Config warning: {warning}")

            # Raise on errors
            result.raise_if_invalid()

        return self._config.copy()

    def build_unsafe(self) -> EvolutionConfigDict:
        """Build without validation (use with caution)."""
        return self._config.copy()
