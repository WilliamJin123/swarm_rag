"""
Fitness Calculation for Evolution

Provides multiple fitness calculation modes for MAP-Elites evolution:
- WEIGHTED_SUM: Original simple weighted average
- MIN_METRIC: Bottleneck - quality = min(all_metrics)
- GEOMETRIC_MEAN: Penalizes zeros harshly, ensures balanced optimization
- THRESHOLD_BONUS: Base score + bonus for meeting thresholds
- HYBRID: Recommended - geometric mean + threshold bonuses + min-metric penalty
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from swarm_rag.evolution.types.fitness_results import FitnessResult
from swarm_rag.evolution.types.genome import Genome


class FitnessMode(Enum):
    """Available fitness calculation modes."""
    WEIGHTED_SUM = "weighted_sum"
    MIN_METRIC = "min_metric"
    GEOMETRIC_MEAN = "geometric_mean"
    THRESHOLD_BONUS = "threshold_bonus"
    HYBRID = "hybrid"


@dataclass
class MetricConfig:
    """Configuration for a single metric in fitness calculation."""
    weight: float = 1.0
    threshold: float = 0.5  # Target threshold to achieve
    min_acceptable: float = 0.1  # Below this is penalized


@dataclass
class FitnessConfig:
    """Configuration for the advanced fitness calculator."""
    mode: FitnessMode = FitnessMode.WEIGHTED_SUM

    # Metric configurations
    metric_configs: Dict[str, MetricConfig] = field(default_factory=lambda: {
        "Hit@1": MetricConfig(weight=0.25, threshold=0.60, min_acceptable=0.10),
        "Hit@5": MetricConfig(weight=0.25, threshold=0.80, min_acceptable=0.20),
        "MRR": MetricConfig(weight=0.25, threshold=0.80, min_acceptable=0.20),
        "Recall@20": MetricConfig(weight=0.25, threshold=0.85, min_acceptable=0.30),
    })

    # Mode-specific parameters
    threshold_bonus_amount: float = 0.05  # Bonus per threshold met
    all_thresholds_bonus: float = 0.10  # Extra bonus when all thresholds met
    min_metric_penalty_factor: float = 0.1  # Penalty factor for below-minimum

    # Stability weight (for hybrid scoring)
    stability_weight: float = 0.1


class FitnessCalculator:
    """
    Unified fitness calculator with multiple modes.

    Modes: WEIGHTED_SUM, MIN_METRIC, GEOMETRIC_MEAN, THRESHOLD_BONUS, HYBRID

    Example:
        # Simple weights (via factory)
        calc = FitnessCalculator.from_weights({"Recall@20": 0.7, "MRR": 0.3})

        # Full config (recommended)
        config = FitnessConfig(mode=FitnessMode.HYBRID)
        calc = FitnessCalculator(config=config)

        # Factory function
        calc = create_fitness_calculator(mode="hybrid")
    """

    def __init__(
        self,
        config: FitnessConfig = None,
        *,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize fitness calculator.

        Args:
            config: FitnessConfig for all modes (recommended)
            weights: DEPRECATED - use FitnessCalculator.from_weights() instead.
                     Simple weights dict for backward compatibility only.

        Note: If both config and weights are provided, config takes precedence.
        """
        if config is not None:
            self.config = config
        elif weights is not None:
            # Deprecated path - convert weights dict to FitnessConfig
            import warnings
            warnings.warn(
                "FitnessCalculator(weights=...) is deprecated. "
                "Use FitnessCalculator.from_weights(...) or pass a FitnessConfig instead.",
                DeprecationWarning,
                stacklevel=2
            )
            self.config = self._config_from_weights(weights)
        else:
            # Default config
            self.config = FitnessConfig()

    @classmethod
    def from_weights(cls, weights: Dict[str, float]) -> "FitnessCalculator":
        """
        Create a fitness calculator from simple weights dict.

        This is the recommended way to create a simple weighted-sum calculator.

        Args:
            weights: Dictionary mapping metric names to weights

        Returns:
            FitnessCalculator configured for weighted sum mode

        Example:
            calc = FitnessCalculator.from_weights({"Recall@20": 0.7, "MRR": 0.3})
        """
        config = cls._config_from_weights(weights)
        return cls(config=config)

    @staticmethod
    def _config_from_weights(weights: Dict[str, float]) -> FitnessConfig:
        """Convert simple weights dict to FitnessConfig."""
        metric_configs = {}
        for name, weight in weights.items():
            metric_configs[name] = MetricConfig(weight=weight)
        return FitnessConfig(
            mode=FitnessMode.WEIGHTED_SUM,
            metric_configs=metric_configs
        )

    def calculate(
        self,
        metrics: Dict[str, float],
        genome: Genome,
        generation: int = 0,
    ) -> FitnessResult:
        """
        Calculate fitness using the configured mode.

        Args:
            metrics: Dict of metric name -> value
            genome: The genome being evaluated
            generation: Current generation (unused, kept for API compatibility)

        Returns:
            FitnessResult with quality, stability, and cost scores
        """
        # Get weights from config
        effective_weights = self._get_weights()

        # Calculate quality based on mode
        mode = self.config.mode

        if mode == FitnessMode.WEIGHTED_SUM:
            quality_score = self._weighted_sum(metrics, effective_weights)
        elif mode == FitnessMode.MIN_METRIC:
            quality_score = self._min_metric(metrics, effective_weights)
        elif mode == FitnessMode.GEOMETRIC_MEAN:
            quality_score = self._geometric_mean(metrics, effective_weights)
        elif mode == FitnessMode.THRESHOLD_BONUS:
            quality_score = self._threshold_bonus(metrics, effective_weights)
        elif mode == FitnessMode.HYBRID:
            quality_score = self._hybrid(metrics, effective_weights)
        else:
            quality_score = self._weighted_sum(metrics, effective_weights)

        # Calculate stability from variance (higher stability = lower variance)
        stability = 1.0 - metrics.get("variance", 0.0)

        return FitnessResult(
            quality_score=quality_score,
            stability_score=stability,
        )

    def _get_weights(self) -> Dict[str, float]:
        """Get weights from metric configs."""
        return {
            name: cfg.weight
            for name, cfg in self.config.metric_configs.items()
        }

    def _weighted_sum(
        self,
        metrics: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """Original weighted sum calculation."""
        total = 0.0
        total_weight = 0.0

        for name, weight in weights.items():
            if name in metrics:
                total += metrics[name] * weight
                total_weight += abs(weight)

        return total / total_weight if total_weight > 0 else 0.0

    def _min_metric(
        self,
        metrics: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """
        Bottleneck approach: quality = min(weighted metrics).
        Forces balanced improvement across all metrics.
        """
        weighted_values = []
        for name, weight in weights.items():
            if name in metrics and weight > 0:
                # Scale by weight to make metrics comparable
                weighted_values.append(metrics[name] * weight)

        return min(weighted_values) if weighted_values else 0.0

    def _geometric_mean(
        self,
        metrics: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """
        Geometric mean of weighted metrics.
        Penalizes zeros harshly - any zero metric makes the whole score zero.

        Formula: (m1^w1 * m2^w2 * ...)^(1/sum(weights))
        """
        import math

        product = 1.0
        total_weight = 0.0

        for name, weight in weights.items():
            if name in metrics and weight > 0:
                value = max(metrics[name], 1e-10)  # Avoid zero
                product *= value ** weight
                total_weight += weight

        if total_weight > 0:
            return product ** (1.0 / total_weight)
        return 0.0

    def _threshold_bonus(
        self,
        metrics: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """
        Base weighted sum + bonus for meeting thresholds.
        """
        base_score = self._weighted_sum(metrics, weights)

        bonus = 0.0
        thresholds_met = 0
        total_thresholds = 0

        for name, cfg in self.config.metric_configs.items():
            if name in metrics:
                total_thresholds += 1
                if metrics[name] >= cfg.threshold:
                    thresholds_met += 1
                    bonus += self.config.threshold_bonus_amount

        # Extra bonus if all thresholds met
        if total_thresholds > 0 and thresholds_met == total_thresholds:
            bonus += self.config.all_thresholds_bonus

        return min(1.0, base_score + bonus)

    def _hybrid(
        self,
        metrics: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """
        Recommended mode: Combines multiple strategies.

        1. Geometric mean as base (ensures balance)
        2. Bonus for meeting thresholds
        3. Penalty for metrics below minimum acceptable
        """
        # 1. Geometric mean base
        geo_mean = self._geometric_mean(metrics, weights)

        # 2. Threshold bonuses
        bonus = 0.0
        thresholds_met = 0
        total_thresholds = 0

        for name, cfg in self.config.metric_configs.items():
            if name in metrics:
                total_thresholds += 1
                if metrics[name] >= cfg.threshold:
                    thresholds_met += 1
                    bonus += self.config.threshold_bonus_amount

        if total_thresholds > 0 and thresholds_met == total_thresholds:
            bonus += self.config.all_thresholds_bonus

        # 3. Penalty for below-minimum metrics
        penalty = 0.0
        for name, cfg in self.config.metric_configs.items():
            if name in metrics and metrics[name] < cfg.min_acceptable:
                shortfall = cfg.min_acceptable - metrics[name]
                penalty += self.config.min_metric_penalty_factor * shortfall

        # Combine
        score = geo_mean + bonus - penalty

        # Clamp to valid range
        return max(0.0, min(1.0, score))

    def get_threshold_report(self, metrics: Dict[str, float]) -> Dict[str, Dict]:
        """
        Generate a report on which thresholds are met.

        Returns:
            Dict with metric name -> {value, threshold, met, gap}
        """
        report = {}
        for name, cfg in self.config.metric_configs.items():
            value = metrics.get(name, 0.0)
            report[name] = {
                "value": value,
                "threshold": cfg.threshold,
                "min_acceptable": cfg.min_acceptable,
                "met": value >= cfg.threshold,
                "above_min": value >= cfg.min_acceptable,
                "gap": cfg.threshold - value if value < cfg.threshold else 0.0
            }
        return report


# Factory function for easy creation
def create_fitness_calculator(
    mode: str = "weighted_sum",
    weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None
) -> FitnessCalculator:
    """
    Factory function to create a fitness calculator.

    Args:
        mode: Fitness mode name (weighted_sum, min_metric, geometric_mean,
              threshold_bonus, hybrid)
        weights: Optional custom weights per metric
        thresholds: Optional custom thresholds per metric

    Returns:
        Configured FitnessCalculator
    """
    # Map string to enum
    mode_enum = FitnessMode(mode)

    # Build metric configs
    default_configs = {
        "Hit@1": MetricConfig(weight=0.25, threshold=0.60, min_acceptable=0.10),
        "Hit@5": MetricConfig(weight=0.25, threshold=0.80, min_acceptable=0.20),
        "MRR": MetricConfig(weight=0.25, threshold=0.80, min_acceptable=0.20),
        "Recall@20": MetricConfig(weight=0.25, threshold=0.85, min_acceptable=0.30),
    }

    # Apply custom weights
    if weights:
        for name, weight in weights.items():
            if name in default_configs:
                default_configs[name].weight = weight
            else:
                default_configs[name] = MetricConfig(weight=weight)

    # Apply custom thresholds
    if thresholds:
        for name, threshold in thresholds.items():
            if name in default_configs:
                default_configs[name].threshold = threshold

    config = FitnessConfig(
        mode=mode_enum,
        metric_configs=default_configs,
    )

    return FitnessCalculator(config=config)
