"""
Convergence Detection Configuration.

Dataclass for configuring convergence detection behavior including
sliding window parameters, thresholds, and adaptive settings.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConvergenceConfig:
    """
    Configuration for convergence detection.

    Controls when evolution should terminate early due to stagnation.

    Attributes:
        enabled: Master switch for convergence detection
        window_size: Number of generations in sliding window (default 40)
        min_window_size: Minimum window size for adaptive sizing
        max_window_size: Maximum window size for adaptive sizing
        adaptive_window: Enable adaptive window sizing based on improvement rate
        threshold_percentage: Minimum relative improvement required (0.001 = 0.1%)
        grace_period: Minimum generations before detection activates
        theoretical_max: Optional theoretical maximum QD-score for headroom calculation
    """
    enabled: bool = True
    window_size: int = 40
    min_window_size: int = 20
    max_window_size: int = 60
    adaptive_window: bool = True
    threshold_percentage: float = 0.001  # 0.1% relative improvement required
    grace_period: int = 20
    theoretical_max: Optional[float] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary for checkpointing."""
        return {
            "enabled": self.enabled,
            "window_size": self.window_size,
            "min_window_size": self.min_window_size,
            "max_window_size": self.max_window_size,
            "adaptive_window": self.adaptive_window,
            "threshold_percentage": self.threshold_percentage,
            "grace_period": self.grace_period,
            "theoretical_max": self.theoretical_max,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConvergenceConfig":
        """Deserialize from dictionary."""
        return cls(
            enabled=d.get("enabled", True),
            window_size=d.get("window_size", 40),
            min_window_size=d.get("min_window_size", 20),
            max_window_size=d.get("max_window_size", 60),
            adaptive_window=d.get("adaptive_window", True),
            threshold_percentage=d.get("threshold_percentage", 0.001),
            grace_period=d.get("grace_period", 20),
            theoretical_max=d.get("theoretical_max"),
        )
