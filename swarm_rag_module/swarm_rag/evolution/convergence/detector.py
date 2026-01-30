"""
Convergence Detector for Evolution Termination.

Implements sliding window QD-score analysis to detect when evolution
has stagnated and further generations are unlikely to yield improvement.
"""
import logging
from collections import deque
from enum import Enum
from typing import Any, Dict, Optional

from .config import ConvergenceConfig


class TerminationReason(Enum):
    """
    Reasons for evolution termination.

    Used to distinguish between different termination scenarios
    in checkpoints and logs.
    """
    MAX_GENERATIONS = "max_generations"
    CONVERGENCE = "convergence"
    MEMORY_LIMIT = "memory_limit"
    USER_INTERRUPT = "user_interrupt"


class ConvergenceDetector:
    """
    Detects evolution convergence using sliding window QD-score analysis.

    Tracks QD-scores over a configurable window and detects stagnation
    when the relative improvement falls below a threshold percentage.

    Features:
    - Sliding window with O(1) operations using deque
    - Adaptive window sizing based on improvement rate
    - Grace period before detection activates
    - Optional headroom-based threshold adjustment

    Example:
        config = ConvergenceConfig(window_size=10, grace_period=5)
        detector = ConvergenceDetector(config)

        for gen in range(100):
            qd_score = evolve_one_generation()
            detector.record(qd_score)

            if detector.is_converged():
                print(f"Converged at generation {gen}")
                break
    """

    def __init__(self, config: ConvergenceConfig):
        """
        Initialize convergence detector.

        Args:
            config: ConvergenceConfig with window_size, threshold, etc.
        """
        self.config = config
        self._window: deque = deque(maxlen=config.window_size)
        self._generation_count: int = 0
        self._last_improvement_gen: int = 0
        self._previous_max: float = 0.0
        self._logger = logging.getLogger(__name__)

    def record(self, qd_score: float) -> None:
        """
        Record a QD-score observation.

        Appends to sliding window, tracks improvement, and
        adapts window size if enabled.

        Args:
            qd_score: The QD-score for the current generation
        """
        self._window.append(qd_score)
        self._generation_count += 1

        # Track improvement
        if qd_score > self._previous_max:
            improvement = qd_score - self._previous_max
            if self._previous_max > 0:
                relative_improvement = improvement / self._previous_max
                if relative_improvement > self.config.threshold_percentage:
                    self._last_improvement_gen = self._generation_count
            else:
                # First non-zero score
                self._last_improvement_gen = self._generation_count
            self._previous_max = qd_score

        # Adapt window size if enabled
        if self.config.adaptive_window:
            self._adapt_window_size()

    def is_converged(self) -> bool:
        """
        Check if evolution has converged.

        Returns True if:
        1. Grace period has passed
        2. Window has sufficient data
        3. Stagnation is detected

        Returns:
            True if evolution is converged, False otherwise
        """
        # Check grace period
        if self._generation_count < self.config.grace_period:
            return False

        # Check window has sufficient data
        if len(self._window) < self.config.min_window_size:
            return False

        return self._check_stagnation()

    def _check_stagnation(self) -> bool:
        """
        Check if QD-score improvement indicates stagnation.

        Calculates the relative improvement ratio (max-min)/min
        and compares against the threshold.

        Returns:
            True if stagnating (improvement below threshold)
        """
        if len(self._window) == 0:
            return False

        window_min = min(self._window)
        window_max = max(self._window)

        # Avoid division by zero or negative values
        if window_min <= 0:
            return False

        # Calculate relative improvement within window
        improvement_ratio = (window_max - window_min) / window_min

        # Get threshold (potentially adjusted for headroom)
        threshold = self._calculate_threshold(window_max)

        is_stagnating = improvement_ratio < threshold

        if is_stagnating:
            self._logger.debug(
                f"Stagnation detected: improvement_ratio={improvement_ratio:.6f}, "
                f"threshold={threshold:.6f}, window_min={window_min:.2f}, "
                f"window_max={window_max:.2f}"
            )

        return is_stagnating

    def _calculate_threshold(self, current_score: float) -> float:
        """
        Calculate threshold, potentially adjusted by headroom.

        If theoretical_max is set, threshold is relaxed as we approach it
        (since improvements naturally slow near the maximum).

        Args:
            current_score: Current QD-score for headroom calculation

        Returns:
            Adjusted threshold percentage
        """
        base_threshold = self.config.threshold_percentage

        if self.config.theoretical_max is None:
            return base_threshold

        # Calculate headroom (how much room left to improve)
        if current_score >= self.config.theoretical_max:
            # Already at max, use base threshold
            return base_threshold

        headroom = (self.config.theoretical_max - current_score) / self.config.theoretical_max

        # As headroom decreases, relax threshold
        # At 50% headroom: threshold unchanged
        # At 10% headroom: threshold reduced to 20% of original
        headroom_factor = max(0.2, headroom * 2)

        return base_threshold * headroom_factor

    def _adapt_window_size(self) -> None:
        """
        Adapt window size based on improvement rate.

        - Expand window when improving frequently (gens since improvement < 5)
        - Shrink window when flat (gens since improvement > 15)

        Window data is preserved when resizing.
        """
        gens_since_improvement = self._generation_count - self._last_improvement_gen
        current_size = self._window.maxlen

        new_size: Optional[int] = None

        if gens_since_improvement < 5:
            # Improving rapidly - expand window to gather more evidence
            new_size = min(current_size + 5, self.config.max_window_size)
        elif gens_since_improvement > 15:
            # Flat for a while - shrink window for faster detection
            new_size = max(current_size - 5, self.config.min_window_size)

        if new_size is not None and new_size != current_size:
            # Preserve window data while resizing
            old_data = list(self._window)
            self._window = deque(old_data, maxlen=new_size)
            self._logger.debug(
                f"Adapted window size: {current_size} -> {new_size} "
                f"(gens_since_improvement={gens_since_improvement})"
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get detector statistics.

        Returns:
            Dictionary with generation_count, window_size, window_min/max,
            last_improvement_gen, etc.
        """
        window_list = list(self._window)

        return {
            "generation_count": self._generation_count,
            "window_size": self._window.maxlen,
            "window_fill": len(self._window),
            "window_min": min(window_list) if window_list else 0.0,
            "window_max": max(window_list) if window_list else 0.0,
            "last_improvement_gen": self._last_improvement_gen,
            "gens_since_improvement": self._generation_count - self._last_improvement_gen,
            "previous_max": self._previous_max,
            "config": self.config.to_dict(),
        }

    def get_window_stats(self) -> str:
        """
        Get human-readable window statistics for logging.

        Returns:
            String summary like "min=100.5, max=101.2, size=40/40"
        """
        window_list = list(self._window)
        if not window_list:
            return "empty"

        window_min = min(window_list)
        window_max = max(window_list)

        return (
            f"min={window_min:.2f}, max={window_max:.2f}, "
            f"size={len(window_list)}/{self._window.maxlen}"
        )
