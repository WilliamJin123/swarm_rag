from dataclasses import dataclass, field
import functools
import math

QUALITY_TOLERANCE = 0.005
STABILITY_TOLERANCE = 0.05

@dataclass
@functools.total_ordering
class FitnessResult:
    """
    Multi-objective fitness to support Lexicographic Selection.
    """
    quality_score: float = field(default=-math.inf, compare=True)   # maximise
    stability_score: float = field(default=-math.inf, compare=True) # maximise
    cost_score: float = field(default=math.inf, compare=True)      # minimise

    # Magic methods allow sorting to work natively with Python's sort() and max()
    def _get_sort_key(self):
        """
        Creates a key for comparison that incorporates tolerance.
        We round the scores to a precision that reflects our tolerance.
        This ensures that values within the tolerance are treated as equal,
        creating a proper total ordering.
        """
        quality_prec = FitnessResult._precision_from_tolerance(QUALITY_TOLERANCE)
        stability_prec = FitnessResult._precision_from_tolerance(STABILITY_TOLERANCE)
        quality = round(self.quality_score, quality_prec)
        stability = round(self.stability_score, stability_prec)
        return (quality, stability, -self.cost_score)
    
    def __lt__(self, other: 'FitnessResult') -> bool:
        """Compares based on the sort key."""
        if not isinstance(other, FitnessResult):
            return NotImplemented
        return self._get_sort_key() < other._get_sort_key()

    def __eq__(self, other: 'FitnessResult') -> bool:
        """Compares based on the sort key."""
        if not isinstance(other, FitnessResult):
            return NotImplemented
        return self._get_sort_key() == other._get_sort_key()

    def __float__(self):
        """Allows legacy code expecting a float to still run (returns quality)."""
        return self.quality_score
    
    @staticmethod
    def _precision_from_tolerance(tol: float) -> int:
        """
        Return the number of decimal places that faithfully represent the tolerance.
        Example: 0.005 → 2 decimal places (since 0.01 > 0.005 ≥ 0.001).
        """
        # log10(0.005) = -2.301..., we need ceil(abs(...)) = 2
        return max(0, int(math.ceil(-math.log10(tol))))