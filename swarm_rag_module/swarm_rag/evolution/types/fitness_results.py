from dataclasses import dataclass, field
import functools
import math
from typing import Dict

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
    
    # Custom sort key for flexible strategies (Lexicographic, Pareto, etc.)
    # Defaults to None, which triggers lazy computation of Lexicographic key
    sort_key: tuple = field(default=None, compare=False)

    def update_sort_key(self, mode="lexicographic"):
        """Explicitly updates the sort key based on the mode."""
        if mode == "lexicographic":
            quality_prec = FitnessResult._precision_from_tolerance(QUALITY_TOLERANCE)
            stability_prec = FitnessResult._precision_from_tolerance(STABILITY_TOLERANCE)
            quality = round(self.quality_score, quality_prec)
            stability = round(self.stability_score, stability_prec)
            self.sort_key = (quality, stability, -self.cost_score)
        else:
            # Other modes should set sort_key directly via strategy
            pass

    def _get_sort_key(self):
        """
        Returns the cached sort key or computes default lexicographic key.
        """
        if self.sort_key is not None:
            return self.sort_key
            
        # Fallback (Legacy/Default)
        self.update_sort_key("lexicographic")
        return self.sort_key
    
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
    
    def to_dict(self) -> Dict[str, float]:
        """
        Serializes the FitnessResult into a plain Python dictionary.

        Returns:
            A dictionary of the fitness scores.
        """
        return {
            "quality_score": self.quality_score,
            "stability_score": self.stability_score,
            "cost_score": self.cost_score,
            "sort_key": self.sort_key
        }

    @staticmethod
    def _precision_from_tolerance(tol: float) -> int:
        """
        Return the number of decimal places that faithfully represent the tolerance.
        Example: 0.005 → 2 decimal places (since 0.01 > 0.005 ≥ 0.001).
        """
        # log10(0.005) = -2.301..., we need ceil(abs(...)) = 2
        return max(0, int(math.ceil(-math.log10(tol))))