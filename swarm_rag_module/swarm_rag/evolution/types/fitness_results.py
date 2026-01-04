from dataclasses import dataclass
import functools

QUALITY_TOLERANCE = 0.005
STABILITY_TOLERANCE = 0.05

@dataclass
@functools.total_ordering
class FitnessResult:
    """
    Multi-objective fitness to support Lexicographic Selection.
    """
    quality_score: float   # Primary (Recall, MRR)
    stability_score: float # Secondary (Variance)
    cost_score: float      # Tertiary (Latency)

    # Magic methods allow sorting to work natively with Python's sort() and max()
    def _get_sort_key(self):
        """
        Creates a key for comparison that incorporates tolerance.
        We round the scores to a precision that reflects our tolerance.
        This ensures that values within the tolerance are treated as equal,
        creating a proper total ordering.
        """
        quality_precision = -int(f"{QUALITY_TOLERANCE:e}".split('e')[1])
        stability_precision = -int(f"{STABILITY_TOLERANCE:e}".split('e')[1])
        
        return (
            -round(self.quality_score, quality_precision),    # Negate to sort descending
            -round(self.stability_score, stability_precision), # Negate to sort descending
            self.cost_score                                    # Sort ascending
        )
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