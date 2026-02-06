"""Early exit policy for genome evaluation.

Encapsulates the quarter-checkpoint early exit decision logic.
Genomes scoring below the threshold at the 25% checkpoint are
terminated early, saving ~75% of evaluation time for poor performers.
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default early exit threshold at quarter checkpoint (25% of queries)
DEFAULT_EARLY_EXIT_THRESHOLD: float = 0.30


@dataclass
class EarlyExitPolicy:
    """Policy for deciding whether to early-exit genome evaluation.

    The quarter-checkpoint approach evaluates 25% of queries first,
    then decides whether to continue based on the quality score.

    Args:
        threshold: Minimum quality score to continue past quarter checkpoint
        enabled: Whether adaptive early exit is active
    """
    threshold: float = DEFAULT_EARLY_EXIT_THRESHOLD
    enabled: bool = True

    def compute_quarter(self, n_queries: int) -> int:
        """Compute the quarter checkpoint size.

        Returns at least 1 to ensure early exit always has data.
        """
        return max(1, n_queries // 4)

    def should_exit(self, quality_score: float) -> bool:
        """Decide whether to early-exit based on quarter fitness.

        Args:
            quality_score: The quality_score from FitnessResult at quarter checkpoint

        Returns:
            True if the genome should be terminated early (below threshold)
        """
        if not self.enabled:
            return False
        return quality_score < self.threshold
