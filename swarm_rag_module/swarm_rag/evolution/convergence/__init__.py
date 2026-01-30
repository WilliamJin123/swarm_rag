"""
Convergence Detection Module.

Provides early stopping for evolution when QD-score improvement stagnates.
"""
from .config import ConvergenceConfig
from .detector import ConvergenceDetector, TerminationReason

__all__ = ["ConvergenceConfig", "ConvergenceDetector", "TerminationReason"]
