from dataclasses import dataclass
from typing import Dict

from swarm_rag.evolution.types.genome import Genome

@dataclass
class FitnessResult:
    """
    Multi-objective fitness to support Lexicographic Selection.
    """
    quality_score: float  # Primary (Recall/MRR, etc.)
    cost_score: float     # Secondary (Latency/Steps)

    # Magic methods allow sorting to work natively with Python's sort() and max()
    def __lt__(self, other):
        # Primary Objective (with tolerance epsilon, e.g., 0.01)
        if abs(self.quality_score - other.quality_score) > 0.01:
            return self.quality_score < other.quality_score
        # Seoncdary (Lower cost is better, so reverse logic)
        return self.cost_score > other.cost_score

    def __float__(self):
        """Allows legacy code expecting a float to still run (returns quality)."""
        return self.quality_score

class FitnessCalculator:
    def __init__(self, weights: Dict[str, float]):
        """
        Args:
            weights: Dict mapping MetricRegistry names to their importance.
                     e.g. {'Recall@20': 0.7, 'MRR': 0.2, 'latency': -0.1}
        """
        self.weights = weights

    def calculate(self, metrics: Dict[str, float], genome: Genome) -> float:
        # Quality (Weighted Sum of Retrieval Metrics)
        quality = 0.0
        for name, w in self.weights.items():
            quality += metrics.get(name, 0.0) * w
            
        # Penalize complexity lightly in the quality score
        complexity_penalty = 0.0005 * genome.complexity()
        quality -= complexity_penalty

        # Cost (Latency or Steps)
        cost = metrics.get("latency_ms", 0.0)
        
        return FitnessResult(quality, cost)