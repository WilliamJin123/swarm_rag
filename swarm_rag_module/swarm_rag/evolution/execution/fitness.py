from dataclasses import dataclass
from typing import Dict

from swarm_rag.evolution.types.fitness_results import FitnessResult
from swarm_rag.evolution.types.genome import Genome



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
        quality_score = sum(metrics.get(name, 0.0) * w for name, w in self.weights.items())
            
        # Penalize complexity lightly in the quality score
        complexity_penalty = 0.0005 * genome.complexity()
        quality_score -= complexity_penalty

        # Stability
        stability = 1.0 - metrics.get("variance", 0.0)

        # Cost (Latency or Steps)
        cost = metrics.get("latency_ms", 0.0)
        
        return FitnessResult(
            quality_score=quality_score,
            stability_score=stability,
            cost_score=cost
        )