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

    def calculate(self, metrics: Dict[str, float], genome: Genome) -> FitnessResult:
        quality_score = 0.0
        for metric_name, weight in self.weights.items():
            metric_value = metrics.get(metric_name, 0.0)
            quality_score += metric_value * weight

        # Normalize
        total_abs_weight = sum(abs(w) for w in self.weights.values())
        if total_abs_weight > 0:
            quality_score = quality_score / total_abs_weight
        else:
            quality_score = 0.0
            
        # Stability
        stability = 1.0 - metrics.get("variance", 0.0)

        # Cost (Latency or Steps)
        cost = metrics.get("latency", 0.0)
        
        return FitnessResult(
            quality_score=quality_score,
            stability_score=stability,
            cost_score=cost
        )