from typing import Dict, List

class FitnessCalculator:
    def __init__(self, weights: Dict[str, float]):
        """
        Args:
            weights: Dict mapping MetricRegistry names to their importance.
                     e.g. {'Recall@20': 0.7, 'MRR': 0.2, 'latency': -0.1}
        """
        self.weights = weights

    def calculate(self, metrics: Dict[str, float]) -> float:
        score = 0.0
        for metric_name, weight in self.weights.items():
            # If the metric wasn't calculated, decide policy (skip or fail)
            val = metrics.get(metric_name, 0.0)
            score += val * weight
        return max(0.0, score) # Ensure non-negative fitness if desired