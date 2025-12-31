from typing import Dict

from swarm_rag.evolution.genome import Genome

# TO CHANGE: MORE CUSTOMIZATION and better way of passing in weights
class FitnessCalculator:
    def __init__(self, weights: Dict[str, float]):
        """
        Args:
            weights: Dict mapping MetricRegistry names to their importance.
                     e.g. {'Recall@20': 0.7, 'MRR': 0.2, 'latency': -0.1}
        """
        self.weights = weights

    def calculate(self, metrics: Dict[str, float], genome: Genome) -> float:
        base_fitness = 0.0
        for metric_name, weight in self.weights.items():
            # If the metric wasn't calculated, decide policy (skip or fail)
            val = metrics.get(metric_name, 0.0)
            base_fitness += val * weight

        # Add parsimony pressure
        complexity_penalty = 0.001 * genome.complexity() # alpha is a tuning parameter
        
        return base_fitness - complexity_penalty