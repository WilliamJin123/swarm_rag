from typing import List, Tuple, Dict, Any, Callable
from ..types.genome import Genome
import numpy as np

class DescriptorCalculator:
    """
    Calculates behavioral descriptors for a Genome.
    Used to place Genomes into the MAP-Elites archive.
    """
    def __init__(self, dimensions: List[str], ranges: List[Tuple[float, float]]):
        self.dimensions = dimensions
        self.ranges = ranges
        self.registry: Dict[str, Callable[[Genome], float]] = {
            "complexity": self._calc_complexity,
            "n_agents": self._calc_n_agents,
            "cost": self._calc_cost,
            "aggressiveness": self._calc_aggressiveness,
            "latency": self._calc_latency,
            "recall": self._calc_recall,
            "quality": self._calc_quality
        }

    def get_descriptor(self, genome: Genome) -> Tuple[float, ...]:
        """
        Returns a tuple of descriptor values corresponding to configured dimensions.
        Example: (25.0, 0.85) for ["complexity", "cost"]
        """
        values = []
        for dim in self.dimensions:
            if dim not in self.registry:
                raise ValueError(f"Unknown MAP-Elites dimension: '{dim}'. Available: {list(self.registry.keys())}")
            
            val = self.registry[dim](genome)
            values.append(val)
        return tuple(values)

    def _calc_complexity(self, genome: Genome) -> float:
        """Genotypic: Total size of expression trees."""
        return float(genome.complexity())

    def _calc_n_agents(self, genome: Genome) -> float:
        """Genotypic: Number of agents deployed."""
        return float(genome.params.get("n_agents", 0))

    def _calc_cost(self, genome: Genome) -> float:
        """Phenotypic: Normalized cost score from fitness."""
        # Note: Cost score is usually inverted (higher is better) in FitnessResult.
        # Here we might want raw cost if available, but for now we use the score.
        return genome.fitness.cost_score

    def _calc_aggressiveness(self, genome: Genome) -> float:
        """Genotypic: Proxy for computational load/aggressiveness."""
        n = genome.params.get("n_agents", 1)
        steps = genome.params.get("steps", 1)
        return float(n * steps)

    def _calc_latency(self, genome: Genome) -> float:
        """Phenotypic: Execution latency."""
        return genome.latency

    def _calc_recall(self, genome: Genome) -> float:
        """Phenotypic: Recall@20."""
        return genome.metrics.get("Recall@20", 0.0)
        
    def _calc_quality(self, genome: Genome) -> float:
        """Phenotypic: Quality Score."""
        return genome.fitness.quality_score
