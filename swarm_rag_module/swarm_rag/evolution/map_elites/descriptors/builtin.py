"""
Built-in behavioral descriptors for MAP-Elites.
"""
from .base import GenotypicDescriptor, PhenotypicDescriptor
from .registry import DescriptorRegistry
from ...types.genome import Genome


@DescriptorRegistry.register
class ComplexityDescriptor(GenotypicDescriptor):
    """Genotypic: Total size of expression trees."""

    @property
    def name(self) -> str:
        return "complexity"

    def calculate(self, genome: Genome) -> float:
        return float(genome.complexity())


@DescriptorRegistry.register
class NAgentsDescriptor(GenotypicDescriptor):
    """Genotypic: Number of agents deployed."""

    @property
    def name(self) -> str:
        return "n_agents"

    def calculate(self, genome: Genome) -> float:
        return float(genome.params.get("n_agents", 0))


@DescriptorRegistry.register
class CostDescriptor(PhenotypicDescriptor):
    """Phenotypic: Cost score from fitness."""

    @property
    def name(self) -> str:
        return "cost"

    def calculate(self, genome: Genome) -> float:
        return genome.fitness.cost_score


@DescriptorRegistry.register
class QualityDescriptor(PhenotypicDescriptor):
    """Phenotypic: Quality score from fitness."""

    @property
    def name(self) -> str:
        return "quality"

    def calculate(self, genome: Genome) -> float:
        return genome.fitness.quality_score


@DescriptorRegistry.register
class RecallDescriptor(PhenotypicDescriptor):
    """Phenotypic: Recall@20 metric."""

    @property
    def name(self) -> str:
        return "recall"

    def calculate(self, genome: Genome) -> float:
        return genome.metrics.get("Recall@20", 0.0)


@DescriptorRegistry.register
class AggressivenessDescriptor(GenotypicDescriptor):
    """Genotypic: Computational load proxy (n_agents * steps)."""

    @property
    def name(self) -> str:
        return "aggressiveness"

    def calculate(self, genome: Genome) -> float:
        n = genome.params.get("n_agents", 1)
        steps = genome.params.get("steps", 1)
        return float(n * steps)


@DescriptorRegistry.register
class LatencyDescriptor(PhenotypicDescriptor):
    """Phenotypic: Execution latency."""

    @property
    def name(self) -> str:
        return "latency"

    def calculate(self, genome: Genome) -> float:
        return genome.latency


@DescriptorRegistry.register
class StepsDescriptor(GenotypicDescriptor):
    """Genotypic: Number of traversal steps."""

    @property
    def name(self) -> str:
        return "steps"

    def calculate(self, genome: Genome) -> float:
        return float(genome.params.get("steps", 0))


@DescriptorRegistry.register
class DecayDescriptor(GenotypicDescriptor):
    """Genotypic: Pheromone decay rate."""

    @property
    def name(self) -> str:
        return "decay"

    def calculate(self, genome: Genome) -> float:
        return genome.params.get("decay", 0.5)
