"""
MAP-Elites Archive with Configurable Comparison Modes

Stores high-performing elites in a structured phenotypic grid with
configurable comparison strategies for determining which genomes replace others.
"""
import ast
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any, TypedDict

from ..types.genome import Genome
from .descriptors import DescriptorCalculator


class MAPStats(TypedDict):
    coverage: float
    filled_cells: int
    qd_score: float
    max_fitness: float


class ArchiveComparisonMode(Enum):
    """Available comparison modes for archive cell replacement."""
    QUALITY_ONLY = "quality_only"           # Current: just quality_score
    WEIGHTED_COMPOSITE = "weighted_composite"  # Configurable weights
    METRIC_THRESHOLD = "metric_threshold"    # Must meet targets first
    LEXICOGRAPHIC = "lexicographic"          # Quality > Stability > Cost


@dataclass
class ArchiveComparatorConfig:
    """
    Configurable archive comparison settings.

    Controls how genomes are compared when competing for archive cells.
    """
    mode: ArchiveComparisonMode = ArchiveComparisonMode.QUALITY_ONLY

    # Weights for WEIGHTED_COMPOSITE mode
    quality_weight: float = 0.8
    stability_weight: float = 0.2

    # Thresholds for METRIC_THRESHOLD mode (must meet these to compete)
    entry_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "Hit@1": 0.10,      # Lower bar for entry (easier than targets)
        "Hit@5": 0.20,
        "MRR": 0.20,
        "Recall@20": 0.25,
    })

    # Require minimum improvement margin for replacement
    min_improvement_margin: float = 0.001


class ArchiveComparator:
    """
    Compares genomes for archive cell replacement.

    Supports multiple comparison strategies configured via ArchiveComparatorConfig.

    Modes:
    - QUALITY_ONLY: Simple quality_score comparison (default)
    - WEIGHTED_COMPOSITE: Weighted combination of quality, stability, and cost
    - METRIC_THRESHOLD: Must meet entry thresholds to qualify, then compare quality
    - LEXICOGRAPHIC: Quality first, stability as tiebreaker, cost as last resort
    """

    def __init__(self, config: Optional[ArchiveComparatorConfig] = None):
        self.config = config or ArchiveComparatorConfig()

    def is_better(
        self,
        challenger: Genome,
        incumbent: Optional[Genome]
    ) -> bool:
        """
        Determine if challenger should replace incumbent in archive cell.

        Args:
            challenger: New genome attempting to enter the cell
            incumbent: Current genome in the cell (None if cell is empty)

        Returns:
            True if challenger should replace incumbent
        """
        # Empty cell - check if challenger meets entry requirements
        if incumbent is None:
            return self._meets_entry_requirements(challenger)

        mode = self.config.mode

        if mode == ArchiveComparisonMode.QUALITY_ONLY:
            return self._compare_quality_only(challenger, incumbent)

        elif mode == ArchiveComparisonMode.WEIGHTED_COMPOSITE:
            return self._compare_weighted_composite(challenger, incumbent)

        elif mode == ArchiveComparisonMode.METRIC_THRESHOLD:
            return self._compare_metric_threshold(challenger, incumbent)

        elif mode == ArchiveComparisonMode.LEXICOGRAPHIC:
            return self._compare_lexicographic(challenger, incumbent)

        # Fallback to quality only
        return self._compare_quality_only(challenger, incumbent)

    def _meets_entry_requirements(self, genome: Genome) -> bool:
        """
        Check if genome meets minimum requirements to enter archive.

        For METRIC_THRESHOLD mode, checks against entry_thresholds.
        For other modes, always returns True (any genome can enter empty cell).
        """
        if self.config.mode != ArchiveComparisonMode.METRIC_THRESHOLD:
            return True

        # Check entry thresholds
        if not hasattr(genome.fitness, 'metrics') or genome.fitness.metrics is None:
            # No metrics available - allow entry (backwards compatibility)
            return True

        metrics = genome.fitness.metrics
        for metric_name, threshold in self.config.entry_thresholds.items():
            if metric_name in metrics and metrics[metric_name] < threshold:
                return False

        return True

    def _compare_quality_only(self, challenger: Genome, incumbent: Genome) -> bool:
        """Simple quality score comparison."""
        margin = self.config.min_improvement_margin
        return challenger.fitness.quality_score > incumbent.fitness.quality_score + margin

    def _compare_weighted_composite(self, challenger: Genome, incumbent: Genome) -> bool:
        """Weighted combination of quality and stability."""
        def score(g: Genome) -> float:
            f = g.fitness
            return (
                self.config.quality_weight * f.quality_score +
                self.config.stability_weight * f.stability_score
            )

        margin = self.config.min_improvement_margin
        return score(challenger) > score(incumbent) + margin

    def _compare_metric_threshold(self, challenger: Genome, incumbent: Genome) -> bool:
        """
        Threshold-based comparison.

        1. If challenger meets thresholds and incumbent doesn't -> challenger wins
        2. If incumbent meets thresholds and challenger doesn't -> incumbent stays
        3. If both meet (or both don't meet) thresholds -> compare quality
        """
        challenger_qualifies = self._meets_entry_requirements(challenger)
        incumbent_qualifies = self._meets_entry_requirements(incumbent)

        if challenger_qualifies and not incumbent_qualifies:
            return True
        if not challenger_qualifies and incumbent_qualifies:
            return False

        # Both qualify (or both don't) - compare quality
        return self._compare_quality_only(challenger, incumbent)

    def _compare_lexicographic(self, challenger: Genome, incumbent: Genome) -> bool:
        """
        Lexicographic comparison: Quality > Stability

        First compares quality. If equal (within margin), compares stability.
        """
        cf = challenger.fitness
        if_ = incumbent.fitness
        margin = self.config.min_improvement_margin

        # Quality comparison
        if cf.quality_score > if_.quality_score + margin:
            return True
        if cf.quality_score < if_.quality_score - margin:
            return False

        # Quality is approximately equal - check stability
        return cf.stability_score > if_.stability_score + margin

    def get_comparison_score(self, genome: Genome) -> float:
        """
        Get a single score for ranking purposes.

        Useful for sorting genomes when mode is WEIGHTED_COMPOSITE.
        """
        f = genome.fitness
        return (
            self.config.quality_weight * f.quality_score +
            self.config.stability_weight * f.stability_score
        )


class MapElitesArchive:
    """
    Stores high-performing elites in a structured phenotypic grid.

    Features:
    - Configurable comparison modes via ArchiveComparator
    - Quality-diversity optimization
    - Serialization for checkpointing
    """

    def __init__(
        self,
        descriptor_calc: DescriptorCalculator,
        bins: List[int],
        ranges: List[Tuple[float, float]],
        comparator_config: Optional[ArchiveComparatorConfig] = None
    ):
        """
        Initialize MAP-Elites archive.

        Args:
            descriptor_calc: Calculator for behavior descriptors
            bins: Number of bins per dimension
            ranges: Value ranges per dimension [(min, max), ...]
            comparator_config: Configuration for archive comparison (optional)
        """
        self.descriptor_calc = descriptor_calc
        self.bins = bins
        self.ranges = ranges

        # Grid: Maps tuple of bin indices -> Genome
        # Example: (3, 5) -> GenomeObject
        self.grid: Dict[Tuple[int, ...], Genome] = {}

        # Keep track of history
        self.history: List[Dict[str, Any]] = []

        # Configurable comparator
        self.comparator = ArchiveComparator(comparator_config)

    def get_bin_index(self, descriptor_values: Tuple[float, ...]) -> Tuple[int, ...]:
        """
        Discretizes continuous descriptor values into grid indices.
        """
        indices = []
        for i, val in enumerate(descriptor_values):
            # Clip value to range
            min_val, max_val = self.ranges[i]
            val = max(min_val, min(val, max_val))

            # Calculate bin
            # Formula: floor( (val - min) / (max - min) * n_bins )
            if max_val == min_val:
                idx = 0
            else:
                norm = (val - min_val) / (max_val - min_val)
                idx = int(norm * self.bins[i])
                # Handle edge case where val == max_val (goes to n_bins, needs to be n_bins-1)
                idx = min(idx, self.bins[i] - 1)

            indices.append(idx)
        return tuple(indices)

    def add(self, genome: Genome) -> bool:
        """
        Attempts to add a genome to the archive.
        Returns True if the genome was added (empty cell) or replaced an inferior one.

        Uses configurable comparison via self.comparator.
        """
        # 1. Calculate Descriptors
        descriptors = self.descriptor_calc.get_descriptor(genome)

        # 2. Determine Bin
        bin_idx = self.get_bin_index(descriptors)

        # 3. Check existing
        current_elite = self.grid.get(bin_idx)

        # 4. Use comparator to decide
        if self.comparator.is_better(genome, current_elite):
            # Store a copy to prevent mutation issues
            self.grid[bin_idx] = genome.copy()
            return True

        return False

    def select_random(self) -> Optional[Genome]:
        """Returns a copy of a random elite from the archive."""
        if not self.grid:
            return None
        key = random.choice(list(self.grid.keys()))
        # Return copy to maintain archive immutability
        return self.grid[key].copy()

    def select_k_random(self, k: int) -> List[Genome]:
        """Returns k random elites with replacement."""
        if not self.grid:
            return []
        keys = list(self.grid.keys())
        selected_keys = random.choices(keys, k=k)
        return [self.grid[key] for key in selected_keys]

    def select_best(self, k: int = 1) -> List[Genome]:
        """
        Returns the top k genomes by comparison score.

        Uses the comparator's scoring method for ranking.
        """
        if not self.grid:
            return []

        ranked = sorted(
            self.grid.values(),
            key=lambda g: self.comparator.get_comparison_score(g),
            reverse=True
        )
        return ranked[:k]

    def as_population(self) -> List[Genome]:
        """Returns all elites as a list."""
        return list(self.grid.values())

    def stats(self) -> MAPStats:
        """Returns coverage metrics."""
        total_cells = math.prod(self.bins)
        filled_cells = len(self.grid)
        coverage = filled_cells / total_cells if total_cells > 0 else 0.0

        sum_fitness = sum(g.fitness.quality_score for g in self.grid.values())
        max_fitness = max(g.fitness.quality_score for g in self.grid.values()) if self.grid else 0.0

        return {
            "coverage": coverage,
            "filled_cells": filled_cells,
            "qd_score": sum_fitness,
            "max_fitness": max_fitness
        }

    def clear(self):
        self.grid.clear()

    def set_comparator_mode(self, mode: ArchiveComparisonMode):
        """Change the comparison mode at runtime."""
        self.comparator.config.mode = mode

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize archive state for checkpointing.

        Returns:
            Dictionary containing archive configuration and grid contents
        """
        grid_data = {}
        for key, genome in self.grid.items():
            # Store key as string for JSON compatibility
            key_str = str(key)
            grid_data[key_str] = {
                "genome": genome.to_dict() if hasattr(genome, 'to_dict') else None,
                "id": genome.id,
                "fitness_quality": genome.fitness.quality_score
            }

        return {
            "bins": self.bins,
            "ranges": self.ranges,
            "dimensions": getattr(self.descriptor_calc, 'dimensions', []),
            "grid": grid_data,
            "filled_cells": len(self.grid),
            "comparator_mode": self.comparator.config.mode.value
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        descriptor_calc: 'DescriptorCalculator',
        comparator_config: Optional[ArchiveComparatorConfig] = None
    ) -> 'MapElitesArchive':
        """
        Restore archive from checkpoint data.

        Args:
            data: Dictionary from to_dict()
            descriptor_calc: DescriptorCalculator for the archive
            comparator_config: Optional comparator configuration

        Returns:
            Restored MapElitesArchive instance
        """
        # Restore comparator mode from checkpoint if available
        if comparator_config is None and "comparator_mode" in data:
            comparator_config = ArchiveComparatorConfig(
                mode=ArchiveComparisonMode(data["comparator_mode"])
            )

        archive = cls(
            descriptor_calc=descriptor_calc,
            bins=data["bins"],
            ranges=data["ranges"],
            comparator_config=comparator_config
        )

        # Restore grid if genome data is available
        if "grid" in data:
            from ..types.genome import Genome
            for key_str, genome_data in data["grid"].items():
                if genome_data.get("genome"):
                    try:
                        # Parse tuple key from string safely (no code execution)
                        key = ast.literal_eval(key_str)
                        genome = Genome.from_dict(genome_data["genome"])
                        archive.grid[key] = genome
                    except (ValueError, SyntaxError):
                        pass  # Skip malformed entries

        return archive
