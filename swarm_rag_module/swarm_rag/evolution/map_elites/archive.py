import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, TypedDict
from ..types.genome import Genome
from .descriptors import DescriptorCalculator

class MAPStats(TypedDict):
    coverage: float
    filled_cells: int
    qd_score: float
    max_fitness: float

class MapElitesArchive:
    """
    Stores high-performing elites in a structured phenotypic grid.
    """
    def __init__(
        self, 
        descriptor_calc: DescriptorCalculator,
        bins: List[int],
        ranges: List[Tuple[float, float]]
    ):
        self.descriptor_calc = descriptor_calc
        self.bins = bins
        self.ranges = ranges
        
        # Grid: Maps tuple of bin indices -> Genome
        # Example: (3, 5) -> GenomeObject
        self.grid: Dict[Tuple[int, ...], Genome] = {}
        
        # Keep track of history
        self.history: List[Dict[str, Any]] = []

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
        """
        # 1. Calculate Descriptors
        descriptors = self.descriptor_calc.get_descriptor(genome)
        
        # 2. Determine Bin
        bin_idx = self.get_bin_index(descriptors)
        
        # 3. Check existing
        current_elite = self.grid.get(bin_idx)
        
        # 4. Compare Fitness (using quality_score as primary metric)
        # Note: We assume lexicographic comparison of FitnessResult works, 
        # but here we specifically want to maximize the primary objective usually.
        # Let's rely on the > operator of FitnessResult if possible, or quality_score.
        
        should_replace = False
        if current_elite is None:
            should_replace = True
        else:
            # We use the __gt__ of Genome/FitnessResult if available,
            # or fallback to quality score.
            if genome.fitness.quality_score > current_elite.fitness.quality_score:
                should_replace = True
        
        if should_replace:
            # Store a copy to prevent mutation issues
            self.grid[bin_idx] = genome.copy()
            return True
            
        return False

    def select_random(self) -> Optional[Genome]:
        """Returns a random elite from the archive."""
        if not self.grid:
            return None
        # Convert keys to list is O(N), but acceptable for typical archive sizes (~100-1000)
        # For very large archives, might want to cache the keys list.
        key = random.choice(list(self.grid.keys()))
        return self.grid[key]

    def select_k_random(self, k: int) -> List[Genome]:
        """Returns k random elites with replacement."""
        if not self.grid:
            return []
        keys = list(self.grid.keys())
        selected_keys = random.choices(keys, k=k)
        return [self.grid[key] for key in selected_keys]

    def as_population(self) -> List[Genome]:
        """Returns all elites as a list."""
        return list(self.grid.values())

    def stats(self) -> MAPStats:
        """Returns coverage metrics."""
        total_cells = np.prod(self.bins)
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
