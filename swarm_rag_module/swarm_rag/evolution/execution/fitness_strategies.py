from abc import ABC, abstractmethod
from typing import List
import torch

from ..types.genome import Genome

class FitnessStrategy(ABC):
    """
    Abstract base class for population ranking strategies.
    Responsible for assigning a comparable `sort_key` to each genome's fitness.
    """
    @abstractmethod
    def assign_fitness(self, population: List[Genome], generation: int = 0) -> None:
        """
        Calculates and assigns the sort_key for every genome in the population.
        """
        pass

class LexicographicStrategy(FitnessStrategy):
    """
    Standard sorting: Quality > Stability > Cost.
    """
    def assign_fitness(self, population: List[Genome], generation: int = 0) -> None:
        for genome in population:
            genome.fitness.update_sort_key(mode="lexicographic")

class ParetoStrategy(FitnessStrategy):
    """
    NSGA-II inspired Non-Dominated Sorting.
    """
    def assign_fitness(self, population: List[Genome], generation: int = 0) -> None:
        # 1. Extract objectives (all maximization)
        # Objectives: Quality (max), Stability (max)

        pop_size = len(population)
        if pop_size == 0: return

        # Shape: (N, 2) -> [Quality, Stability]
        objectives = torch.zeros((pop_size, 2))
        for i, g in enumerate(population):
            objectives[i] = torch.as_tensor([
                g.fitness.quality_score,
                g.fitness.stability_score,
            ])

        # 2. Non-Dominated Sort
        fronts = self._fast_non_dominated_sort(objectives)

        # 3. Crowding Distance (per front)
        crowding_distances = torch.zeros(pop_size)

        for front in fronts:
            self._calculate_crowding_distance(objectives, front, crowding_distances)
            
        # 4. Assign Sort Keys
        # Key: (Rank (asc), Crowding Distance (desc))
        # We want Rank 0 to be best. Python sorts ascending by default?
        # Genome.fitness compares using <. 
        # If g1 < g2, g1 comes first? No, reverse=True usually means bigger is better.
        # Let's align with Lexicographic: Bigger tuple = Better.
        # Lexicographic: (0.9, ...) > (0.8, ...)
        
        # Pareto: Rank 0 is best. So we want Rank 0 > Rank 1.
        # To make "Bigger is Better", we can use negative Rank.
        # (-0, Crowding) > (-1, Crowding)
        
        for i, g in enumerate(population):
            # Find which front 'i' belongs to
            rank = -1
            for r, front in enumerate(fronts):
                if i in front:
                    rank = r
                    break
            
            cd = crowding_distances[i]
            # Primary: Lower Rank is better (so -rank is higher)
            # Secondary: Higher Crowding Distance is better (more diversity)
            g.fitness.sort_key = (-rank, cd) 

    def _fast_non_dominated_sort(self, objectives: torch.Tensor) -> List[List[int]]:
        """
        Returns a list of fronts, where fronts[0] is the Pareto front.
        objectives: (N, M) tensor, all maximization.
        """
        n = objectives.shape[0]
        domination_count = torch.zeros(n, dtype=torch.long)
        dominated_solutions = [[] for _ in range(n)]
        ranks = torch.zeros(n, dtype=torch.long)

        fronts = [[]]

        for p in range(n):
            for q in range(n):
                if self._dominates(objectives[p], objectives[q]):
                    dominated_solutions[p].append(q)
                elif self._dominates(objectives[q], objectives[p]):
                    domination_count[p] += 1

            if domination_count[p] == 0:
                ranks[p] = 0
                fronts[0].append(p)

        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        ranks[q] = i + 1
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    def _dominates(self, ind1: torch.Tensor, ind2: torch.Tensor) -> bool:
        """Returns True if ind1 dominates ind2 (Maximization)."""
        # Dominate: At least one objective better, none worse
        better_or_equal = ind1 >= ind2
        strictly_better = ind1 > ind2
        return bool(torch.all(better_or_equal)) and bool(torch.any(strictly_better))

    def _calculate_crowding_distance(self, objectives: torch.Tensor, front: List[int], distances: torch.Tensor):
        """Updates distances tensor in-place for indices in front."""
        if not front: return

        l = len(front)
        # Infinite distance to boundaries
        for i in front:
            distances[i] = 0

        m = objectives.shape[1] # number of objectives

        for m_idx in range(m):
            # Sort front by objective m
            sorted_front = sorted(front, key=lambda x: objectives[x, m_idx].item())

            distances[sorted_front[0]] = float('inf')
            distances[sorted_front[-1]] = float('inf')

            obj_min = objectives[sorted_front[0], m_idx]
            obj_max = objectives[sorted_front[-1], m_idx]

            if obj_max == obj_min: continue

            norm = obj_max - obj_min

            for i in range(1, l - 1):
                prev_obj = objectives[sorted_front[i-1], m_idx]
                next_obj = objectives[sorted_front[i+1], m_idx]
                distances[sorted_front[i]] += (next_obj - prev_obj) / norm

