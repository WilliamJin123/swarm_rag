import math
from typing import List
from swarm_rag.evolution.execution.fitness_strategies import (
    LexicographicStrategy,
    ParetoStrategy,
)
from swarm_rag.evolution.types.genome import Genome
from swarm_rag.evolution.types.fitness_results import FitnessResult

def make_genome(id, quality, stability):
    g = Genome(id=id)
    g.fitness = FitnessResult(
        quality_score=quality,
        stability_score=stability,
    )
    return g

def test_lexicographic_strategy():
    print("\n--- Testing Lexicographic Strategy ---")
    strategy = LexicographicStrategy()

    # G1: High Quality
    # G2: Low Quality
    # Lexicographic prefers Quality
    g1 = make_genome("high_qual", 0.9, 1.0)
    g2 = make_genome("low_qual", 0.5, 1.0)

    pop = [g1, g2]
    strategy.assign_fitness(pop)

    # Sort descending (best first)
    pop.sort(key=lambda g: g.fitness, reverse=True)

    print(f"  Sorted: {[g.id for g in pop]}")
    assert pop[0].id == "high_qual", "Lexicographic failed to prioritize quality"

    # Tie-break on Stability
    g3 = make_genome("stable", 0.9, 1.0)
    g4 = make_genome("unstable", 0.9, 0.1)
    pop2 = [g4, g3]
    strategy.assign_fitness(pop2)
    pop2.sort(key=lambda g: g.fitness, reverse=True)

    print(f"  Tie-break: {[g.id for g in pop2]}")
    assert pop2[0].id == "stable", "Lexicographic failed to tie-break on stability"
    print("  ✓ Lexicographic logic confirmed")

def test_pareto_strategy():
    print("\n--- Testing Pareto Strategy ---")
    strategy = ParetoStrategy()

    # Pareto Front check (Quality, Stability)
    # A: (0.9, 0.5) -> Good quality, lower stability
    # B: (0.5, 0.9) -> Lower quality, high stability
    # C: (0.4, 0.4) -> Dominated by both A and B

    g_a = make_genome("A", 0.9, 0.5)
    g_b = make_genome("B", 0.5, 0.9)
    g_c = make_genome("C", 0.4, 0.4)

    pop = [g_a, g_b, g_c]
    strategy.assign_fitness(pop)

    # Sort descending (Rank 0 > Rank 1, Crowding High > Low)
    # Our strategy sets sort_key = (-rank, crowding)
    pop.sort(key=lambda g: g.fitness, reverse=True)

    print(f"  Sorted: {[g.id for g in pop]}")

    # A and B should be top 2 (order depends on crowding distance)
    # C should be last
    assert pop[-1].id == "C", "Dominated individual C was not ranked last"
    assert "A" in [pop[0].id, pop[1].id]
    assert "B" in [pop[0].id, pop[1].id]

    print("  ✓ Pareto non-dominated sorting confirmed")

if __name__ == "__main__":
    test_lexicographic_strategy()
    test_pareto_strategy()
    print("\nALL FITNESS STRATEGY TESTS PASSED")
