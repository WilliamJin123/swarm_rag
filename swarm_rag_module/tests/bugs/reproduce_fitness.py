from swarm_rag.evolution.types.fitness_results import FitnessResult
import math

f1 = FitnessResult(quality_score=0.1389, stability_score=0.76)
f2 = FitnessResult(quality_score=0.0595, stability_score=0.76)

print(f"F1 (Gen 0): {f1}")
print(f"F2 (Gen 5): {f2}")
print(f"F1 > F2: {f1 > f2}")
print(f"F2 > F1: {f2 > f1}")

print(f"Sort Key F1: {f1._get_sort_key()}")
print(f"Sort Key F2: {f2._get_sort_key()}")
