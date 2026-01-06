from swarm_rag.evolution.types.genome import Genome

g = Genome(id="test", group_ratios={"g0": 10.0, "g1": 30.0})
print(f"Raw ratios: {{'g0': 10.0, 'g1': 30.0}}")
print(f"Normalized: {g.group_ratios}")

assert abs(sum(g.group_ratios.values()) - 1.0) < 1e-6
assert abs(g.group_ratios['g0'] - 0.25) < 1e-6
print("Normalization works.")
