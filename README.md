# swarm_rag

Experimenting with swarm RAG

# 12-14

This dude [Richard](https://www.youtube.com/@richardaragon8471) is claiming that swarm RAG is just better than KG RAG straight up.

Since I've intuition-pumped that symbolic knowledge-graphs is how humans think / represent "knowledge", I naturally turned to KG RAG. These experiments serve to prove to myself the validity of swarmRAG and application of biological algorithms to AI in general.


# 12 - 22

Sprint to finish V1 today.

Try and implement GraphTransformers + Kimi Attention Optimizations and / or CLaRA and / or ZRIA (GNN for the unstructured knowledge graph) within the next week.


# 12 - 28

Update, have the main scaffolding built out. We are not doing GNN bullshit and Graph Attention with Kimi Delta Attention (thats ill)

What we will be doing though is spamming genetic algorithms to optimize hyperparameters for different memory constraints (probably just unconstrained for now) and heuristic linear combinations (symbolic regression)

Hopefully we cooking with this over the next few days and we can benchmarkmaxx on Stark

# 01-03

Learning about genetic algorithm nuances
- We are doing (μ + λ): elitist; parents and offspring compete (best case for our proj)
- Need to decide selection strategies (tournament vs SUS vs Top-k trunc vs Roulette vs Rank-based, etc.) --> Multiple impleemnted in strategies.py
- Crossover: Subtree Crossover for expressions
- Mutation: ought to use "decaying" mutation rate or adaptive based on population "health" signals
- Fitness function strategies: ought to use pareto optimization, EX: Maximize Hit@k, Minimize traversal cost, Minimize variance across queries
- Ought to normalize fitness, fitness sharing (penalize super similar "genomes")
- Reward Novelty?

- Techniques: Niching (cluster similar heuristics, different ones coexist), Island models (occasional migration), random immigrants (inject fresh heuristics periodically)

NOTE: Switching to CMA-ES for hyperparameters in comb with Symbolic Regression for heuristic functions