# Architecture Improvement Brainstorm

## Goal Reminder
Target metrics across stark prime, amazon, and mag graphs:
- **Hit@1 > 50%**
- **Hit@5 > 75%**
- **Recall@20 > 80%**
- **MRR > 75%**

---

## Part 1: Hyperparameter Search Space Refinement

### 1.1 Streamline Evolved Hyperparameters

**Current:** 6 hyperparameters with wide ranges
```python
n_agents: 5-100
steps: 1-10
decay: 0.1-0.9
drop_zone_inc: 0.01-0.2
initial_pool_size: 10-100
start_subset: 3-30
```

**Refinement:**

#### A) Remove Low-Impact Parameters from Evolution
Based on typical swarm behavior, some parameters have weak effects:
- **Fix drop_zone_inc = 0.05** (rarely impacts results significantly)
- **Fix start_subset = 10** (10 starting nodes is usually sufficient)

This leaves 4 key parameters to evolve: `n_agents`, `steps`, `decay`, `initial_pool_size`

#### B) Tighten Ranges Based on Observations
```python
n_agents: 15-50         # Fewer agents = faster, too many = redundant
steps: 3-7              # Most signal captured in 3-6 steps
decay: 0.3-0.8          # Keep dynamic, but tighten range
initial_pool_size: 20-60  # Tight range around optimal
```

**Benefits:**
- Smaller search space = faster convergence
- Dynamic decay preserved for exploration/exploitation balance
- Expression trees remain intact for discovering complex heuristics

---

## Part 2: Evolution Process Speedups

### 2.1 Parallel LLM Mutations (Already Planned)
As documented in OPTIMIZATION_PLAN.md, parallelize the mutation loop:
- Expected speedup: **3-4x**
- Implementation: ThreadPoolExecutor with 4 workers

### 2.2 More Aggressive Early Stopping

**Current tiers:** 5q @ 0.10, 15q @ 0.25, 40q @ 0.50

**Proposed tiers (higher thresholds, fewer queries):**
```python
tiers = [
    (3, 0.15),   # 3 queries: filter completely broken
    (8, 0.30),   # 8 queries: filter poor performers
    (20, 0.45),  # 20 queries: filter mediocre
    (ALL, None)  # Full eval only for promising
]
```

**Rationale:**
- If a genome can't hit 0.15 on 3 random queries, it's broken
- If it can't hit 0.30 on 8 queries, it won't reach 0.65 target
- Only ~20% of genomes should reach full evaluation

### 2.3 Cached Graph Operations

**Current bottleneck:** CPU graph traversal (125+ calls per query)

**Optimization:** Pre-compute and cache:
1. **Neighbor lists** as tensors for all nodes (one-time cost)
2. **Degree centrality** for all nodes (one-time cost)
3. **2-hop neighborhoods** for frequently visited nodes

```python
# Pre-compute on init:
self._neighbor_tensor = self._build_neighbor_tensor()  # Padded tensor
self._degree_tensor = torch.tensor([g.degree(n) for n in range(n_nodes)])
```

### 2.4 Smaller Population, More Generations

**Current:** Large batches of offspring per generation

**Proposal:**
- Reduce batch_size from 15-20 to 8-10
- Run 2x more generations
- **Benefit:** Faster feedback loop, more exploitation of good discoveries

---

## Part 3: Algorithm Improvements for Faster Convergence

### 3.1 Warm-Start from Known Good Configurations

**Problem:** Evolution starts from random genomes, wasting cycles discovering basics.

**Solution:** Seed initial population with known good configurations:
```python
SEED_GENOMES = [
    # High-semantic config with balanced exploration
    {
        "n_agents": 25,
        "steps": 4,
        "decay": 0.5,
        "initial_pool_size": 30,
        "movement_tree": "ADD(MUL(semantic_similarity, 0.7), MUL(node_centrality, 0.3))",
        "deposit_tree": "semantic",
    },
    # Hub-explorer config - emphasizes graph structure
    {
        "n_agents": 30,
        "steps": 5,
        "decay": 0.6,
        "initial_pool_size": 40,
        "movement_tree": "ADD(MUL(node_centrality, 0.5), MUL(semantic_similarity, 0.3), MUL(pheromone_repulsion, 0.2))",
        "deposit_tree": "hub",
    },
    # Diversity-focused config - avoids clustering
    {
        "n_agents": 20,
        "steps": 4,
        "decay": 0.4,
        "initial_pool_size": 35,
        "movement_tree": "ADD(MUL(pheromone_repulsion, 0.4), MUL(semantic_similarity, 0.6))",
        "deposit_tree": "exploration_bonus",
    },
]
```

### 3.2 Gradient-Based Hyperparameter Tuning

**Problem:** Random mutation of continuous hyperparameters is inefficient.

**Solution:** For numeric hyperparameters, use local gradient estimation:
```python
def mutate_hyperparams_with_gradient(genome, context, quick_eval_fn):
    """
    Use finite difference gradient estimation to guide hyperparameter mutations.
    Only applies to continuous params: n_agents, steps, decay, initial_pool_size
    """
    learning_rate = 0.1

    for param_name in ["decay", "n_agents", "steps", "initial_pool_size"]:
        current_val = genome.params[param_name]

        # Scale delta based on parameter range
        if param_name == "decay":
            delta = 0.05
        elif param_name in ["n_agents", "steps"]:
            delta = max(1, int(current_val * 0.1))
        else:
            delta = max(2, int(current_val * 0.1))

        # Evaluate +/- perturbations (using quick 3-query eval)
        genome_plus = genome.copy_with_param(param_name, current_val + delta)
        genome_minus = genome.copy_with_param(param_name, current_val - delta)

        fitness_plus = quick_eval_fn(genome_plus)
        fitness_minus = quick_eval_fn(genome_minus)

        # Estimate gradient direction
        if fitness_plus > fitness_minus:
            new_val = current_val + delta * learning_rate
        elif fitness_minus > fitness_plus:
            new_val = current_val - delta * learning_rate
        else:
            continue  # No clear gradient, skip

        # Clamp to valid range
        genome.params[param_name] = clamp_to_range(param_name, new_val)

    return genome
```

**Trade-off:** Requires 2 extra quick evaluations per parameter, but provides directed search instead of random walk.

### 3.3 Focused Mutation (Metric-Aware)

**Current:** Mutations randomly perturb all genome components.

**Proposal:** Focus mutations on components most likely to improve the weakest metric:
```python
def focused_mutate(genome, metrics, context):
    """
    Analyze which metric is weakest and target mutations accordingly.
    """
    # Identify worst metric relative to target
    metric_gaps = {
        "Hit@1": 0.50 - metrics.get("Hit@1", 0),
        "Hit@5": 0.75 - metrics.get("Hit@5", 0),
        "Recall@20": 0.80 - metrics.get("Recall@20", 0),
        "MRR": 0.75 - metrics.get("MRR", 0),
    }
    worst_metric = max(metric_gaps, key=metric_gaps.get)

    if worst_metric == "Hit@1":
        # Hit@1 is low -> need better precision at top
        # Focus on ranking strategy mutations
        mutate_component(genome, "ranking", intensity="high")
        mutate_component(genome, "movement", intensity="low")

    elif worst_metric == "Hit@5":
        # Hit@5 low -> need better early ranking
        # Balance between ranking and movement
        mutate_component(genome, "ranking", intensity="medium")
        mutate_component(genome, "movement", intensity="medium")

    elif worst_metric == "Recall@20":
        # Recall@20 is low -> need better coverage/exploration
        # Focus on movement and increase exploration parameters
        mutate_component(genome, "movement", intensity="high")
        mutate_component(genome, "deposit", intensity="medium")
        # Bias toward increasing n_agents or steps
        if random.random() < 0.5:
            genome.params["n_agents"] = int(genome.params["n_agents"] * 1.1)
        else:
            genome.params["steps"] = min(genome.params["steps"] + 1, 7)

    elif worst_metric == "MRR":
        # MRR low -> first hit is too deep in results
        # Need both better movement (find good nodes) and ranking (surface them)
        mutate_component(genome, "movement", intensity="medium")
        mutate_component(genome, "ranking", intensity="medium")

    return genome
```

### 3.4 Elitism with Diversity Preservation

**Current:** MAP-Elites preserves diversity via behavior grid.

**Enhancement:** Add explicit diversity pressure during parent selection:
```python
def select_parent_with_diversity(archive, recent_offspring):
    """
    Select parent that balances fitness with behavioral diversity.
    Prevents evolution from collapsing to a single region of behavior space.
    """
    candidates = archive.top_k(20)  # Top 20 by fitness

    # Compute diversity score: distance to nearest recent offspring
    for candidate in candidates:
        min_distance = float('inf')
        for offspring in recent_offspring:
            dist = behavioral_distance(candidate, offspring)
            min_distance = min(min_distance, dist)
        candidate._diversity_score = min_distance

    # Combined selection score
    for candidate in candidates:
        # Normalize fitness to [0, 1]
        norm_fitness = (candidate.fitness.quality_score - 0.2) / 0.5
        # Diversity bonus (capped at 0.3)
        diversity_bonus = min(0.3, candidate._diversity_score * 0.1)
        candidate._selection_score = norm_fitness + diversity_bonus

    # Weighted sampling
    weights = [max(0.01, c._selection_score) for c in candidates]
    return random.choices(candidates, weights=weights, k=1)[0]

def behavioral_distance(g1, g2):
    """Distance in behavior space (n_agents, complexity)."""
    return math.sqrt(
        ((g1.params["n_agents"] - g2.params["n_agents"]) / 50) ** 2 +
        ((g1.complexity() - g2.complexity()) / 30) ** 2
    )
```

---

## Part 4: Retrieval Algorithm Improvements

### 4.1 Adaptive Step Count

**Current:** Fixed number of steps per query.

**Improvement:** Stop early if agents have converged (saves computation on easy queries):
```python
def should_continue_stepping(agent_positions, prev_positions, step_idx, min_steps=2):
    """
    Check if agents have converged and further steps are unlikely to help.
    Always run at least min_steps to allow initial exploration.
    """
    if step_idx < min_steps:
        return True

    # Count agents that haven't moved
    same_position = sum(
        1 for curr, prev in zip(agent_positions, prev_positions)
        if curr == prev
    )
    convergence_ratio = same_position / len(agent_positions)

    # If 80%+ of agents are stuck, stop early
    if convergence_ratio >= 0.8:
        return False

    return True

# In retrieval loop:
for step in range(max_steps):
    prev_positions = agent_positions.copy()
    agent_positions = step_agents(...)

    if not should_continue_stepping(agent_positions, prev_positions, step):
        logger.debug(f"Early stop at step {step+1}/{max_steps} - agents converged")
        break
```

**Benefits:**
- Faster evaluation on "easy" queries where answer is obvious
- No penalty on hard queries that need full exploration
- Can be evolved as a genome parameter (convergence_threshold)

### 4.2 Query-Adaptive Agent Count

**Current:** Same n_agents for all queries regardless of difficulty.

**Improvement:** Allocate more agents to harder queries:
```python
def get_adaptive_n_agents(query_vec, initial_pool, base_n_agents):
    """
    Estimate query difficulty and adjust agent count accordingly.
    Hard queries (weak initial pool) get more agents.
    """
    # Compute similarity spread in initial pool
    similarities = [cosine_similarity(query_vec, node.embedding) for node in initial_pool]

    avg_sim = sum(similarities) / len(similarities)
    max_sim = max(similarities)
    spread = max_sim - avg_sim

    # Difficulty heuristics:
    # - Low avg_sim: Query is semantically distant from corpus
    # - Low spread: No clear winners in initial pool
    # - Low max_sim: Best match is still weak

    difficulty_score = 0.0

    if avg_sim < 0.4:
        difficulty_score += 0.3
    if max_sim < 0.6:
        difficulty_score += 0.3
    if spread < 0.1:
        difficulty_score += 0.2

    # Scale agents: 1.0x for easy, up to 1.5x for hard
    multiplier = 1.0 + (difficulty_score * 0.5)

    return int(base_n_agents * multiplier)
```

**Alternative: Evolved Difficulty Thresholds**
```python
# Add to genome params:
genome.params["easy_query_threshold"] = 0.7   # avg_sim above this = easy
genome.params["hard_query_multiplier"] = 1.3  # agent multiplier for hard queries
```

---

## Part 5: Alternative Optimization Strategies

### 5.1 Bayesian Optimization for Hyperparameters

**Observation:** With 4 continuous hyperparameters (n_agents, steps, decay, initial_pool_size), Bayesian optimization may converge faster than evolutionary search for the numeric portion.

**Hybrid Approach:** Use Bayesian optimization for hyperparameters, evolution for expression trees:

```python
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

class HybridOptimizer:
    """
    Combines Bayesian optimization (for hyperparameters)
    with MAP-Elites (for expression trees).
    """

    def __init__(self, archive, evaluator):
        self.archive = archive
        self.evaluator = evaluator

        # Bayesian model for hyperparameters
        self.hyperparam_bounds = torch.tensor([
            [15, 50],    # n_agents
            [3, 7],      # steps
            [0.3, 0.8],  # decay
            [20, 60],    # initial_pool_size
        ]).T

        self.observed_X = []  # Hyperparameter configurations
        self.observed_Y = []  # Fitness scores

    def suggest_hyperparams(self):
        """Use Bayesian optimization to suggest promising hyperparameters."""
        if len(self.observed_X) < 5:
            # Not enough data yet - random sampling
            return self._random_hyperparams()

        # Fit GP model
        train_X = torch.tensor(self.observed_X)
        train_Y = torch.tensor(self.observed_Y).unsqueeze(-1)
        model = SingleTaskGP(train_X, train_Y)

        # Optimize acquisition function
        EI = ExpectedImprovement(model, best_f=max(self.observed_Y))
        candidate, _ = optimize_acqf(
            EI, bounds=self.hyperparam_bounds, q=1, num_restarts=5, raw_samples=20
        )

        return {
            "n_agents": int(candidate[0, 0].item()),
            "steps": int(candidate[0, 1].item()),
            "decay": candidate[0, 2].item(),
            "initial_pool_size": int(candidate[0, 3].item()),
        }

    def step(self):
        """One optimization step: BO for hyperparams, mutation for trees."""
        # Get parent from archive (for expression trees)
        parent = self.archive.select_random()

        # Suggest hyperparameters via BO
        suggested_params = self.suggest_hyperparams()

        # Create child with BO-suggested params + mutated trees from parent
        child = parent.copy()
        child.params.update(suggested_params)
        child = mutate_expression_trees(child)  # Keep tree evolution

        # Evaluate
        fitness = self.evaluator.evaluate_single(child)

        # Update BO model
        self.observed_X.append(list(suggested_params.values()))
        self.observed_Y.append(fitness.quality_score)

        # Update archive
        self.archive.try_add(child)

        return child
```

**When to use:**
- After initial random exploration phase (generation 10+)
- When hyperparameter sensitivity is high
- Can run alongside standard evolution as an alternative mutation source

---

## Part 6: Infrastructure Improvements

### 6.1 GPU Graph Store

Moving graph operations to GPU for massive speedup on neighbor lookups:
```python
class GPUGraphStore:
    """
    Graph store with adjacency matrix on GPU for fast neighbor lookups.
    """
    def __init__(self, adjacency_matrix, device="cuda"):
        self.device = device

        # Convert to sparse CUDA tensor
        if hasattr(adjacency_matrix, 'tocoo'):
            # From scipy sparse
            coo = adjacency_matrix.tocoo()
            indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
            values = torch.tensor(coo.data, dtype=torch.float32)
            self.adj = torch.sparse_coo_tensor(
                indices, values, coo.shape, device=device
            )
        else:
            self.adj = adjacency_matrix.to_sparse().to(device)

        # Pre-compute degrees
        self.degrees = torch.sparse.sum(self.adj, dim=1).to_dense()
        self.avg_degree = self.degrees.float().mean().item()

        # Pre-compute neighbor lists (padded tensor for batch access)
        self._neighbor_tensor, self._neighbor_counts = self._build_neighbor_tensor()

    def _build_neighbor_tensor(self, max_neighbors=100):
        """Build padded tensor of neighbor IDs for batch access."""
        n_nodes = self.adj.shape[0]
        neighbor_tensor = torch.full(
            (n_nodes, max_neighbors), -1, dtype=torch.long, device=self.device
        )
        neighbor_counts = torch.zeros(n_nodes, dtype=torch.long, device=self.device)

        # Extract neighbors from sparse adjacency
        adj_coo = self.adj.coalesce()
        rows, cols = adj_coo.indices()

        for i in range(n_nodes):
            mask = rows == i
            neighbors = cols[mask]
            n_neighbors = min(len(neighbors), max_neighbors)
            neighbor_tensor[i, :n_neighbors] = neighbors[:n_neighbors]
            neighbor_counts[i] = n_neighbors

        return neighbor_tensor, neighbor_counts

    def get_neighbors_batch(self, node_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get neighbors for multiple nodes in single GPU operation.

        Returns:
            neighbor_ids: (batch, max_neighbors) padded with -1
            neighbor_counts: (batch,) actual neighbor count per node
        """
        return self._neighbor_tensor[node_ids], self._neighbor_counts[node_ids]

    def get_degrees_batch(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Get degrees for multiple nodes."""
        return self.degrees[node_ids]
```

### 6.2 Vectorized Agent Steps

Batch all agent movements in one tensor operation instead of sequential processing:
```python
def step_all_agents_vectorized(
    positions: torch.Tensor,           # (n_agents,) current positions
    neighbor_tensor: torch.Tensor,     # (n_nodes, max_neighbors) pre-computed
    neighbor_counts: torch.Tensor,     # (n_nodes,) neighbor counts
    query_vec: torch.Tensor,           # (embed_dim,)
    embeddings: torch.Tensor,          # (n_nodes, embed_dim)
    pheromones: torch.Tensor,          # (n_nodes,)
    degrees: torch.Tensor,             # (n_nodes,)
    movement_fn: Callable,             # Compiled movement heuristic
    device: str = "cuda"
) -> torch.Tensor:
    """
    Step all agents in parallel using vectorized operations.

    Returns:
        new_positions: (n_agents,) new position for each agent
    """
    n_agents = positions.shape[0]
    max_neighbors = neighbor_tensor.shape[1]

    # Get neighbors for all current positions: (n_agents, max_neighbors)
    agent_neighbors = neighbor_tensor[positions]
    agent_neighbor_counts = neighbor_counts[positions]

    # Create mask for valid neighbors: (n_agents, max_neighbors)
    valid_mask = torch.arange(max_neighbors, device=device).unsqueeze(0) < agent_neighbor_counts.unsqueeze(1)

    # Get embeddings for all neighbors: (n_agents, max_neighbors, embed_dim)
    # Use -1 padding -> index 0 (will be masked out anyway)
    safe_neighbors = agent_neighbors.clamp(min=0)
    neighbor_embeddings = embeddings[safe_neighbors]

    # Compute semantic similarity: (n_agents, max_neighbors)
    semantic_scores = torch.matmul(neighbor_embeddings, query_vec)

    # Get pheromones and degrees for neighbors
    neighbor_pheromones = pheromones[safe_neighbors]
    neighbor_degrees = degrees[safe_neighbors]

    # Build context and compute movement scores
    # (This would call the compiled expression tree on batched data)
    movement_scores = movement_fn(
        semantic_similarity=semantic_scores,
        node_centrality=torch.log1p(neighbor_degrees) / (torch.log1p(neighbor_degrees) + 1),
        pheromone_repulsion=1.0 - neighbor_pheromones / (pheromones.max() + 1e-6),
    )

    # Mask invalid neighbors with -inf for softmax
    movement_scores = movement_scores.masked_fill(~valid_mask, float('-inf'))

    # Probabilistic selection via softmax + multinomial
    probs = F.softmax(movement_scores, dim=1)
    selected_idx = torch.multinomial(probs, 1).squeeze(1)  # (n_agents,)

    # Gather selected neighbor IDs
    new_positions = agent_neighbors[torch.arange(n_agents, device=device), selected_idx]

    return new_positions
```

**Expected speedup:** 5-10x for agent stepping phase

### 6.3 Query Embedding Batching

Batch embed all queries at start of generation (partially implemented in SharedPrecomputeContext):
```python
class EnhancedSharedPrecompute:
    """
    Extended shared pre-computation with additional optimizations.
    """
    def __init__(self, retriever, queries, ground_truth, device="cuda"):
        self.device = device

        # Batch embed ALL queries upfront
        logger.info(f"Pre-computing embeddings for {len(queries)} queries...")
        self.query_embeddings = retriever.embed_fn.embed_batch(queries)
        if isinstance(self.query_embeddings, list):
            self.query_embeddings = torch.stack(self.query_embeddings)
        self.query_embeddings = self.query_embeddings.to(device)

        # Pre-compute ground truth as sets (for fast metric computation)
        self.ground_truth_sets = [set(gt) for gt in ground_truth]

        # Pre-compute initial pools for common pool sizes
        self.initial_pools = {}
        for pool_size in [20, 30, 40, 50]:
            logger.info(f"Pre-computing initial pools for pool_size={pool_size}...")
            pools = []
            for i, q_emb in enumerate(self.query_embeddings):
                pool = retriever.vector_store.search(q_emb, k=pool_size)
                pools.append(pool)
            self.initial_pools[pool_size] = pools

        logger.info("Shared pre-computation complete")
```

---

## Priority Ranking

### Highest Impact (Implement First)
1. **Parallel LLM mutations** - 3-4x speedup, already planned
2. **Aggressive early stopping tiers** - 2-3x speedup for bad genomes
3. **Warm-start population** - Fewer wasted generations discovering basics
4. **Focused mutation (metric-aware)** - More directed search

### Medium Impact
5. GPU graph store with pre-computed neighbor tensors
6. Vectorized agent stepping
7. Gradient-based hyperparameter tuning
8. Smaller population, more generations
9. Adaptive step count (early convergence detection)

### Experimental / Long-Term
10. Bayesian optimization hybrid for hyperparameters
11. Query-adaptive agent count
12. 2-hop neighborhood pre-computation
13. Elitism with explicit diversity pressure

---

## Recommended Immediate Actions

1. **Implement parallel mutations** (as in OPTIMIZATION_PLAN.md)
2. **Raise tier thresholds** to (3q @ 0.15, 8q @ 0.30, 20q @ 0.45)
3. **Add 3-5 seed genomes** with known good configurations
4. **Implement focused mutation** that targets weak metrics
5. **Remove drop_zone_inc and start_subset** from evolved params (fix at defaults)
6. **Profile graph operations** to prioritize GPU migration

---

## Questions to Answer with Experiments

1. What's the correlation between performance at 8 queries vs full eval? (validates early stopping thresholds)
2. How much does gradient-based hyperparameter tuning improve convergence vs random mutation?
3. What fraction of evaluation time is spent in graph traversal vs vector operations?
4. Does adaptive step count (early convergence) significantly reduce average steps per query?
5. What's the minimum n_agents that still achieves target metrics on each dataset?
