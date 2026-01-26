# Evolution Optimization Plan

## Target Metrics
- **Hit@1:** > 50%
- **Hit@5:** > 75%
- **Recall@20:** > 75%
- **MRR:** > 75%
- **Target quality_score:** ~0.65-0.70

---

## Current Bottlenecks

### 1. Sequential LLM Mutations (~250s/gen with LLM vs ~48s/gen without)
**Location:** `map_elites/loop.py:54-76`

Each offspring is mutated sequentially. With 15-20 offspring and 1-3 LLM calls per mutation (retries), this serializes ~30-60 API calls.

### 2. Fixed Probe Evaluation (20 queries, threshold 0.1)
**Location:** `execution/evaluator.py:143-180`

Current logic is too permissive:
- 20 queries is still expensive for obviously bad genomes
- Threshold (0.1) is far too low for high-quality targets
- No progressive refinement

---

## Solution 1: Parallel LLM Mutations

**File:** `map_elites/loop.py`

Replace the sequential mutation loop with a parallel implementation.

### Implementation

```python
# At top of file - add imports
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

# Add worker function before class
def _apply_mutation_worker(args: Tuple[Genome, EvolutionContext, callable]) -> Genome:
    """Worker function for parallel mutation."""
    child, context, mutation_fn = args
    return mutation_fn(child, context)


class MapElitesLoop:
    def __init__(self, context: EvolutionContext):
        # ... existing init code ...

        # Add: Number of parallel mutation workers
        self.mutation_workers = 4

    def step(self, archive: MapElitesArchive) -> List[Genome]:
        """Generates offspring with parallel mutations."""
        if not archive.grid:
            return []

        crossover_rate = self.context.config.genetic.crossover_rate

        # Phase 1: Create all children (selection + crossover) - fast, sequential
        children_to_mutate = []
        while len(children_to_mutate) < self.batch_size:
            p1 = archive.select_random()

            if random.random() < crossover_rate:
                p2 = archive.select_random()
                child = self.crossover_fn(p1, p2, self.context)
                child._parent_id = p1.id
                child._parent2_id = p2.id
            else:
                self._offspring_counter += 1
                child_id = f"g{self.context.generation}_c{self._offspring_counter}"
                child = p1.copy(new_id=child_id)
                child._parent_id = p1.id

            children_to_mutate.append(child)

        # Phase 2: Mutate in parallel - slow (LLM calls)
        offspring = self._parallel_mutate(children_to_mutate)
        return offspring

    def _parallel_mutate(self, children: List[Genome]) -> List[Genome]:
        """Apply mutations in parallel using thread pool."""
        with ThreadPoolExecutor(max_workers=self.mutation_workers) as executor:
            future_to_idx = {
                executor.submit(
                    _apply_mutation_worker,
                    (child, self.context, self.mutation_fn)
                ): i
                for i, child in enumerate(children)
            }

            results = [None] * len(children)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Mutation failed for child {idx}: {e}")
                    results[idx] = children[idx]  # Fallback: unmutated

        return results
```

**Expected Speedup:** 3-4x for LLM-guided mutations

---

## Solution 2: Multi-Stage Progressive Evaluation

**File:** `execution/evaluator.py`

Replace the single-stage probe with multi-stage progressive evaluation using higher thresholds.

### Stage Thresholds

| Stage | Queries | Min Quality | What It Filters |
|-------|---------|-------------|-----------------|
| 1 | 5 | 0.20 | Broken/garbage genomes |
| 2 | 15 | 0.35 | Mediocre genomes with no potential |
| 3 | 30 | 0.45 | Genomes unlikely to reach targets |
| Full | All | — | Only serious contenders |

### Implementation

```python
class PopulationEvaluator:
    def __init__(
        self,
        retriever: RetrievalBackend,
        evaluator: Evaluator,
        fitness_calc: FitnessCalculator,
        concurrent_evaluations: int = 4,
        max_workers_per_retrieval: int = 1,
        queries: List[str] = None,
        ground_truth: List[List[Any]] = None,
        track_decisions: bool = False,
        decision_sample_rate: float = 1.0,
    ):
        self.retriever = retriever
        self.evaluator = evaluator
        self.fitness_calc = fitness_calc
        self.queries = queries
        self.ground_truth = ground_truth
        self.compiler = GenomeCompiler()
        self.concurrent_evaluations = concurrent_evaluations
        self.max_workers_per_retrieval = max_workers_per_retrieval
        self.track_decisions = track_decisions
        self.decision_sample_rate = decision_sample_rate

        # Progressive evaluation stages: (n_queries, min_quality_to_continue)
        self.progressive_stages = [
            (5, 0.20),    # Ultra-fast: filter broken genomes
            (15, 0.35),   # Quick: filter mediocre genomes
            (30, 0.45),   # Standard: only contenders continue
        ]

    # ... keep evaluate() and _evaluate_batch() unchanged ...

    def _evaluate_single(
        self,
        genome: Genome,
        queries: List[str],
        ground_truth: List[List[Any]]
    ):
        """Progressive multi-stage evaluation with high-quality thresholds."""
        retriever_kwargs = self.compiler.compile(genome)
        decision_tracker = self._create_decision_tracker()

        start_time = time.time()
        all_results = []
        all_gt = []

        # Run progressive stages
        queries_used = 0
        for stage_queries, min_quality in self.progressive_stages:
            if stage_queries > len(queries):
                stage_queries = len(queries)

            # Get queries for this stage (incremental from last stage)
            stage_start = queries_used
            stage_end = stage_queries
            stage_q = queries[stage_start:stage_end]
            stage_gt = ground_truth[stage_start:stage_end]

            if not stage_q:
                break

            # Evaluate stage queries
            if decision_tracker is not None:
                stage_results = [
                    self.retriever.retrieve(
                        q, decision_tracker=decision_tracker, **retriever_kwargs
                    )
                    for q in stage_q
                ]
            else:
                stage_results = self.retriever.retrieve_batch(
                    queries=stage_q,
                    max_workers=self.max_workers_per_retrieval,
                    genome_id=f"{genome.id}_stage{stage_queries}",
                    **retriever_kwargs
                )

            all_results.extend(stage_results)
            all_gt.extend(stage_gt)
            queries_used = stage_end

            # Calculate cumulative metrics for all queries so far
            stage_metrics = []
            for i, res in enumerate(all_results):
                m = self.evaluator.calculate_metrics(res, all_gt[i], latency_sec=0)
                stage_metrics.append(m)

            avg_metrics = self._mean_metrics(stage_metrics)
            avg_metrics['latency'] = (time.time() - start_time) / len(all_results)
            fitness = self.fitness_calc.calculate(avg_metrics, genome)

            # Check if genome should be filtered out
            if fitness.quality_score < min_quality:
                logger.info(
                    f"  > [Stage {stage_queries}] {genome.id} aborted. "
                    f"Quality: {fitness.quality_score:.4f} < {min_quality}"
                )
                genome.metrics = avg_metrics
                genome.fitness = fitness
                genome.evaluated = True
                if decision_tracker:
                    genome.decision_context = decision_tracker.to_summary_dict()
                return

        # Passed all stages - evaluate on remaining queries
        remaining_q = queries[queries_used:]
        remaining_gt = ground_truth[queries_used:]

        if remaining_q:
            remaining_results = self.retriever.retrieve_batch(
                queries=remaining_q,
                max_workers=self.max_workers_per_retrieval,
                genome_id=genome.id,
                **retriever_kwargs
            )
            all_results.extend(remaining_results)
            all_gt.extend(remaining_gt)

        # Final metrics calculation
        total_latency = time.time() - start_time
        all_metrics = [
            self.evaluator.calculate_metrics(res, gt, latency_sec=0)
            for res, gt in zip(all_results, all_gt)
        ]

        avg_metrics = self._mean_metrics(all_metrics)
        avg_metrics['latency'] = total_latency / max(1, len(queries))
        avg_metrics['complexity'] = float(genome.complexity())

        genome.metrics = avg_metrics
        genome.fitness = self.fitness_calc.calculate(avg_metrics, genome)
        genome.evaluated = True

        if decision_tracker:
            genome.decision_context = decision_tracker.to_summary_dict()
```

---

## Why These Thresholds?

### Quality Score Formula
```
quality = 0.25 * (Hit@1 + Hit@5 + MRR + Recall@20)
```

**Target quality ~0.65:**
```
0.65 = 0.25 * (0.50 + 0.75 + 0.75 + 0.60)
```

### Threshold Logic

**Stage 1 (5 queries, threshold 0.20):**
- Genomes averaging <20% across metrics are broken
- Example: H@1=5%, H@5=20%, MRR=15%, R@20=20% → quality=0.15 → ABORT
- Saves 75% of queries for garbage genomes

**Stage 2 (15 queries, threshold 0.35):**
- After 15 queries, need real signal
- Example: H@1=20%, H@5=40%, MRR=35%, R@20=45% → quality=0.35 → CONTINUE
- Filters mediocre genomes that won't reach 0.65+ targets

**Stage 3 (30 queries, threshold 0.45):**
- Statistical confidence is high after 30 queries
- A genome at 0.45 can reasonably improve 40% with evolution
- A genome at 0.30 would need 120% improvement (unlikely)

---

## Expected Performance Impact

| Scenario | Before | After | Speedup |
|----------|--------|-------|---------|
| LLM mutation (15 offspring) | ~180s | ~50s | **3.6x** |
| Garbage genome (quality<0.20) | 20 queries | 5 queries | **4x** |
| Poor genome (quality<0.35) | 20+ queries | 15 queries | **1.5x** |
| Mediocre genome (quality<0.45) | All queries | 30 queries | **2-3x** |
| Good genome (quality≥0.45) | All queries | All queries | 1x |
| **Overall generation time** | ~300s | **~80-100s** | **3-4x** |

---

## Implementation Checklist

- [ ] Add `ThreadPoolExecutor` imports to `loop.py`
- [ ] Add `_apply_mutation_worker` function to `loop.py`
- [ ] Add `mutation_workers = 4` to `MapElitesLoop.__init__`
- [ ] Replace `step()` method with parallel version
- [ ] Add `_parallel_mutate()` method
- [ ] Add `progressive_stages` to `PopulationEvaluator.__init__`
- [ ] Replace `_evaluate_single()` with progressive version
- [ ] Test with small dataset to verify stages filter correctly
- [ ] Benchmark generation time improvement

---

## Testing Verification

After implementation, logs should show:
```
  > [Stage 5] gen0_3 aborted. Quality: 0.0821 < 0.20
  > [Stage 15] gen0_7 aborted. Quality: 0.2891 < 0.35
  > [Stage 30] gen0_12 aborted. Quality: 0.4012 < 0.45
  > Finished 'gen0_2' (1/15) | Qual: 0.5234 | ...  (passed all stages)
```

This confirms the progressive filtering is working correctly.
