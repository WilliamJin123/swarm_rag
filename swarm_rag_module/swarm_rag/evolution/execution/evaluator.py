from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np
from typing import List, Any

from swarm_rag.interfaces.protocols import RetrievalBackend
from ...eval.metrics import Evaluator
from .fitness import FitnessCalculator
from ..types.genome import GenomeCompiler, Genome


# TO CHANGE: PLUGGING IN CUSTOM EVALUATION FUNCTIONS / PROCESSESES
class PopulationEvaluator:
    """
    Isolates the heavy lifting: Running the retriever and computing metrics.
    """
    def __init__(
        self, 
        retriever: RetrievalBackend, # SwarmRetriever conforms to this protocol
        evaluator: Evaluator,
        fitness_calc: FitnessCalculator,
        concurrent_evaluations: int = 4,
        queries: List[str] = None,
        ground_truth: List[List[Any]] = None
    ):
        self.retriever = retriever
        self.evaluator = evaluator
        self.fitness_calc = fitness_calc
        self.queries = queries
        self.ground_truth = ground_truth
        self.compiler = GenomeCompiler()
        self.concurrent_evaluations = concurrent_evaluations

    def evaluate(
        self, 
        population: List[Genome],
        queries: List[str] = None,
        ground_truth: List[List[Any]] = None
    ) -> None:
        """
        Evaluates the population in-place.
        """
        queries = queries or self.queries
        ground_truth = ground_truth or self.ground_truth

        unevaluated = [g for g in population if not g.evaluated]
        if not unevaluated:
            return
        batch_size = self.concurrent_evaluations
        
        print(f"Evaluating {len(unevaluated)} genomes...")
        print(f"  > Concurrency: {batch_size} genomes parallel")
        print(f"  > Mode: Sequential Queries per Genome (max_workers=1)")

        batch_size = self.concurrent_evaluations
        for i in range(0, len(unevaluated), batch_size):
            batch = unevaluated[i : i + batch_size]
            self._evaluate_batch(batch, queries, ground_truth)

    def _evaluate_batch(
        self, 
        batch: List[Genome], 
        queries: List[str], 
        ground_truth: List[List[Any]],
    ):
        """
        Runs a batch of evaluations concurrently.
        """
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            future_to_genome = {
                executor.submit(self._evaluate_single, g, queries, ground_truth): g 
                for g in batch
            }
            
            for future in as_completed(future_to_genome):
                genome = future_to_genome[future]
                # try:
                future.result()
                # except Exception as e:
                #     print(f"Genome {genome.id} evaluation failed: {e}")
                #     from .fitness import FitnessResult
                #     genome.fitness = FitnessResult(0.0, 0.0, 9999.0)
                #     genome.evaluated = True

    def _evaluate_single(
        self, 
        genome: Genome, 
        queries: List[str],
        ground_truth: List[List[Any]]
    ):
        """
        Runs a single evaluation with a strict thread budget.
        """
        # Compile
        retriever_kwargs = self.compiler.compile(genome)

        # Run Retrieval
        start_time = time.time()
        
        batch_results = self.retriever.retrieve_batch(
            queries=queries,
            max_workers=1,
            **retriever_kwargs
        )
        
        total_latency = time.time() - start_time

        # Compute Metrics
        all_metrics = []
        for q_idx, retrieved_items in enumerate(batch_results):
            m = self.evaluator.calculate_metrics(
                retrieved_nodes=retrieved_items, 
                ground_truth_ids=ground_truth[q_idx],
                latency_sec=0 
            )
            all_metrics.append(m)
        
        avg_metrics = self._mean_metrics(all_metrics)
        avg_metrics['latency_ms'] = total_latency / max(1, len(queries))

        # Assign to Genome
        genome.metrics = avg_metrics
        genome.fitness = self.fitness_calc.calculate(avg_metrics, genome)
        genome.evaluated = True

    def _mean_metrics(self, all_metrics):
        if not all_metrics: return {}
        keys = all_metrics[0].keys()
        aggregated = {}
        for k in keys:
            values = [m[k] for m in all_metrics]
            aggregated[k] = float(np.mean(values))
            aggregated[f"var_{k}"] = float(np.var(values))

        # We still pick one main metric to represent the overall "Stability Score"
        priority_keys = [
            # Preferred (The standard benchmarks for this project)
            "Recall@10", "Hit@10",  "MRR",
            
            # Stricter Metrics (High Precision)
            "Recall@5", "Hit@5", 
            "Recall@1", "Hit@1",
            
            # Looser Metrics (Broad Recall)
            "Recall@20", "Hit@20"
        ]

        main_key = next((k for k in priority_keys if k in keys), None)
        
        if main_key:
            aggregated["variance"] = aggregated[f"var_{main_key}"]
        else:
            # This handles edge cases like "Recall@15" or custom metrics
            fallback = next((k for k in keys if "Recall" in k or "Hit" in k), None)
            aggregated["variance"] = aggregated[f"var_{fallback}"] if fallback else 0.0
            
        return aggregated