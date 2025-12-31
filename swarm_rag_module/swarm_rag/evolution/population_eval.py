import time
import numpy as np
from typing import List, Dict, Any
from ..core.swarm_retriever import SwarmRetriever
from ..eval.metrics import Evaluator
from .genome import Genome
from .fitness import FitnessCalculator
from .genome import GenomeCompiler

# TO CHANGE: PLUGGING IN CUSTOM EVALUATION FUNCTIONS / PROCESSESES
class PopulationEvaluator:
    """
    Isolates the heavy lifting: Running the retriever and computing metrics.
    """
    def __init__(
        self, 
        retriever: SwarmRetriever,
        evaluator: Evaluator,
        fitness_calc: FitnessCalculator,
        queries: List[str],
        ground_truth: List[List[Any]]
    ):
        self.retriever = retriever
        self.evaluator = evaluator
        self.fitness_calc = fitness_calc
        self.queries = queries
        self.ground_truth = ground_truth
        self.compiler = GenomeCompiler()

    def evaluate(self, population: List[Genome]) -> None:
        """
        Evaluates the population in-place.
        """
        # 1. Filter unevaluated genomes to save compute
        unevaluated = [g for g in population if not g.evaluated]
        if not unevaluated:
            return

        print(f"Evaluating {len(unevaluated)} new genomes...")

        for genome in unevaluated:
            self._evaluate_single(genome)

    def _evaluate_single(self, genome: Genome):
        """
        Evaluating one by one allows for finer error handling, though
        batching at the retriever level (as you had) is still possible here.
        """
        # 1. Compile
        retriever_kwargs = self.compiler.compile(genome)

        # 2. Run Retrieval (The expensive part)
        # Note: If your retriever supports batching DIFFERENT configs, use that. 
        # Since it likely doesn't, we iterate. 
        # (If your retriever.retrieve_batch supports running the SAME config for multiple queries, this is correct).
        start_time = time.time()
        
        batch_results = self.retriever.retrieve_batch(
            queries=self.queries,
            parallel_queries=True, 
            **retriever_kwargs
        )
        
        total_latency = time.time() - start_time
        avg_latency = total_latency / max(1, len(self.queries))

        # 3. Compute Metrics
        all_query_metrics = []
        for q_idx, retrieved_items in enumerate(batch_results):
            q_metrics = self.evaluator.calculate_metrics(
                retrieved_nodes=retrieved_items, 
                ground_truth_ids=self.ground_truth[q_idx], 
                latency_sec=avg_latency 
            )
            all_query_metrics.append(q_metrics)
        
        averaged_metrics = self._mean_metrics(all_query_metrics)

        # 4. Assign to Genome
        genome.fitness = self.fitness_calc.calculate(averaged_metrics, genome)
        genome.metrics = averaged_metrics
        genome.evaluated = True

    def _mean_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        if not all_metrics: return {}
        keys = all_metrics[0].keys()
        return {
            k: float(np.mean([m[k] for m in all_metrics])) 
            for k in keys if isinstance(all_metrics[0][k], (int, float))
        }