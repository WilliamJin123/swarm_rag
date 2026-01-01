from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np
from typing import List, Dict, Any, Protocol, runtime_checkable
from ...eval.metrics import Evaluator
from ...core.swarm_retriever import SwarmRetriever
from ..types.genome import Genome
from .fitness import FitnessCalculator
from ..types.genome import GenomeCompiler



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
        queries: List[str],
        ground_truth: List[List[Any]],
        max_workers: int = 16
    ):
        self.retriever = retriever
        self.evaluator = evaluator
        self.fitness_calc = fitness_calc
        self.queries = queries
        self.ground_truth = ground_truth
        self.compiler = GenomeCompiler()
        self.max_workers = max_workers


    def evaluate(self, population: List[Genome], parallel_eval: bool = True) -> None:
        """
        Evaluates the population in-place.
        """
        # 1. Filter unevaluated genomes to save compute
        unevaluated = [g for g in population if not g.evaluated]
        if not unevaluated:
            return

        print(f"Evaluating {len(unevaluated)} new genomes...")

        if parallel_eval and len(unevaluated) > 1:
            # Heuristic: Don't run more genomes than we have threads/2
            # (Reserve threads for the actual retrieval work)
            concurrent_genomes = min(len(unevaluated), max(1, self.max_workers // 4))

            # Calculate worker budget per genome
            workers_per_genome = max(1, self.max_workers // concurrent_genomes)

            print(f"Parallel Eval: {concurrent_genomes} genomes at a time (Limit: {workers_per_genome} threads/genome)")

            with ThreadPoolExecutor(max_workers=concurrent_genomes) as executor:
                # We map the evaluate_single function with the allocated worker budget
                futures = {
                    executor.submit(self._evaluate_single, g, workers_per_genome): g 
                    for g in unevaluated
                }
                
                for future in as_completed(futures):
                    g = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Genome evaluation failed for {g}: {e}")
                        g.fitness = -1.0 # Penalize failures
                        g.evaluated = True
        else:
            for genome in unevaluated:
                self._evaluate_single(genome, max_workers=self.max_workers)

    def _evaluate_single(self, genome: Genome, max_workers: int):
        """
        Evaluating one by one allows for finer error handling, though
        batching at the retriever level (as you had) is still possible here.
        """
        # 1. Compile
        retriever_kwargs = self.compiler.compile(genome)

        # 2. Run Retrieval (The expensive part)
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