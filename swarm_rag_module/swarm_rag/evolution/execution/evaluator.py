from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np
from typing import List, Dict, Any, Protocol, runtime_checkable

from swarm_rag.interfaces.protocols import RetrievalBackend
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
        global_max_threads: int = 16,
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
        
        self.concurrent_evaluations = min(concurrent_evaluations, global_max_threads)
        self.global_max_threads = global_max_threads


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
        if queries is None or ground_truth is None:
            raise Exception

        unevaluated = [g for g in population if not g.evaluated]
        if not unevaluated:
            return
        print(f"Evaluating {len(unevaluated)} genomes...")

        # Calculate Resource Budget
        batch_size = min(len(unevaluated), self.concurrent_evaluations)
        threads_per_retriever = max(1, self.global_max_threads // batch_size)
        print(f"  > Concurrency: {batch_size} genomes")
        print(f"  > Thread Budget: {threads_per_retriever} threads per genome")

        # Process in chunks to respect concurrency limits
        for i in range(0, len(unevaluated), batch_size):
            batch = unevaluated[i : i + batch_size]
            self._evaluate_batch(batch, queries, ground_truth, threads_per_retriever)

    def _evaluate_batch(
        self, 
        batch: List[Genome], 
        queries: List[str], 
        ground_truth: List[List[Any]],
        max_workers_per_query: int
    ):
        """
        Runs a batch of evaluations concurrently.
        """
        # We use a ThreadPool here to run multiple *Genome Evaluations* at once.
        # Each Genome Evaluation will internally run multiple *Query Retrievals*.
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            future_to_genome = {
                executor.submit(
                    self._evaluate_single, 
                    genome, 
                    queries, 
                    ground_truth, 
                    max_workers_per_query
                ): genome 
                for genome in batch
            }
            
            for future in as_completed(future_to_genome):
                genome = future_to_genome[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Genome {genome.id} evaluation failed: {e}")
                    genome.fitness = -1.0
                    genome.evaluated = True

    def _evaluate_single(
        self, 
        genome: Genome, 
        max_workers: int,
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
            max_workers=max_workers,
            **retriever_kwargs
        )
        
        total_latency = time.time() - start_time
        avg_latency = total_latency / max(1, len(queries))

        # 3. Compute Metrics
        all_query_metrics = []
        for q_idx, retrieved_items in enumerate(batch_results):
            q_metrics = self.evaluator.calculate_metrics(
                retrieved_nodes=retrieved_items, 
                ground_truth_ids=ground_truth[q_idx], 
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