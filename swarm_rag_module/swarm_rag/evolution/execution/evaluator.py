from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
import numpy as np
from typing import List, Any, Optional

from swarm_rag.interfaces.protocols import RetrievalBackend
from ...eval.metrics import Evaluator
from .fitness import FitnessCalculator
from ..types.genome import GenomeCompiler, Genome


logger = logging.getLogger(__name__)


class PopulationEvaluator:
    """
    Isolates the heavy lifting: Running the retriever and computing metrics.

    Supports optional decision tracking for LLM-guided mutations.
    When enabled, agent decision data is captured during evaluation and
    stored on the genome for use by the LLM mutation operator.
    """
    def __init__(
        self,
        retriever: RetrievalBackend,  # SwarmRetriever conforms to this protocol
        evaluator: Evaluator,
        fitness_calc: FitnessCalculator,
        concurrent_evaluations: int = 4,
        max_workers_per_retrieval: int = 1,
        queries: List[str] = None,
        ground_truth: List[List[Any]] = None,
        track_decisions: bool = False,  # Enable decision tracking for LLM context
        decision_sample_rate: float = 1.0,  # Fraction of queries to track (1.0 = all)
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
        
        logger.info(f"Evaluating {len(unevaluated)} genomes...")
        logger.info(f"  > Concurrency: {batch_size} genomes parallel")
        logger.info(f"  > max_workers: {self.max_workers_per_retrieval} workers")

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
        logger.info(f"  > Starting batch of {len(batch)} genomes...")

        total_genomes = len(batch)
        completed_count = 0

        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            future_to_genome = {
                executor.submit(self._evaluate_single, g, queries, ground_truth): g 
                for g in batch
            }
            
            for future in as_completed(future_to_genome):
                genome = future_to_genome[future]
                completed_count += 1
                # try:
                future.result()

                qual = genome.fitness.quality_score
                cost = genome.fitness.cost_score
                
                r20 = genome.metrics.get("Recall@20", 0.0)
                h1  = genome.metrics.get("Hit@1", 0.0)
                h5  = genome.metrics.get("Hit@5", 0.0)
                mrr = genome.metrics.get("MRR", 0.0)

                logger.info(
                    f"  > Finished '{genome.id}' ({completed_count}/{total_genomes}) | "
                    f"Qual: {qual:.4f} | Cost: {cost:.1f} | "
                    f"R@20: {r20:.4f} | H@1: {h1:.4f} | H@5: {h5:.4f} | MRR: {mrr:.4f}"
                )
                # except Exception as e:
                #     logger.error(f"Genome {genome.id} evaluation failed: {e}")
                #     # Assign worst possible fitness so it dies out
                #     from .fitness import FitnessResult
                #     # Create a default bad result. 
                #     # Note: fitness calculation might have failed, so we set manually.
                #     genome.fitness = FitnessResult(quality_score=0.0, stability_score=0.0, cost_score=9999.0)
                #     genome.evaluated = True

    def _create_decision_tracker(self) -> Optional[Any]:
        """Create a DecisionTracker if decision tracking is enabled."""
        if not self.track_decisions:
            return None
        try:
            from ..llm.decision_tracker import DecisionTracker
            return DecisionTracker(
                enabled=True,
                sample_rate=self.decision_sample_rate
            )
        except ImportError:
            logger.warning("DecisionTracker not available, disabling decision tracking")
            return None

    def _evaluate_single(
        self,
        genome: Genome,
        queries: List[str],
        ground_truth: List[List[Any]]
    ):
        """
        Runs a single evaluation with a strict thread budget.

        If decision tracking is enabled, captures agent decision data
        and stores a summary on the genome for LLM mutation context.
        """
        # Compile
        retriever_kwargs = self.compiler.compile(genome)

        # Create decision tracker if enabled
        decision_tracker = self._create_decision_tracker()

        # --- STOCHASTIC FILTERING (PROBE PHASE) ---
        probe_size = 20  # Evaluate on first 20 queries
        probe_queries = queries[:probe_size]
        probe_gt = ground_truth[:probe_size]

        start_time = time.time()

        # Run probe with decision tracking on a few queries
        probe_results = []
        if decision_tracker is not None:
            # Use single-query retrieval for probe to capture decisions
            for q in probe_queries:
                res = self.retriever.retrieve(
                    query=q,
                    decision_tracker=decision_tracker,
                    **retriever_kwargs
                )
                probe_results.append(res)
        else:
            # Use batch retrieval for speed when not tracking
            probe_results = self.retriever.retrieve_batch(
                queries=probe_queries,
                max_workers=self.max_workers_per_retrieval,
                genome_id=f"{genome.id}_probe",
                **retriever_kwargs
            )

        probe_metrics = []
        for i, res in enumerate(probe_results):
            m = self.evaluator.calculate_metrics(res, probe_gt[i], latency_sec=0)
            probe_metrics.append(m)

        avg_probe_metrics = self._mean_metrics(probe_metrics)
        avg_probe_metrics['latency'] = (time.time() - start_time) / probe_size

        probe_fitness = self.fitness_calc.calculate(avg_probe_metrics, genome)

        if probe_fitness.quality_score < 0.1:
            logger.info(
                f"  > [Short-Circuit] {genome.id} aborted. "
                f"Probe Quality: {probe_fitness.quality_score:.4f}"
            )
            genome.metrics = avg_probe_metrics
            genome.fitness = probe_fitness
            genome.evaluated = True

            # Store decision context even for short-circuited genomes
            if decision_tracker is not None:
                genome.decision_context = decision_tracker.to_summary_dict()

            return

        # Passed fitness probe on first 20 or queries
        remaining_queries = queries[probe_size:]

        remaining_results = self.retriever.retrieve_batch(
            queries=remaining_queries,
            max_workers=self.max_workers_per_retrieval,
            genome_id=genome.id,
            **retriever_kwargs
        )

        total_latency = time.time() - start_time

        batch_results = probe_results + remaining_results

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
        avg_metrics['latency'] = total_latency / max(1, len(queries))
        avg_metrics['complexity'] = float(genome.complexity())

        # Assign to Genome
        genome.metrics = avg_metrics
        genome.fitness = self.fitness_calc.calculate(avg_metrics, genome)
        genome.evaluated = True

        # Store decision context for LLM mutations
        if decision_tracker is not None:
            genome.decision_context = decision_tracker.to_summary_dict()

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