"""
Evaluable Retriever Protocol - Interface for evolution-compatible retrievers.

This protocol decouples the evolution system from specific retriever implementations.
Any retriever conforming to this protocol can be optimized by the evolution engine.

This enables:
1. Testing evolution with mock retrievers
2. Supporting multiple retriever backends
3. Clear separation of concerns between evolution and retrieval
"""
from typing import Protocol, Dict, Any, List, runtime_checkable


@runtime_checkable
class EvaluableRetriever(Protocol):
    """
    Interface that evolution expects from any retriever.

    This protocol defines the minimal contract required for a retriever
    to be optimized by the evolution engine. Implementations can add
    additional methods, but must provide these core capabilities.

    The evolution system will:
    1. Compile genomes to retriever-compatible kwargs
    2. Call evaluate_genome() to get fitness metrics
    3. Use metrics to drive selection and breeding

    Example Implementation:
        class MyRetriever:
            def evaluate_genome(
                self,
                genome_kwargs: Dict[str, Any],
                queries: List[str],
                ground_truth: List[List[str]]
            ) -> Dict[str, float]:
                # Run retrieval with genome config
                results = self._retrieve_batch(queries, **genome_kwargs)
                # Calculate metrics
                return {
                    "quality": calculate_recall(results, ground_truth),
                    "cost": calculate_latency(results),
                    "stability": 1.0 - calculate_variance(results)
                }
    """

    def evaluate_genome(
        self,
        genome_kwargs: Dict[str, Any],
        queries: List[str],
        ground_truth: List[List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate a genome configuration and return metrics.

        This method should:
        1. Apply the genome configuration (genome_kwargs)
        2. Run retrieval on the provided queries
        3. Compare results against ground truth
        4. Return a dictionary of metrics

        Required Metrics:
            - quality: float (0-1, higher is better)
            - cost: float (lower is better, can be latency or other)

        Optional Metrics:
            - stability: float (0-1, higher is better)
            - recall_at_k: float for various k values
            - hit_at_k: float for various k values
            - mrr: float (Mean Reciprocal Rank)

        Args:
            genome_kwargs: Configuration produced by GenomeCompiler.compile()
            queries: List of query strings to evaluate
            ground_truth: List of ground truth ID lists (one per query)

        Returns:
            Dictionary mapping metric names to values
        """
        ...
