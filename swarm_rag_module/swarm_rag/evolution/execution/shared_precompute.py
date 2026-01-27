"""
Shared Pre-computation Module for Evolution Loop Optimization.

Pre-computes shared data ONCE per generation to reduce redundant computation:
- Batch embed ALL queries into a single GPU tensor
- Pre-compute initial pools for each unique initial_pool_size
- Store ground truth as sets once

This eliminates N_genomes * N_queries embedding lookups down to N_queries lookups.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
import torch

from swarm_rag.interfaces.protocols import RetrievalBackend

logger = logging.getLogger(__name__)


@dataclass
class SharedPrecomputeContext:
    """
    Context containing pre-computed shared data for a generation's evaluation.

    Attributes:
        query_embeddings: Pre-computed query embeddings tensor (n_queries, dim)
        initial_pools: Dict mapping pool_size -> List[List[int]] of candidate IDs per query
        ground_truth_sets: Pre-converted ground truth as sets for fast lookup (CPU fallback)
        ground_truth_tensor: Pre-computed ground truth as GPU tensor (n_queries, max_gt_size)
        gt_sizes: Tensor of ground truth sizes per query (n_queries,)
        queries: Original query strings (for reference)
        device: Device where tensors are stored
    """
    query_embeddings: torch.Tensor
    initial_pools: Dict[int, List[List[int]]]
    ground_truth_sets: List[Set[Any]]
    queries: List[str]
    device: str = "cpu"

    # GPU-precomputed ground truth tensors (None if CPU-only)
    ground_truth_tensor: Optional[torch.Tensor] = None  # (n_queries, max_gt_size) padded with -1
    gt_sizes: Optional[torch.Tensor] = None  # (n_queries,) number of relevant items per query

    # Statistics for monitoring
    n_queries: int = 0
    n_pool_sizes: int = 0
    precompute_time_sec: float = 0.0


def prepare_shared_context(
    retriever: RetrievalBackend,
    queries: List[str],
    ground_truth: List[List[Any]],
    unique_pool_sizes: List[int],
    device: str = "cpu"
) -> SharedPrecomputeContext:
    """
    Pre-compute shared data for all genomes in a generation.

    This function is called ONCE at the start of evaluation to compute:
    1. All query embeddings in a single batch
    2. Initial search pools for each unique pool size needed by genomes
    3. Ground truth converted to sets for O(1) lookup

    Args:
        retriever: The retrieval backend with embedding and search capabilities
        queries: List of query strings
        ground_truth: List of ground truth ID lists per query
        unique_pool_sizes: List of unique initial_pool_size values needed
        device: Target device for tensors ("cuda" or "cpu")

    Returns:
        SharedPrecomputeContext with all pre-computed data
    """
    import time
    start_time = time.time()

    n_queries = len(queries)
    logger.info(f"Pre-computing shared context for {n_queries} queries...")

    # 1. Batch embed all queries
    logger.debug("  > Batch embedding queries...")
    if hasattr(retriever, '_get_cached_query_embeddings_batch'):
        # Use retriever's internal batched embedding method
        query_embeddings = retriever._get_cached_query_embeddings_batch(queries)
    elif hasattr(retriever, 'embed_fn') and hasattr(retriever.embed_fn, 'embed_query_batch'):
        # Direct access to embedding provider
        query_embeddings = retriever.embed_fn.embed_query_batch(queries)
        if not isinstance(query_embeddings, torch.Tensor):
            query_embeddings = torch.as_tensor(query_embeddings, dtype=torch.float32)
    else:
        # Fallback: embed one at a time (shouldn't happen with SwarmRetriever)
        embeddings = []
        for q in queries:
            if hasattr(retriever, '_get_cached_query_vector'):
                emb = retriever._get_cached_query_vector(q)
            else:
                emb = retriever.embed_fn.embed_query(q)
            if not isinstance(emb, torch.Tensor):
                emb = torch.as_tensor(emb, dtype=torch.float32)
            embeddings.append(emb)
        query_embeddings = torch.stack(embeddings)

    # Move to target device if needed
    if device == "cuda" and query_embeddings.device.type != "cuda":
        query_embeddings = query_embeddings.to(device=device)

    logger.debug(f"  > Query embeddings shape: {query_embeddings.shape}")

    # 2. Pre-compute initial pools for each unique pool size
    logger.debug(f"  > Pre-computing initial pools for {len(unique_pool_sizes)} unique sizes...")
    initial_pools: Dict[int, List[List[int]]] = {}

    for pool_size in sorted(unique_pool_sizes, reverse=True):
        # Compute pools for this size
        # We can reuse the largest pool and slice down
        if initial_pools and pool_size < max(initial_pools.keys()):
            # Slice from a larger pool
            larger_size = min(s for s in initial_pools.keys() if s > pool_size)
            initial_pools[pool_size] = [
                pool[:pool_size] for pool in initial_pools[larger_size]
            ]
        else:
            # Compute fresh
            pools = _batch_initial_search(
                retriever,
                query_embeddings,
                pool_size
            )
            initial_pools[pool_size] = pools

        logger.debug(f"    > Pool size {pool_size}: {len(initial_pools[pool_size])} pools computed")

    # 3. Convert ground truth to sets (CPU fallback) and precompute GPU tensors
    logger.debug("  > Converting ground truth to sets and GPU tensors...")
    ground_truth_sets = []
    gt_int_lists = []  # For GPU tensor construction
    all_ints = True

    for gt in ground_truth:
        try:
            gt_ints = [int(g) for g in gt]
            gt_set = set(gt_ints)
            gt_int_lists.append(gt_ints)
        except (ValueError, TypeError):
            gt_set = set(str(g) for g in gt)
            gt_int_lists.append([])  # Can't use GPU for string IDs
            all_ints = False
        ground_truth_sets.append(gt_set)

    # 4. Pre-compute ground truth GPU tensor if on CUDA and all IDs are integers
    ground_truth_tensor = None
    gt_sizes = None

    if device == "cuda" and all_ints and gt_int_lists:
        logger.debug("  > Building GPU ground truth tensor...")
        max_gt_size = max(len(gt) for gt in gt_int_lists) if gt_int_lists else 0
        if max_gt_size > 0:
            # Create padded tensor directly on GPU
            ground_truth_tensor = torch.full(
                (n_queries, max_gt_size), -1, dtype=torch.long, device=device
            )
            gt_sizes = torch.zeros(n_queries, dtype=torch.float32, device=device)

            for i, gt_list in enumerate(gt_int_lists):
                if gt_list:
                    gt_sizes[i] = len(gt_list)
                    ground_truth_tensor[i, :len(gt_list)] = torch.tensor(
                        gt_list, dtype=torch.long, device=device
                    )

            logger.debug(f"    > GT tensor shape: {ground_truth_tensor.shape}")

    precompute_time = time.time() - start_time
    logger.info(f"  > Pre-computation complete in {precompute_time:.2f}s")

    return SharedPrecomputeContext(
        query_embeddings=query_embeddings,
        initial_pools=initial_pools,
        ground_truth_sets=ground_truth_sets,
        queries=queries,
        device=device,
        ground_truth_tensor=ground_truth_tensor,
        gt_sizes=gt_sizes,
        n_queries=n_queries,
        n_pool_sizes=len(unique_pool_sizes),
        precompute_time_sec=precompute_time
    )


def _batch_initial_search(
    retriever: RetrievalBackend,
    query_embeddings: torch.Tensor,
    pool_size: int
) -> List[List[int]]:
    """
    Perform batch initial search for all queries.

    Uses GPU-accelerated batch search when available.

    Args:
        retriever: The retrieval backend
        query_embeddings: Pre-computed query embeddings (n_queries, dim)
        pool_size: Number of candidates per query

    Returns:
        List of candidate ID lists, one per query
    """
    # Check if retriever has batch search capability
    if hasattr(retriever, '_batch_initial_search'):
        return retriever._batch_initial_search(query_embeddings, pool_size)

    # Check if vector store supports batch search
    if hasattr(retriever, 'vector_store') and hasattr(retriever.vector_store, 'search_batch'):
        try:
            results = retriever.vector_store.search_batch(query_embeddings, pool_size)
            pools = []
            for res in results:
                valid_ids = [
                    r['id'] for r in res
                    if retriever.graph_store.contains(r['id'])
                ]
                pools.append(valid_ids)
            return pools
        except Exception as e:
            logger.debug(f"Batch search failed, falling back to sequential: {e}")

    # Fallback: sequential search
    pools = []
    for vec in query_embeddings:
        search_res = retriever.vector_store.search(vec, limit=pool_size)
        valid_ids = [
            r['id'] for r in search_res
            if retriever.graph_store.contains(r['id'])
        ]
        pools.append(valid_ids)

    return pools


def get_unique_pool_sizes(genomes: List[Any], compiler: Any = None) -> List[int]:
    """
    Extract unique initial_pool_size values from a list of genomes.

    Args:
        genomes: List of genome objects
        compiler: Optional compiler (not used, kept for backward compatibility)

    Returns:
        Sorted list of unique pool sizes
    """
    unique_sizes = set()
    default_pool_size = 30  # Default from SwarmRetriever._DEFAULT_PARAMS

    for genome in genomes:
        # Get pool size directly from genome params (works for both modes)
        pool_size = genome.params.get('initial_pool_size', default_pool_size)
        unique_sizes.add(pool_size)

    return sorted(unique_sizes)


@dataclass
class BatchedRetrievalResults:
    """
    Container for batched retrieval results across all genomes.

    Used for cross-genome metric batching.
    """
    # Results indexed by genome_id
    results_by_genome: Dict[str, List[List[Dict]]] = field(default_factory=dict)

    # Flattened data for batch metric computation
    all_retrieved_ids: Optional[torch.Tensor] = None  # (n_genomes * n_queries, max_k)
    genome_query_indices: Optional[List[Tuple[str, int]]] = None  # [(genome_id, query_idx), ...]

    def add_genome_results(self, genome_id: str, results: List[List[Dict]]):
        """Add results for a single genome."""
        self.results_by_genome[genome_id] = results

    def prepare_for_batch_metrics(
        self, max_k: int = 20, device: str = "cpu"
    ) -> Tuple[torch.Tensor, List[Tuple[str, int]]]:
        """
        Prepare flattened tensors for batch metric computation.

        Creates tensors directly on the target device to avoid GPU-CPU handoffs.

        Args:
            max_k: Maximum number of retrieved items to consider
            device: Target device for tensors ("cpu" or "cuda")

        Returns:
            Tuple of (retrieved_ids tensor, index mapping)
        """
        # Count total rows for pre-allocation
        total_rows = sum(len(results_list) for results_list in self.results_by_genome.values())

        if total_rows == 0:
            self.all_retrieved_ids = torch.tensor([], device=device, dtype=torch.long)
            self.genome_query_indices = []
            return self.all_retrieved_ids, self.genome_query_indices

        # Pre-allocate full tensor on target device
        self.all_retrieved_ids = torch.full(
            (total_rows, max_k), -1, dtype=torch.long, device=device
        )
        self.genome_query_indices = []

        row_idx = 0
        for genome_id, results_list in self.results_by_genome.items():
            for query_idx, results in enumerate(results_list):
                # Extract IDs from results directly into pre-allocated tensor
                for j, item in enumerate(results[:max_k]):
                    if isinstance(item, dict):
                        try:
                            self.all_retrieved_ids[row_idx, j] = int(item.get('id', -1))
                        except (ValueError, TypeError):
                            pass
                    else:
                        try:
                            self.all_retrieved_ids[row_idx, j] = int(item)
                        except (ValueError, TypeError):
                            pass
                self.genome_query_indices.append((genome_id, query_idx))
                row_idx += 1

        return self.all_retrieved_ids, self.genome_query_indices
