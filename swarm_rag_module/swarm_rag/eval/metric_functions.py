import math
import torch
from typing import Callable, Dict, List, Set
from collections import Counter
from functools import lru_cache

# =============================================================================
# PRECOMPUTED LOG DENOMINATORS
# =============================================================================
# Precompute log2(i+2) for i=0..MAX_K-1 to avoid repeated tensor creation.
# Used for NDCG, DCG, and entropy calculations.
# These are computed on CPU at import time, then moved to GPU on demand.

_MAX_PRECOMPUTED_K = 200  # Should cover most use cases

# log2(2), log2(3), log2(4), ... log2(MAX_K+1)
# For position i (0-indexed), use _LOG_DENOMS_CPU[i] = log2(i+2)
_LOG_DENOMS_CPU = torch.log2(torch.arange(2, _MAX_PRECOMPUTED_K + 2, dtype=torch.float32))

# Pure Python precomputed values for use in non-tensor functions
_LOG_DENOMS_PYTHON = [float(_LOG_DENOMS_CPU[i]) for i in range(_MAX_PRECOMPUTED_K)]


@lru_cache(maxsize=4)
def _get_log_denoms(max_k: int, device: str) -> torch.Tensor:
    """
    Get precomputed log denominators for DCG on the specified device.

    Cached per device to avoid repeated transfers.

    Args:
        max_k: Maximum k value needed
        device: Target device ("cuda" or "cpu")

    Returns:
        Tensor of shape (max_k,) with log2(i+2) for i in 0..max_k-1
    """
    if max_k <= _MAX_PRECOMPUTED_K:
        return _LOG_DENOMS_CPU[:max_k].to(device)
    # For larger k, compute on the fly (rare case)
    return torch.log2(torch.arange(2, max_k + 2, dtype=torch.float32, device=device))


@lru_cache(maxsize=4)
def _get_idcg_denoms(max_gt: int, device: str) -> torch.Tensor:
    """
    Get precomputed IDCG denominators (1/log2(i+1) for i in 1..max_gt) on the specified device.

    Cached per device to avoid repeated transfers.

    Args:
        max_gt: Maximum ground truth size
        device: Target device ("cuda" or "cpu")

    Returns:
        Tensor of shape (max_gt,) with 1/log2(i+1) for i in 1..max_gt
    """
    return 1.0 / torch.log2(torch.arange(1, max_gt + 1, dtype=torch.float32, device=device) + 1)


def _compute_vectorized_idcg(
    gt_sizes: torch.Tensor,
    k: int,
    idcg_denoms: torch.Tensor,
    device: str
) -> torch.Tensor:
    """
    Compute IDCG for all queries in a vectorized manner without .item() calls.

    IDCG = sum(1/log2(i+1) for i in 1..min(gt_size, k))

    Uses cumulative sum indexing instead of Python loops.

    Args:
        gt_sizes: (batch_size,) tensor of ground truth sizes
        k: Maximum k value for NDCG computation
        idcg_denoms: Precomputed 1/log2(i+1) tensor of shape (max_gt,)
        device: Target device

    Returns:
        (batch_size,) tensor of IDCG values
    """
    # Compute cumulative sum of idcg_denoms: idcg_cumsum[i] = sum(idcg_denoms[0:i+1])
    idcg_cumsum = torch.cumsum(idcg_denoms, dim=0)  # (max_gt,)

    # Clamp gt_sizes to k and get effective indices (0-indexed: need -1)
    # For gt_size=3 and k=5, we want idcg_cumsum[2] (sum of first 3 elements)
    # For gt_size=0, we want 0 (no relevant docs)
    effective_sizes = torch.clamp(gt_sizes.long(), min=0, max=k)

    # Handle zero case: where effective_sizes is 0, result should be 0
    # Otherwise, index into cumsum at position effective_sizes - 1
    idcg = torch.where(
        effective_sizes > 0,
        idcg_cumsum[torch.clamp(effective_sizes - 1, min=0)],
        torch.zeros_like(gt_sizes, dtype=torch.float32)
    )

    return idcg


class MetricRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn):
            cls._registry[name] = fn
            return fn
        return decorator

    @classmethod
    def get(cls, name: str):
        return cls._registry[name]

    @classmethod
    def all(cls):
        return cls._registry
class MetricFunctions:
    """
    Library of metric functions.
    Each takes (retrieved_ids, ground_truth_ids) and returns a score.
    """
    
    @staticmethod
    @MetricRegistry.register("Hit@K")
    def hit_at_k(k):
        def fn(retrieved_ids, gt_ids, **_):
            return float(bool(set(retrieved_ids[:k]) & set(gt_ids)))
        return fn

    @staticmethod
    @MetricRegistry.register("Recall@K")
    def recall_at_k(k):
        def fn(retrieved_ids, gt_ids, **_):
            if not gt_ids:
                return 0.0
            return len(set(retrieved_ids[:k]) & set(gt_ids)) / len(gt_ids)
        return fn

    @staticmethod
    @MetricRegistry.register("MRR")
    def mrr(retrieved_ids, gt_ids, **_):
        gt = set(gt_ids)
        for i, rid in enumerate(retrieved_ids):
            if rid in gt:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    @MetricRegistry.register("NDCG@K")
    def ndcg_at_k(k):
        def fn(retrieved_ids, gt_ids, **_):
            if not gt_ids:
                return 0.0
            gt = set(gt_ids)
            # Use precomputed log denominators to avoid tensor creation
            dcg = sum(
                (1.0 if rid in gt else 0.0) / _LOG_DENOMS_PYTHON[i]
                for i, rid in enumerate(retrieved_ids[:k])
            )
            idcg = sum(1.0 / _LOG_DENOMS_PYTHON[i] for i in range(min(len(gt), k)))
            return dcg / idcg if idcg > 0 else 0.0
        return fn

    @staticmethod
    @MetricRegistry.register("DiversityCoverage@K")
    def diversity_coverage_at_k(k):
        def fn(retrieved_ids, gt_ids, retrieved_nodes, **kwargs):
            rel_ids = set(retrieved_ids[:k]) & set(gt_ids)
            if not rel_ids:
                return 0.0
            rel_nodes = [n for n in retrieved_nodes if str(n["id"]) in rel_ids]
            cats = {n.get("node_type", "unknown") for n in rel_nodes}
            gt_cats = {
                n.get("node_type", "unknown")
                for n in retrieved_nodes
                if str(n["id"]) in gt_ids
            }
            return len(cats) / len(gt_cats) if gt_cats else 0.0
        return fn

    @staticmethod
    @MetricRegistry.register("IntentEntropy@K")
    def intent_entropy_at_k(k):
        def fn(retrieved_ids, _, retrieved_nodes, **kwargs):
            nodes = retrieved_nodes[:k]
            if not nodes:
                return 0.0
            counts = Counter(n.get("node_type", "unknown") for n in nodes)
            probs = torch.as_tensor(list(counts.values()), dtype=torch.float32) / sum(counts.values())
            entropy = -torch.sum(probs * torch.log2(probs)).item()
            # Use math.log2 for scalar instead of creating tensor
            return entropy / math.log2(len(counts)) if len(counts) > 1 else 0.0
        return fn

    # --- GRADED METRICS (Better gradient signal for evolution) ---

    @staticmethod
    @MetricRegistry.register("SoftHit@K")
    def soft_hit_at_k(k):
        """
        Returns 1/(position+1) for first hit, providing position gradient.
        Unlike binary Hit@K, this rewards earlier positions more.

        Score = 1/(i+1) where i is 0-indexed position of first hit
        - Position 0 (top-1): 1.0
        - Position 1: 0.5
        - Position 2: 0.333
        - Position 4: 0.2
        - No hit: 0.0
        """
        def fn(retrieved_ids, gt_ids, **_):
            gt = set(gt_ids)
            for i, rid in enumerate(retrieved_ids[:k]):
                if rid in gt:
                    return 1.0 / (i + 1)
            return 0.0
        return fn

    @staticmethod
    @MetricRegistry.register("GradedRecall@K")
    def graded_recall_at_k(k):
        """
        Position-weighted recall: earlier positions contribute more.
        Uses harmonic weights (1/1, 1/2, 1/3, ...) to prioritize top positions.

        This provides gradient signal even when recall count is the same,
        by rewarding solutions that place hits earlier.
        """
        def fn(retrieved_ids, gt_ids, **_):
            if not gt_ids:
                return 0.0
            gt = set(gt_ids)
            # Harmonic weights: position i gets weight 1/(i+1)
            weights = [1.0 / (i + 1) for i in range(k)]
            max_possible = sum(weights[:min(len(gt), k)])
            score = sum(weights[i] for i, rid in enumerate(retrieved_ids[:k]) if rid in gt)
            return score / max_possible if max_possible > 0 else 0.0
        return fn

    @staticmethod
    @MetricRegistry.register("DCG@K")
    def dcg_at_k(k):
        """
        Discounted Cumulative Gain without normalization.
        Useful as a raw signal that doesn't require knowing ideal ranking.

        DCG = sum(rel_i / log2(i+2)) for i in 0..k-1
        where rel_i = 1 if retrieved_ids[i] in ground_truth, else 0
        """
        def fn(retrieved_ids, gt_ids, **_):
            if not gt_ids:
                return 0.0
            gt = set(gt_ids)
            # Use precomputed log denominators to avoid tensor creation
            dcg = sum(
                (1.0 if rid in gt else 0.0) / _LOG_DENOMS_PYTHON[i]
                for i, rid in enumerate(retrieved_ids[:k])
            )
            return dcg
        return fn

    @staticmethod
    @MetricRegistry.register("ExpDecayHit@K")
    def exp_decay_hit_at_k(k, decay: float = 0.8):
        """
        Exponentially decaying hit score based on position.
        Provides smoother gradient than 1/(position+1).

        Score = decay^position for first hit
        - Position 0: 1.0
        - Position 1: 0.8
        - Position 2: 0.64
        - Position 4: 0.41
        - No hit: 0.0
        """
        def fn(retrieved_ids, gt_ids, **_):
            gt = set(gt_ids)
            for i, rid in enumerate(retrieved_ids[:k]):
                if rid in gt:
                    return decay ** i
            return 0.0
        return fn

    # =========================================================================
    # BATCH VECTORIZED METRICS - Optimized for computing metrics over many queries
    # =========================================================================

    @staticmethod
    def mrr_batch(retrieved_ids_batch: torch.Tensor, gt_ids_batch: List[set]) -> torch.Tensor:
        """
        Vectorized MRR for batch of queries.

        Args:
            retrieved_ids_batch: (batch_size, max_k) tensor of retrieved IDs
            gt_ids_batch: List of sets of ground truth IDs

        Returns:
            (batch_size,) tensor of MRR scores
        """
        batch_size = len(gt_ids_batch)
        max_k = retrieved_ids_batch.shape[1] if retrieved_ids_batch.ndim > 1 else 1
        scores = torch.zeros(batch_size, dtype=torch.float32)

        # Convert entire tensor to list once to avoid per-element .item() calls
        retrieved_ids_list = retrieved_ids_batch.tolist()

        for i, gt_set in enumerate(gt_ids_batch):
            if not gt_set:
                continue
            # Find first hit position using pre-converted list
            row = retrieved_ids_list[i]
            for pos in range(max_k):
                if row[pos] in gt_set:
                    scores[i] = 1.0 / (pos + 1)
                    break

        return scores

    @staticmethod
    def hit_at_k_batch(k: int, retrieved_ids_batch: torch.Tensor, gt_ids_batch: List[set]) -> torch.Tensor:
        """
        Vectorized Hit@K for batch of queries.

        Args:
            k: Number of top results to consider
            retrieved_ids_batch: (batch_size, max_k) tensor of retrieved IDs
            gt_ids_batch: List of sets of ground truth IDs

        Returns:
            (batch_size,) tensor of Hit@K scores (0.0 or 1.0)
        """
        batch_size = len(gt_ids_batch)
        scores = torch.zeros(batch_size, dtype=torch.float32)

        # Convert tensor to list once to avoid per-row .tolist() calls
        retrieved_ids_list = retrieved_ids_batch[:, :k].tolist()

        for i, gt_set in enumerate(gt_ids_batch):
            if not gt_set:
                continue
            # Check if any of top-k is in ground truth
            if set(retrieved_ids_list[i]) & gt_set:
                scores[i] = 1.0

        return scores

    @staticmethod
    def recall_at_k_batch(k: int, retrieved_ids_batch: torch.Tensor, gt_ids_batch: List[set]) -> torch.Tensor:
        """
        Vectorized Recall@K for batch of queries.

        Args:
            k: Number of top results to consider
            retrieved_ids_batch: (batch_size, max_k) tensor of retrieved IDs
            gt_ids_batch: List of sets of ground truth IDs

        Returns:
            (batch_size,) tensor of Recall@K scores
        """
        batch_size = len(gt_ids_batch)
        scores = torch.zeros(batch_size, dtype=torch.float32)

        # Convert tensor to list once to avoid per-row .tolist() calls
        retrieved_ids_list = retrieved_ids_batch[:, :k].tolist()

        for i, gt_set in enumerate(gt_ids_batch):
            if not gt_set:
                continue
            top_k_ids = set(retrieved_ids_list[i])
            scores[i] = len(top_k_ids & gt_set) / len(gt_set)

        return scores

    @staticmethod
    def ndcg_at_k_batch(k: int, retrieved_ids_batch: torch.Tensor, gt_ids_batch: List[set]) -> torch.Tensor:
        """
        Vectorized NDCG@K for batch of queries.

        Args:
            k: Number of top results to consider
            retrieved_ids_batch: (batch_size, max_k) tensor of retrieved IDs
            gt_ids_batch: List of sets of ground truth IDs

        Returns:
            (batch_size,) tensor of NDCG@K scores
        """
        batch_size = len(gt_ids_batch)
        scores = torch.zeros(batch_size, dtype=torch.float32)

        # Convert entire tensor to list once to avoid per-element .item() calls
        retrieved_ids_list = retrieved_ids_batch.tolist()
        max_pos = min(k, retrieved_ids_batch.shape[1])

        # Precompute IDCG values for each possible gt_size
        idcg_cache = {}
        for gt_set in gt_ids_batch:
            n = min(len(gt_set), k)
            if n > 0 and n not in idcg_cache:
                idcg_cache[n] = sum(1.0 / _LOG_DENOMS_PYTHON[j] for j in range(n))

        for i, gt_set in enumerate(gt_ids_batch):
            if not gt_set:
                continue

            # Compute DCG using precomputed log denominators
            row = retrieved_ids_list[i]
            dcg = sum(
                1.0 / _LOG_DENOMS_PYTHON[pos]
                for pos in range(max_pos)
                if row[pos] in gt_set
            )

            # Get precomputed IDCG
            n_relevant = min(len(gt_set), k)
            idcg = idcg_cache.get(n_relevant, 0.0)

            scores[i] = dcg / idcg if idcg > 0 else 0.0

        return scores

    @staticmethod
    def compute_all_metrics_batch(
        retrieved_ids_batch: torch.Tensor,
        gt_ids_batch: List[set],
        k_values: List[int] = None
    ) -> dict:
        """
        Compute all standard metrics for a batch of queries in one pass.

        This is more efficient than calling individual metric functions
        because it reuses the iteration over queries.

        Args:
            retrieved_ids_batch: (batch_size, max_k) tensor of retrieved IDs
            gt_ids_batch: List of sets of ground truth IDs
            k_values: List of k values for Hit@K and Recall@K (default: [1, 5, 10, 20])

        Returns:
            Dictionary of metric_name -> mean_score
        """
        if k_values is None:
            k_values = [1, 5, 10, 20]

        batch_size = len(gt_ids_batch)
        max_k = retrieved_ids_batch.shape[1] if retrieved_ids_batch.ndim > 1 else 1
        max_k_val = max(k_values)
        k_values_set = set(k_values)

        # Convert entire tensor to list once to avoid per-element .item() calls
        retrieved_ids_list = retrieved_ids_batch.tolist()

        # Precompute IDCG values for each (n_gt, k) combination needed
        idcg_cache = {}
        for gt_set in gt_ids_batch:
            n_gt = len(gt_set)
            for k in k_values:
                key = (n_gt, k)
                if key not in idcg_cache:
                    n = min(n_gt, k)
                    idcg_cache[key] = sum(1.0 / _LOG_DENOMS_PYTHON[j] for j in range(n)) if n > 0 else 0.0

        # Initialize score tensors
        mrr_scores = torch.zeros(batch_size, dtype=torch.float32)
        hit_scores = {k: torch.zeros(batch_size, dtype=torch.float32) for k in k_values}
        recall_scores = {k: torch.zeros(batch_size, dtype=torch.float32) for k in k_values}
        ndcg_scores = {k: torch.zeros(batch_size, dtype=torch.float32) for k in k_values}

        # Single pass over all queries using precomputed values
        for i, gt_set in enumerate(gt_ids_batch):
            if not gt_set:
                continue

            n_gt = len(gt_set)
            first_hit_found = False
            hits_so_far = 0
            dcg_so_far = 0.0
            row = retrieved_ids_list[i]

            # Process positions up to max needed k
            for pos in range(min(max_k_val, max_k)):
                is_hit = row[pos] in gt_set

                if is_hit:
                    hits_so_far += 1
                    dcg_so_far += 1.0 / _LOG_DENOMS_PYTHON[pos]

                    if not first_hit_found:
                        mrr_scores[i] = 1.0 / (pos + 1)
                        first_hit_found = True

                # Update metrics at each k boundary
                k_pos = pos + 1
                if k_pos in k_values_set:
                    hit_scores[k_pos][i] = 1.0 if hits_so_far > 0 else 0.0
                    recall_scores[k_pos][i] = hits_so_far / n_gt
                    # NDCG using precomputed IDCG
                    idcg = idcg_cache[(n_gt, k_pos)]
                    ndcg_scores[k_pos][i] = dcg_so_far / idcg if idcg > 0 else 0.0

        # Aggregate results
        results = {
            'MRR': float(mrr_scores.mean().item())
        }

        for k in k_values:
            results[f'Hit@{k}'] = float(hit_scores[k].mean().item())
            results[f'Recall@{k}'] = float(recall_scores[k].mean().item())
            results[f'NDCG@{k}'] = float(ndcg_scores[k].mean().item())

        return results

    @staticmethod
    def _create_gt_tensor(gt_ids_batch: List[Set[int]], device: str) -> torch.Tensor:
        """
        Convert ground truth sets to padded tensor for vectorized membership testing.

        Args:
            gt_ids_batch: List of sets of ground truth IDs
            device: Device string

        Returns:
            (batch_size, max_gt_size) tensor padded with -1
        """
        batch_size = len(gt_ids_batch)
        if batch_size == 0:
            return torch.empty((0, 0), device=device, dtype=torch.long)

        max_gt_size = max(len(gt) for gt in gt_ids_batch) if gt_ids_batch else 0
        if max_gt_size == 0:
            return torch.full((batch_size, 1), -1, device=device, dtype=torch.long)

        # Create padded tensor
        gt_tensor = torch.full((batch_size, max_gt_size), -1, device=device, dtype=torch.long)
        for i, gt_set in enumerate(gt_ids_batch):
            gt_list = list(gt_set)
            gt_tensor[i, :len(gt_list)] = torch.as_tensor(gt_list, device=device, dtype=torch.long)

        return gt_tensor

    @staticmethod
    def _vectorized_membership(
        retrieved: torch.Tensor,
        gt_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized membership testing: check if retrieved IDs are in ground truth.

        Uses broadcasting for GPU-parallel comparison instead of Python loops.

        Args:
            retrieved: (batch_size, k) tensor of retrieved IDs
            gt_tensor: (batch_size, max_gt_size) tensor of ground truth IDs (padded with -1)

        Returns:
            (batch_size, k) boolean tensor indicating relevance
        """
        # retrieved: (B, K), gt_tensor: (B, G)
        # Expand for broadcasting: (B, K, 1) vs (B, 1, G) -> (B, K, G)
        retrieved_expanded = retrieved.unsqueeze(-1)  # (B, K, 1)
        gt_expanded = gt_tensor.unsqueeze(1)  # (B, 1, G)

        # Check equality and reduce: any match across G dimension
        matches = (retrieved_expanded == gt_expanded).any(dim=-1)  # (B, K)

        # Exclude padding matches (retrieved == -1 should not match gt == -1)
        matches = matches & (retrieved >= 0)

        return matches

    @staticmethod
    def compute_all_metrics_batch_gpu(
        retrieved_ids_batch: torch.Tensor,
        gt_ids_batch: List[Set[int]],
        k_values: List[int] = None,
        device: str = "cuda",
        return_per_query: bool = False
    ) -> dict:
        """
        GPU-accelerated batch metric computation using PyTorch.

        Vectorized implementation that leverages GPU parallelism for
        computing metrics across many queries simultaneously.

        Args:
            retrieved_ids_batch: (batch_size, max_k) tensor of retrieved IDs on GPU
            gt_ids_batch: List of sets of ground truth IDs (on CPU)
            k_values: List of k values for Hit@K and Recall@K (default: [1, 5, 10, 20])
            device: Device string ("cuda" or "cpu")
            return_per_query: If True, also return per-query score tensors for variance computation

        Returns:
            Dictionary of metric_name -> mean_score
            If return_per_query=True, also includes 'per_query_<metric>': tensor entries
        """
        if k_values is None:
            k_values = [1, 5, 10, 20]

        # Ensure tensor is on the correct device
        if not isinstance(retrieved_ids_batch, torch.Tensor):
            retrieved_ids_batch = torch.as_tensor(retrieved_ids_batch, device=device, dtype=torch.long)
        elif str(retrieved_ids_batch.device) != device:
            retrieved_ids_batch = retrieved_ids_batch.to(device)

        batch_size = len(gt_ids_batch)
        max_k = retrieved_ids_batch.shape[1] if retrieved_ids_batch.ndim > 1 else 1
        max_k_val = max(k_values)

        # Use cached log denominators for NDCG on GPU
        log_denoms = _get_log_denoms(max_k_val, device)

        # Initialize score tensors on GPU
        mrr_scores = torch.zeros(batch_size, dtype=torch.float32, device=device)
        hit_scores = {k: torch.zeros(batch_size, dtype=torch.float32, device=device) for k in k_values}
        recall_scores = {k: torch.zeros(batch_size, dtype=torch.float32, device=device) for k in k_values}
        ndcg_scores = {k: torch.zeros(batch_size, dtype=torch.float32, device=device) for k in k_values}

        # Process each query (gt_sets are on CPU, but metric computation is on GPU)
        for i, gt_set in enumerate(gt_ids_batch):
            if not gt_set:
                continue

            n_gt = len(gt_set)
            # Create relevance mask for this query's retrieved items
            retrieved_row = retrieved_ids_batch[i, :min(max_k_val, max_k)]

            # Check membership - this is the bottleneck with sets on CPU
            # But we vectorize the score computation on GPU
            relevance = torch.as_tensor(
                [int(rid.item()) in gt_set for rid in retrieved_row],
                dtype=torch.float32,
                device=device
            )

            # MRR - find first hit position
            hit_positions = torch.where(relevance > 0)[0]
            if len(hit_positions) > 0:
                mrr_scores[i] = 1.0 / (hit_positions[0].item() + 1)

            # Compute cumulative hits and DCG for each k
            cumulative_hits = torch.cumsum(relevance, dim=0)
            dcg_contributions = relevance / log_denoms[:len(relevance)]
            cumulative_dcg = torch.cumsum(dcg_contributions, dim=0)

            for k in k_values:
                if k <= len(relevance):
                    # Hit@K
                    hit_scores[k][i] = 1.0 if cumulative_hits[k - 1] > 0 else 0.0
                    # Recall@K
                    recall_scores[k][i] = cumulative_hits[k - 1] / n_gt
                    # NDCG@K
                    idcg = torch.sum(1.0 / log_denoms[:min(n_gt, k)])
                    ndcg_scores[k][i] = cumulative_dcg[k - 1] / idcg if idcg > 0 else 0.0

        # Aggregate results (move back to CPU for final mean)
        results = {
            'MRR': float(mrr_scores.mean().item())
        }

        for k in k_values:
            results[f'Hit@{k}'] = float(hit_scores[k].mean().item())
            results[f'Recall@{k}'] = float(recall_scores[k].mean().item())
            results[f'NDCG@{k}'] = float(ndcg_scores[k].mean().item())

        # Include per-query scores if requested (for variance computation)
        if return_per_query:
            results['per_query_MRR'] = mrr_scores.cpu()
            for k in k_values:
                results[f'per_query_Hit@{k}'] = hit_scores[k].cpu()
                results[f'per_query_Recall@{k}'] = recall_scores[k].cpu()
                results[f'per_query_NDCG@{k}'] = ndcg_scores[k].cpu()

        return results

    @staticmethod
    def compute_all_metrics_batch_gpu_vectorized(
        retrieved_ids_batch: torch.Tensor,
        gt_ids_batch: List[Set[int]],
        k_values: List[int] = None,
        device: str = "cuda"
    ) -> dict:
        """
        Fully vectorized GPU batch metric computation.

        Uses tensor broadcasting for membership testing instead of Python loops.
        This provides maximum GPU utilization for large batches.

        Args:
            retrieved_ids_batch: (batch_size, max_k) tensor of retrieved IDs on GPU
            gt_ids_batch: List of sets of ground truth IDs (on CPU)
            k_values: List of k values for Hit@K and Recall@K (default: [1, 5, 10, 20])
            device: Device string ("cuda" or "cpu")

        Returns:
            Dictionary of metric_name -> mean_score
        """
        if k_values is None:
            k_values = [1, 5, 10, 20]

        # Ensure tensor is on the correct device
        if not isinstance(retrieved_ids_batch, torch.Tensor):
            retrieved_ids_batch = torch.as_tensor(retrieved_ids_batch, device=device, dtype=torch.long)
        elif str(retrieved_ids_batch.device) != device:
            retrieved_ids_batch = retrieved_ids_batch.to(device)

        batch_size = len(gt_ids_batch)
        max_k = retrieved_ids_batch.shape[1] if retrieved_ids_batch.ndim > 1 else 1
        max_k_val = max(k_values)

        # Truncate to max_k_val for efficiency
        if max_k > max_k_val:
            retrieved_ids_batch = retrieved_ids_batch[:, :max_k_val]
            max_k = max_k_val

        with torch.no_grad():
            # Convert ground truth to padded tensor
            gt_tensor = MetricFunctions._create_gt_tensor(gt_ids_batch, device)
            gt_sizes = torch.as_tensor([len(gt) for gt in gt_ids_batch], device=device, dtype=torch.float32)

            # Vectorized membership testing: (batch_size, max_k) boolean mask
            relevance = MetricFunctions._vectorized_membership(retrieved_ids_batch, gt_tensor).float()

            # Use cached log denominators for NDCG
            log_denoms = _get_log_denoms(max_k, device)

            # Cumulative metrics: (batch_size, max_k)
            cumulative_hits = torch.cumsum(relevance, dim=1)
            dcg_contributions = relevance / log_denoms[:max_k]
            cumulative_dcg = torch.cumsum(dcg_contributions, dim=1)

            # MRR: find first hit position
            # Create position indices and mask for first hit
            first_hit_mask = relevance.cumsum(dim=1) == 1
            first_hit_mask = first_hit_mask & (relevance > 0)
            positions = torch.arange(1, max_k + 1, device=device, dtype=torch.float32).unsqueeze(0)
            mrr_values = torch.where(first_hit_mask, 1.0 / positions, torch.zeros_like(positions))
            mrr_scores = mrr_values.sum(dim=1)  # Sum across positions (only one will be non-zero)

            # Compute IDCG denominators using cached function
            max_gt = gt_tensor.shape[1]
            idcg_denoms = _get_idcg_denoms(max_gt, device)

            results = {
                'MRR': float(mrr_scores.mean().item())
            }

            for k in k_values:
                if k <= max_k:
                    # Hit@K: any hit in top-k
                    hit_k = (cumulative_hits[:, k - 1] > 0).float()
                    results[f'Hit@{k}'] = float(hit_k.mean().item())

                    # Recall@K: hits / gt_size
                    recall_k = cumulative_hits[:, k - 1] / (gt_sizes + 1e-10)
                    results[f'Recall@{k}'] = float(recall_k.mean().item())

                    # NDCG@K: DCG / IDCG - vectorized without .item() calls
                    idcg_k = _compute_vectorized_idcg(gt_sizes, k, idcg_denoms, device)

                    ndcg_k = cumulative_dcg[:, k - 1] / (idcg_k + 1e-10)
                    results[f'NDCG@{k}'] = float(ndcg_k.mean().item())
                else:
                    results[f'Hit@{k}'] = 0.0
                    results[f'Recall@{k}'] = 0.0
                    results[f'NDCG@{k}'] = 0.0

        return results

    @staticmethod
    def compute_metrics_at_tier_endpoints(
        retrieved_ids_batch: torch.Tensor,
        gt_tensor: torch.Tensor,
        gt_sizes: torch.Tensor,
        tier_endpoints: List[int],
        k_values: List[int] = None,
        device: str = "cuda"
    ) -> Dict[int, dict]:
        """
        Compute metrics at multiple tier endpoints in a single GPU pass.

        This enables early-exit tier decisions without additional GPU calls by
        computing cumulative metrics at each tier boundary (e.g., 20, 50, 100 queries).

        Args:
            retrieved_ids_batch: (n_queries, max_k) tensor of retrieved IDs on GPU
            gt_tensor: (n_queries, max_gt_size) pre-computed ground truth tensor on GPU
            gt_sizes: (n_queries,) tensor of ground truth sizes on GPU
            tier_endpoints: List of query counts to compute metrics at [20, 50, 100]
            k_values: List of k values for Hit@K and Recall@K (default: [1, 5, 10, 20])
            device: Device string ("cuda" or "cpu")

        Returns:
            Dictionary mapping tier_endpoint -> metrics_dict
        """
        if k_values is None:
            k_values = [1, 5, 10, 20]

        n_queries = retrieved_ids_batch.shape[0]
        max_k = retrieved_ids_batch.shape[1] if retrieved_ids_batch.ndim > 1 else 1
        max_k_val = max(k_values)

        # Truncate to max_k_val for efficiency
        if max_k > max_k_val:
            retrieved_ids_batch = retrieved_ids_batch[:, :max_k_val]
            max_k = max_k_val

        results_by_tier = {}

        with torch.no_grad():
            # Vectorized membership testing for all queries at once
            retrieved_expanded = retrieved_ids_batch.unsqueeze(-1)  # (N, K, 1)
            gt_expanded = gt_tensor.unsqueeze(1)  # (N, 1, G)

            relevance = (retrieved_expanded == gt_expanded).any(dim=-1)  # (N, K)
            relevance = relevance & (retrieved_ids_batch >= 0)
            relevance = relevance.float()

            # Use cached log denominators for NDCG
            log_denoms = _get_log_denoms(max_k, device)

            # Cumulative metrics per query
            cumulative_hits = torch.cumsum(relevance, dim=1)  # (N, K)
            dcg_contributions = relevance / log_denoms[:max_k]
            cumulative_dcg = torch.cumsum(dcg_contributions, dim=1)  # (N, K)

            # MRR per query
            first_hit_mask = relevance.cumsum(dim=1) == 1
            first_hit_mask = first_hit_mask & (relevance > 0)
            positions = torch.arange(1, max_k + 1, device=device, dtype=torch.float32).unsqueeze(0)
            mrr_values = torch.where(first_hit_mask, 1.0 / positions, torch.zeros_like(positions))
            mrr_per_query = mrr_values.sum(dim=1)  # (N,)

            # Use cached IDCG denominators
            max_gt = gt_tensor.shape[1]
            idcg_denoms = _get_idcg_denoms(max_gt, device)

            # Compute metrics at each tier endpoint
            for tier_end in tier_endpoints:
                if tier_end > n_queries:
                    tier_end = n_queries

                if tier_end == 0:
                    results_by_tier[tier_end] = {}
                    continue

                # Slice to tier endpoint
                tier_mrr = mrr_per_query[:tier_end]
                tier_gt_sizes = gt_sizes[:tier_end]
                tier_cumulative_hits = cumulative_hits[:tier_end]
                tier_cumulative_dcg = cumulative_dcg[:tier_end]

                tier_metrics = {
                    'MRR': float(tier_mrr.mean().item())
                }

                for k in k_values:
                    if k <= max_k:
                        # Hit@K: any hit in top-k
                        hit_k = (tier_cumulative_hits[:, k - 1] > 0).float()
                        tier_metrics[f'Hit@{k}'] = float(hit_k.mean().item())

                        # Recall@K: hits / gt_size
                        recall_k = tier_cumulative_hits[:, k - 1] / (tier_gt_sizes + 1e-10)
                        tier_metrics[f'Recall@{k}'] = float(recall_k.mean().item())

                        # NDCG@K - vectorized without .item() calls
                        idcg_k = _compute_vectorized_idcg(tier_gt_sizes, k, idcg_denoms, device)

                        ndcg_k = tier_cumulative_dcg[:, k - 1] / (idcg_k + 1e-10)
                        tier_metrics[f'NDCG@{k}'] = float(ndcg_k.mean().item())
                    else:
                        tier_metrics[f'Hit@{k}'] = 0.0
                        tier_metrics[f'Recall@{k}'] = 0.0
                        tier_metrics[f'NDCG@{k}'] = 0.0

                results_by_tier[tier_end] = tier_metrics

        return results_by_tier

    @staticmethod
    def compute_all_metrics_batch_gpu_precomputed(
        retrieved_ids_batch: torch.Tensor,
        gt_tensor: torch.Tensor,
        gt_sizes: torch.Tensor,
        k_values: List[int] = None,
        device: str = "cuda"
    ) -> dict:
        """
        GPU batch metric computation with pre-computed ground truth tensors.

        This is the fastest path when ground truth has been pre-computed as GPU
        tensors, eliminating repeated CPU-to-GPU transfers during evolution.

        Args:
            retrieved_ids_batch: (batch_size, max_k) tensor of retrieved IDs on GPU
            gt_tensor: (batch_size, max_gt_size) pre-computed ground truth tensor on GPU
            gt_sizes: (batch_size,) tensor of ground truth sizes on GPU
            k_values: List of k values for Hit@K and Recall@K (default: [1, 5, 10, 20])
            device: Device string ("cuda" or "cpu")

        Returns:
            Dictionary of metric_name -> mean_score
        """
        if k_values is None:
            k_values = [1, 5, 10, 20]

        # Ensure tensors are on the correct device
        if str(retrieved_ids_batch.device) != device:
            retrieved_ids_batch = retrieved_ids_batch.to(device)
        if str(gt_tensor.device) != device:
            gt_tensor = gt_tensor.to(device)
        if str(gt_sizes.device) != device:
            gt_sizes = gt_sizes.to(device)

        batch_size = retrieved_ids_batch.shape[0]
        max_k = retrieved_ids_batch.shape[1] if retrieved_ids_batch.ndim > 1 else 1
        max_k_val = max(k_values)

        # Truncate to max_k_val for efficiency
        if max_k > max_k_val:
            retrieved_ids_batch = retrieved_ids_batch[:, :max_k_val]
            max_k = max_k_val

        with torch.no_grad():
            # Vectorized membership testing using pre-computed gt_tensor
            # retrieved: (B, K), gt_tensor: (B, G)
            # Expand for broadcasting: (B, K, 1) vs (B, 1, G) -> (B, K, G)
            retrieved_expanded = retrieved_ids_batch.unsqueeze(-1)  # (B, K, 1)
            gt_expanded = gt_tensor.unsqueeze(1)  # (B, 1, G)

            # Check equality and reduce: any match across G dimension
            relevance = (retrieved_expanded == gt_expanded).any(dim=-1)  # (B, K)
            # Exclude padding matches
            relevance = relevance & (retrieved_ids_batch >= 0)
            relevance = relevance.float()

            # Use cached log denominators for NDCG
            log_denoms = _get_log_denoms(max_k, device)

            # Cumulative metrics: (batch_size, max_k)
            cumulative_hits = torch.cumsum(relevance, dim=1)
            dcg_contributions = relevance / log_denoms[:max_k]
            cumulative_dcg = torch.cumsum(dcg_contributions, dim=1)

            # MRR: find first hit position
            first_hit_mask = relevance.cumsum(dim=1) == 1
            first_hit_mask = first_hit_mask & (relevance > 0)
            positions = torch.arange(1, max_k + 1, device=device, dtype=torch.float32).unsqueeze(0)
            mrr_values = torch.where(first_hit_mask, 1.0 / positions, torch.zeros_like(positions))
            mrr_scores = mrr_values.sum(dim=1)

            # Use cached IDCG denominators
            max_gt = gt_tensor.shape[1]
            idcg_denoms = _get_idcg_denoms(max_gt, device)

            results = {
                'MRR': float(mrr_scores.mean().item())
            }

            for k in k_values:
                if k <= max_k:
                    # Hit@K: any hit in top-k
                    hit_k = (cumulative_hits[:, k - 1] > 0).float()
                    results[f'Hit@{k}'] = float(hit_k.mean().item())

                    # Recall@K: hits / gt_size
                    recall_k = cumulative_hits[:, k - 1] / (gt_sizes + 1e-10)
                    results[f'Recall@{k}'] = float(recall_k.mean().item())

                    # NDCG@K: DCG / IDCG - vectorized without .item() calls
                    idcg_k = _compute_vectorized_idcg(gt_sizes, k, idcg_denoms, device)

                    ndcg_k = cumulative_dcg[:, k - 1] / (idcg_k + 1e-10)
                    results[f'NDCG@{k}'] = float(ndcg_k.mean().item())
                else:
                    results[f'Hit@{k}'] = 0.0
                    results[f'Recall@{k}'] = 0.0
                    results[f'NDCG@{k}'] = 0.0

        return results