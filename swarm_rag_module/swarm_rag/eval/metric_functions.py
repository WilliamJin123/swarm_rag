import numpy as np
from typing import Callable, List
from collections import Counter

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
            dcg = sum(
                (1.0 if rid in gt else 0.0) / np.log2(i + 2)
                for i, rid in enumerate(retrieved_ids[:k])
            )
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt), k)))
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
            probs = np.array(list(counts.values())) / sum(counts.values())
            entropy = -np.sum(probs * np.log2(probs))
            return entropy / np.log2(len(counts)) if len(counts) > 1 else 0.0
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
            dcg = sum(
                (1.0 if rid in gt else 0.0) / np.log2(i + 2)
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
    def mrr_batch(retrieved_ids_batch: np.ndarray, gt_ids_batch: List[set]) -> np.ndarray:
        """
        Vectorized MRR for batch of queries.

        Args:
            retrieved_ids_batch: (batch_size, max_k) array of retrieved IDs
            gt_ids_batch: List of sets of ground truth IDs

        Returns:
            (batch_size,) array of MRR scores
        """
        batch_size = len(gt_ids_batch)
        max_k = retrieved_ids_batch.shape[1] if retrieved_ids_batch.ndim > 1 else 1
        scores = np.zeros(batch_size, dtype=np.float32)

        for i, gt_set in enumerate(gt_ids_batch):
            if not gt_set:
                continue
            # Find first hit position
            for pos in range(max_k):
                if retrieved_ids_batch[i, pos] in gt_set:
                    scores[i] = 1.0 / (pos + 1)
                    break

        return scores

    @staticmethod
    def hit_at_k_batch(k: int, retrieved_ids_batch: np.ndarray, gt_ids_batch: List[set]) -> np.ndarray:
        """
        Vectorized Hit@K for batch of queries.

        Args:
            k: Number of top results to consider
            retrieved_ids_batch: (batch_size, max_k) array of retrieved IDs
            gt_ids_batch: List of sets of ground truth IDs

        Returns:
            (batch_size,) array of Hit@K scores (0.0 or 1.0)
        """
        batch_size = len(gt_ids_batch)
        scores = np.zeros(batch_size, dtype=np.float32)

        for i, gt_set in enumerate(gt_ids_batch):
            if not gt_set:
                continue
            # Check if any of top-k is in ground truth
            top_k_ids = set(retrieved_ids_batch[i, :k].tolist())
            if top_k_ids & gt_set:
                scores[i] = 1.0

        return scores

    @staticmethod
    def recall_at_k_batch(k: int, retrieved_ids_batch: np.ndarray, gt_ids_batch: List[set]) -> np.ndarray:
        """
        Vectorized Recall@K for batch of queries.

        Args:
            k: Number of top results to consider
            retrieved_ids_batch: (batch_size, max_k) array of retrieved IDs
            gt_ids_batch: List of sets of ground truth IDs

        Returns:
            (batch_size,) array of Recall@K scores
        """
        batch_size = len(gt_ids_batch)
        scores = np.zeros(batch_size, dtype=np.float32)

        for i, gt_set in enumerate(gt_ids_batch):
            if not gt_set:
                continue
            top_k_ids = set(retrieved_ids_batch[i, :k].tolist())
            scores[i] = len(top_k_ids & gt_set) / len(gt_set)

        return scores

    @staticmethod
    def ndcg_at_k_batch(k: int, retrieved_ids_batch: np.ndarray, gt_ids_batch: List[set]) -> np.ndarray:
        """
        Vectorized NDCG@K for batch of queries.

        Args:
            k: Number of top results to consider
            retrieved_ids_batch: (batch_size, max_k) array of retrieved IDs
            gt_ids_batch: List of sets of ground truth IDs

        Returns:
            (batch_size,) array of NDCG@K scores
        """
        batch_size = len(gt_ids_batch)
        scores = np.zeros(batch_size, dtype=np.float32)

        # Pre-compute log denominators
        log_denoms = np.log2(np.arange(2, k + 2, dtype=np.float32))

        for i, gt_set in enumerate(gt_ids_batch):
            if not gt_set:
                continue

            # Compute DCG
            dcg = 0.0
            for pos in range(min(k, retrieved_ids_batch.shape[1])):
                if retrieved_ids_batch[i, pos] in gt_set:
                    dcg += 1.0 / log_denoms[pos]

            # Compute ideal DCG
            n_relevant = len(gt_set)
            idcg = np.sum(1.0 / log_denoms[:min(n_relevant, k)])

            scores[i] = dcg / idcg if idcg > 0 else 0.0

        return scores

    @staticmethod
    def compute_all_metrics_batch(
        retrieved_ids_batch: np.ndarray,
        gt_ids_batch: List[set],
        k_values: List[int] = None
    ) -> dict:
        """
        Compute all standard metrics for a batch of queries in one pass.

        This is more efficient than calling individual metric functions
        because it reuses the iteration over queries.

        Args:
            retrieved_ids_batch: (batch_size, max_k) array of retrieved IDs
            gt_ids_batch: List of sets of ground truth IDs
            k_values: List of k values for Hit@K and Recall@K (default: [1, 5, 10, 20])

        Returns:
            Dictionary of metric_name -> mean_score
        """
        if k_values is None:
            k_values = [1, 5, 10, 20]

        batch_size = len(gt_ids_batch)
        max_k = retrieved_ids_batch.shape[1] if retrieved_ids_batch.ndim > 1 else 1

        # Pre-compute log denominators for NDCG
        max_k_val = max(k_values)
        log_denoms = np.log2(np.arange(2, max_k_val + 2, dtype=np.float32))

        # Initialize score arrays
        mrr_scores = np.zeros(batch_size, dtype=np.float32)
        hit_scores = {k: np.zeros(batch_size, dtype=np.float32) for k in k_values}
        recall_scores = {k: np.zeros(batch_size, dtype=np.float32) for k in k_values}
        ndcg_scores = {k: np.zeros(batch_size, dtype=np.float32) for k in k_values}

        # Single pass over all queries
        for i, gt_set in enumerate(gt_ids_batch):
            if not gt_set:
                continue

            n_gt = len(gt_set)
            first_hit_found = False
            hits_so_far = 0
            dcg_so_far = 0.0

            # Process positions up to max needed k
            for pos in range(min(max_k_val, max_k)):
                rid = retrieved_ids_batch[i, pos]
                is_hit = rid in gt_set

                if is_hit:
                    hits_so_far += 1
                    dcg_so_far += 1.0 / log_denoms[pos]

                    if not first_hit_found:
                        mrr_scores[i] = 1.0 / (pos + 1)
                        first_hit_found = True

                # Update metrics at each k boundary
                k_pos = pos + 1
                if k_pos in k_values:
                    hit_scores[k_pos][i] = 1.0 if hits_so_far > 0 else 0.0
                    recall_scores[k_pos][i] = hits_so_far / n_gt
                    # NDCG
                    idcg = np.sum(1.0 / log_denoms[:min(n_gt, k_pos)])
                    ndcg_scores[k_pos][i] = dcg_so_far / idcg if idcg > 0 else 0.0

        # Aggregate results
        results = {
            'MRR': float(mrr_scores.mean())
        }

        for k in k_values:
            results[f'Hit@{k}'] = float(hit_scores[k].mean())
            results[f'Recall@{k}'] = float(recall_scores[k].mean())
            results[f'NDCG@{k}'] = float(ndcg_scores[k].mean())

        return results