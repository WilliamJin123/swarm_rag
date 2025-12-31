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