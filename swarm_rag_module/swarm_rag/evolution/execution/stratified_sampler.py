"""
Stratified Query Sampling for Evolution.

Implements stratified sampling that ensures evaluation queries are
representative of all difficulty categories. This provides better
fitness signal with fewer queries.

Expected Impact: Better fitness signal with 50% fewer queries.
"""
import random
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QueryCategory:
    """Represents a category of queries."""
    name: str
    indices: List[int] = field(default_factory=list)
    weight: float = 1.0  # Sampling weight (higher = more samples)


@dataclass
class StratifiedSample:
    """Result of stratified sampling."""
    indices: List[int]
    queries: List[str]
    ground_truth: List[List[Any]]
    category_counts: Dict[str, int]


class StratifiedQuerySampler:
    """
    Stratified sampler for evaluation queries.

    Ensures queries are sampled proportionally from categories based on
    difficulty, type, or MAP-Elites descriptors. This provides a more
    representative fitness signal than random sampling.

    Args:
        queries: All available queries
        ground_truth: Ground truth for all queries
        categories: Optional pre-assigned categories (list of category names)
        categorizer: Optional function to categorize queries
    """

    def __init__(
        self,
        queries: List[str],
        ground_truth: List[List[Any]],
        categories: Optional[List[str]] = None,
        categorizer: Optional[Callable[[str, List[Any]], str]] = None,
    ):
        self.queries = queries
        self.ground_truth = ground_truth
        self._categories: Dict[str, QueryCategory] = {}
        self._rng = random.Random()

        # Auto-categorize if needed
        if categories is not None:
            self._build_from_categories(categories)
        elif categorizer is not None:
            self._categorize_queries(categorizer)
        else:
            self._auto_categorize()

    def _build_from_categories(self, categories: List[str]):
        """Build category structure from provided category labels."""
        cat_indices = defaultdict(list)
        for i, cat in enumerate(categories):
            cat_indices[cat].append(i)

        for name, indices in cat_indices.items():
            self._categories[name] = QueryCategory(name=name, indices=indices)

    def _categorize_queries(self, categorizer: Callable[[str, List[Any]], str]):
        """Categorize queries using provided function."""
        cat_indices = defaultdict(list)
        for i, (q, gt) in enumerate(zip(self.queries, self.ground_truth)):
            category = categorizer(q, gt)
            cat_indices[category].append(i)

        for name, indices in cat_indices.items():
            self._categories[name] = QueryCategory(name=name, indices=indices)

    def _auto_categorize(self):
        """Auto-categorize based on query characteristics."""
        # Default: categorize by ground truth size (difficulty proxy)
        cat_indices = defaultdict(list)

        for i, gt in enumerate(self.ground_truth):
            gt_size = len(gt) if gt else 0
            if gt_size <= 1:
                category = "sparse"  # Few relevant documents
            elif gt_size <= 5:
                category = "medium"  # Moderate relevance
            else:
                category = "dense"   # Many relevant documents

            cat_indices[category].append(i)

        for name, indices in cat_indices.items():
            self._categories[name] = QueryCategory(name=name, indices=indices)

        logger.debug(f"Auto-categorized queries: {[(k, len(v.indices)) for k, v in self._categories.items()]}")

    def set_category_weights(self, weights: Dict[str, float]):
        """
        Set sampling weights for categories.

        Higher weights = more samples from that category.
        Useful to oversample hard queries or underrepresented categories.
        """
        for name, weight in weights.items():
            if name in self._categories:
                self._categories[name].weight = weight

    def sample(
        self,
        sample_size: int,
        seed: Optional[int] = None,
        proportional: bool = True
    ) -> StratifiedSample:
        """
        Sample queries with stratification.

        Args:
            sample_size: Total number of queries to sample
            seed: Random seed for reproducibility
            proportional: If True, sample proportionally to category size
                         If False, sample equally from each category

        Returns:
            StratifiedSample with indices, queries, and ground truth
        """
        if seed is not None:
            self._rng.seed(seed)

        # Calculate samples per category
        if proportional:
            samples_per_cat = self._proportional_allocation(sample_size)
        else:
            samples_per_cat = self._equal_allocation(sample_size)

        # Sample from each category
        all_indices = []
        category_counts = {}

        for name, category in self._categories.items():
            n_samples = samples_per_cat.get(name, 0)
            n_samples = min(n_samples, len(category.indices))

            if n_samples > 0:
                sampled = self._rng.sample(category.indices, n_samples)
                all_indices.extend(sampled)
                category_counts[name] = n_samples

        # Shuffle to avoid category clustering
        self._rng.shuffle(all_indices)

        return StratifiedSample(
            indices=all_indices,
            queries=[self.queries[i] for i in all_indices],
            ground_truth=[self.ground_truth[i] for i in all_indices],
            category_counts=category_counts
        )

    def _proportional_allocation(self, sample_size: int) -> Dict[str, int]:
        """Allocate samples proportionally to category size and weight."""
        total_weighted = sum(
            len(cat.indices) * cat.weight
            for cat in self._categories.values()
        )

        if total_weighted == 0:
            return {}

        allocation = {}
        remaining = sample_size

        for name, category in self._categories.items():
            weighted_size = len(category.indices) * category.weight
            n_samples = int(sample_size * weighted_size / total_weighted)
            allocation[name] = n_samples
            remaining -= n_samples

        # Distribute remaining samples to largest categories
        if remaining > 0:
            sorted_cats = sorted(
                self._categories.items(),
                key=lambda x: len(x[1].indices),
                reverse=True
            )
            for name, _ in sorted_cats:
                if remaining <= 0:
                    break
                allocation[name] = allocation.get(name, 0) + 1
                remaining -= 1

        return allocation

    def _equal_allocation(self, sample_size: int) -> Dict[str, int]:
        """Allocate samples equally across categories."""
        n_categories = len(self._categories)
        if n_categories == 0:
            return {}

        base_per_cat = sample_size // n_categories
        remainder = sample_size % n_categories

        allocation = {}
        for i, (name, category) in enumerate(self._categories.items()):
            n_samples = base_per_cat + (1 if i < remainder else 0)
            n_samples = min(n_samples, len(category.indices))
            allocation[name] = n_samples

        return allocation

    def sample_tiered(
        self,
        tier_sizes: List[int],
        seed: Optional[int] = None
    ) -> List[StratifiedSample]:
        """
        Create stratified samples for multiple evaluation tiers.

        Each tier builds on previous tiers (tier 2 includes tier 1 queries).

        Args:
            tier_sizes: List of cumulative sample sizes [10, 30, 60, 100]
            seed: Random seed

        Returns:
            List of StratifiedSample for each tier
        """
        if seed is not None:
            self._rng.seed(seed)

        # Sample the largest tier first
        max_size = max(tier_sizes)
        full_sample = self.sample(max_size, seed=None)  # Already seeded

        # Create tiered samples (subsets of full)
        samples = []
        for tier_size in tier_sizes:
            tier_indices = full_sample.indices[:tier_size]
            tier_queries = full_sample.queries[:tier_size]
            tier_gt = full_sample.ground_truth[:tier_size]

            # Count categories in this tier
            cat_counts = defaultdict(int)
            for idx in tier_indices:
                for name, cat in self._categories.items():
                    if idx in cat.indices:
                        cat_counts[name] += 1
                        break

            samples.append(StratifiedSample(
                indices=tier_indices,
                queries=tier_queries,
                ground_truth=tier_gt,
                category_counts=dict(cat_counts)
            ))

        return samples

    def get_category_stats(self) -> Dict[str, Any]:
        """Get statistics about query categories."""
        return {
            "total_queries": len(self.queries),
            "n_categories": len(self._categories),
            "categories": {
                name: {
                    "count": len(cat.indices),
                    "weight": cat.weight,
                    "percentage": len(cat.indices) / len(self.queries) * 100
                }
                for name, cat in self._categories.items()
            }
        }


def categorize_by_query_length(query: str, ground_truth: List[Any]) -> str:
    """Categorize by query length (word count)."""
    word_count = len(query.split())
    if word_count <= 5:
        return "short"
    elif word_count <= 15:
        return "medium"
    else:
        return "long"


def categorize_by_difficulty(query: str, ground_truth: List[Any]) -> str:
    """Categorize by estimated difficulty based on ground truth."""
    gt_size = len(ground_truth) if ground_truth else 0
    if gt_size == 0:
        return "impossible"
    elif gt_size == 1:
        return "needle"  # Finding a needle in haystack
    elif gt_size <= 5:
        return "sparse"
    else:
        return "abundant"


def categorize_by_embedding_variance(
    query: str,
    ground_truth: List[Any],
    query_embeddings: Optional[np.ndarray] = None,
    query_idx: Optional[int] = None
) -> str:
    """
    Categorize by semantic complexity (requires pre-computed embeddings).

    This is a placeholder - actual implementation would analyze embedding
    variance to estimate query ambiguity.
    """
    # Fallback to difficulty-based
    return categorize_by_difficulty(query, ground_truth)


class AdaptiveSampler:
    """
    Adaptive sampler that adjusts strategy based on archive state.

    For sparse archives: Sample more diverse queries
    For full archives: Focus on discriminative queries
    """

    def __init__(
        self,
        base_sampler: StratifiedQuerySampler,
        archive_fill_rate: float = 0.0
    ):
        self.base_sampler = base_sampler
        self.archive_fill_rate = archive_fill_rate

    def update_archive_state(self, fill_rate: float):
        """Update archive fill rate for adaptive sampling."""
        self.archive_fill_rate = fill_rate

    def sample_adaptive(self, base_size: int, seed: Optional[int] = None) -> StratifiedSample:
        """
        Sample with size adapted to archive state.

        Sparse archive (< 30% full): Larger samples to explore
        Full archive (> 70% full): Smaller, focused samples to refine
        """
        if self.archive_fill_rate < 0.3:
            # Explore more
            sample_size = int(base_size * 1.5)
            # Emphasize hard queries
            self.base_sampler.set_category_weights({
                "sparse": 1.5,
                "needle": 2.0,
                "medium": 1.0,
                "dense": 0.8,
                "abundant": 0.8
            })
        elif self.archive_fill_rate > 0.7:
            # Refine with focused evaluation
            sample_size = int(base_size * 0.6)
            # Balanced weights
            self.base_sampler.set_category_weights({
                "sparse": 1.0,
                "needle": 1.0,
                "medium": 1.0,
                "dense": 1.0,
                "abundant": 1.0
            })
        else:
            sample_size = base_size
            # Slight emphasis on medium difficulty
            self.base_sampler.set_category_weights({
                "sparse": 1.0,
                "needle": 1.2,
                "medium": 1.3,
                "dense": 1.0,
                "abundant": 0.9
            })

        return self.base_sampler.sample(sample_size, seed=seed)
