"""
Weighted Sum Mode Components for Dual-Mode Evolution

This module provides GPU-optimized components for weighted sum genome mode:
- WeightedSumCompiler: Compiles weight tensors into executable strategies
- WeightedSumMutator: Self-adaptive ES-style mutation operators
- WeightedSumSeeder: Seed population generation with baseline variants

All operations use PyTorch tensors for GPU acceleration.
"""

from typing import Dict, Any, List, Callable, Tuple, Optional
from dataclasses import dataclass
import random
import math
import uuid
import torch

from ..types.genome import Genome, FIXED_PARAMS
from ..types.config import (
    WeightTensors,
    MutationSigmas,
    HeuristicFeatureConfig,
    EvolutionContext,
    SwarmParamRanges,
)
from ..types.fitness_results import FitnessResult
from ...core.heuristics import HeuristicContext, HeuristicRegistry
from ...interfaces.types import AgentGroupConfig


# =============================================================================
# WeightedSumCompiler
# =============================================================================

class WeightedSumCompiler:
    """
    Compiles weighted sum genomes into executable strategies.

    Converts weight tensors into callable strategy functions that can be
    used by SwarmRetriever. All computation is GPU-optimized using PyTorch.
    """

    def __init__(self, feature_config: HeuristicFeatureConfig):
        """
        Initialize compiler with feature configuration.

        Args:
            feature_config: Configuration specifying which features to use
        """
        self.feature_config = feature_config
        self._feature_getters: Dict[str, Callable] = {}
        self._build_feature_getters()

    def _build_feature_getters(self):
        """Build mapping from feature names to getter functions."""
        for feature_list in [
            self.feature_config.movement,
            self.feature_config.deposit,
            self.feature_config.ranking,
        ]:
            for name in feature_list:
                if name in self._feature_getters:
                    continue

                # Handle context attributes (fast path)
                if name == 'degree':
                    self._feature_getters[name] = lambda ctx: ctx.node_degrees
                elif name == 'pheromone':
                    self._feature_getters[name] = lambda ctx: ctx.pheromone_values
                elif name == 'max_pheromone':
                    self._feature_getters[name] = lambda ctx: ctx.max_pheromone
                elif name == 'avg_degree':
                    self._feature_getters[name] = lambda ctx: ctx.avg_degree
                elif name == 'votes':
                    self._feature_getters[name] = lambda ctx: ctx.votes
                else:
                    # Get from registry
                    try:
                        func = HeuristicRegistry.get(name)
                        self._feature_getters[name] = func
                    except KeyError:
                        # Fallback to constant zero
                        self._feature_getters[name] = lambda ctx, n=name: torch.zeros(1)

    def compile(self, genome: Genome) -> Dict[str, Any]:
        """
        Compile a weighted sum genome into SwarmRetriever kwargs.

        Args:
            genome: Genome with mode="weighted_sum" and weight_tensors set

        Returns:
            Dictionary of kwargs for SwarmRetriever.retrieve()
        """
        if genome.mode != "weighted_sum":
            raise ValueError(f"Cannot compile non-weighted_sum genome: {genome.mode}")

        if genome.weight_tensors is None:
            raise ValueError(f"Genome {genome.id} has no weight_tensors")

        wt = genome.weight_tensors

        # Build agent groups
        agent_groups = []
        total_agents = genome.params['n_agents']
        sorted_groups = sorted(genome.group_ratios.keys())

        if sorted_groups:
            ratios = [genome.group_ratios[g] for g in sorted_groups]
            total_ratio = sum(ratios)
            if total_ratio <= 1e-9:
                total_ratio = 1.0

            counts = [int(round(total_agents * (r / total_ratio))) for r in ratios]

            # Fix rounding remainder
            if counts:
                current_sum = sum(counts)
                if current_sum < total_agents:
                    counts[0] += (total_agents - current_sum)
                elif current_sum > total_agents:
                    counts[0] -= (current_sum - total_agents)

            for i, group_key in enumerate(sorted_groups):
                if counts[i] <= 0:
                    continue

                group_idx = int(group_key[1:])  # Extract index from "g0", "g1", etc.

                # Create movement strategy for this group
                mov_strategy = self._create_movement_strategy(wt, group_idx)
                dep_strategy = self._create_deposit_strategy(wt, group_idx)

                group_config: AgentGroupConfig = {
                    "count": counts[i],
                    "movement_strategies": {f"ws_mov_{group_idx}": (mov_strategy, 1.0)},
                    "deposit_strategies": {f"ws_dep_{group_idx}": (dep_strategy, 1.0)},
                }
                agent_groups.append(group_config)

        # Create ranking strategy (shared across groups)
        ranking_strategy = self._create_ranking_strategy(wt)
        ranking_strategies = {"ws_ranking": (ranking_strategy, 1.0)}

        return {
            **genome.params,
            "agent_groups": agent_groups,
            "ranking_strategies": ranking_strategies,
            # Pass raw weight tensors for GPU batched processing
            "weight_tensors": wt,
            "feature_config": self.feature_config,
        }

    def _create_movement_strategy(
        self, wt: WeightTensors, group_idx: int
    ) -> Callable[[HeuristicContext], torch.Tensor]:
        """Create movement strategy function for a specific group."""
        # Get weights for this group (clamp to valid index)
        idx = min(group_idx, wt.n_groups - 1)
        weights = wt.movement_weights[idx]  # (n_features,) - already on correct device
        bias = wt.movement_biases[idx].item()

        feature_names = self.feature_config.movement
        getters = [self._feature_getters[name] for name in feature_names]

        def movement_strategy(ctx: HeuristicContext) -> torch.Tensor:
            # Stack features: (n_candidates, n_features)
            features = torch.stack([getter(ctx) for getter in getters], dim=-1)
            # Weighted sum: (n_candidates,)
            scores = features @ weights + bias
            return torch.nan_to_num(scores, nan=0.0, posinf=10.0, neginf=-10.0)

        return movement_strategy

    def _create_deposit_strategy(
        self, wt: WeightTensors, group_idx: int
    ) -> Callable[[HeuristicContext], torch.Tensor]:
        """Create deposit strategy function for a specific group."""
        idx = min(group_idx, wt.n_groups - 1)
        weights = wt.deposit_weights[idx]  # Already on correct device
        bias = wt.deposit_biases[idx].item()

        feature_names = self.feature_config.deposit
        getters = [self._feature_getters[name] for name in feature_names]

        def deposit_strategy(ctx: HeuristicContext) -> torch.Tensor:
            features = torch.stack([getter(ctx) for getter in getters], dim=-1)
            scores = features @ weights + bias
            return torch.nan_to_num(scores, nan=0.0, posinf=10.0, neginf=-10.0)

        return deposit_strategy

    def _create_ranking_strategy(
        self, wt: WeightTensors
    ) -> Callable[[HeuristicContext], torch.Tensor]:
        """Create ranking strategy function (shared across groups)."""
        weights = wt.ranking_weights
        bias = wt.ranking_bias

        feature_names = self.feature_config.ranking
        getters = [self._feature_getters[name] for name in feature_names]

        def ranking_strategy(ctx: HeuristicContext) -> torch.Tensor:
            features = torch.stack([getter(ctx) for getter in getters], dim=-1)
            scores = features @ weights + bias
            return torch.nan_to_num(scores, nan=0.0, posinf=10.0, neginf=-10.0)

        return ranking_strategy


# =============================================================================
# Batch Score Computation (GPU-Optimized)
# =============================================================================

def compute_movement_scores_batched(
    features: torch.Tensor,      # (N_candidates, F_movement)
    weights: torch.Tensor,       # (G_groups, F_movement)
    biases: torch.Tensor,        # (G_groups,)
) -> torch.Tensor:
    """
    Compute movement scores for all candidates across all groups.

    Single cuBLAS matmul - extremely fast on GPU.

    Returns: (N_candidates, G_groups)
    """
    scores = features @ weights.T + biases  # (N, G)
    return scores


def assign_scores_to_agents(
    scores_all_groups: torch.Tensor,  # (N_candidates, G_groups)
    agent_group_ids: torch.Tensor,    # (A_agents,) values 0..G-1
) -> torch.Tensor:
    """
    Select appropriate group scores for each agent.

    Returns: (N_candidates, A_agents)
    """
    return scores_all_groups[:, agent_group_ids]


# =============================================================================
# WeightedSumMutator
# =============================================================================

class WeightedSumMutator:
    """
    Self-adaptive ES-style mutation for weighted sum genomes.

    Mutation operators:
    - Weight perturbation: w += N(0, sigma)
    - Bias perturbation: b += N(0, sigma)
    - Group ratio shift: Rebalance proportions
    - Hyperparam mutation: Mutate n_agents, steps, decay, pool_size
    - Group add/remove: Change n_groups by +/-1

    Probabilities are configurable via GeneticConfig.
    """

    # Default mutation probabilities (used when no config provided)
    # These are deprecated - use GeneticConfig instead
    DEFAULT_PROB_WEIGHT = 0.60
    DEFAULT_PROB_BIAS = 0.15
    DEFAULT_PROB_RATIO = 0.10
    DEFAULT_PROB_HYPERPARAM = 0.10
    DEFAULT_PROB_GROUP_CHANGE = 0.05

    def __init__(
        self,
        feature_config: HeuristicFeatureConfig,
        prob_weight: float = None,
        prob_bias: float = None,
        prob_ratio: float = None,
        prob_hyperparam: float = None,
        prob_group_change: float = None,
    ):
        """
        Initialize mutator with feature configuration and optional probability overrides.

        Args:
            feature_config: Configuration specifying feature dimensions
            prob_weight: Probability of weight mutation (default: 0.60)
            prob_bias: Probability of bias mutation (default: 0.15)
            prob_ratio: Probability of ratio mutation (default: 0.10)
            prob_hyperparam: Probability of hyperparam mutation (default: 0.10)
            prob_group_change: Probability of group change mutation (default: 0.05)
        """
        self.feature_config = feature_config
        self.n_movement_features = len(feature_config.movement)
        self.n_deposit_features = len(feature_config.deposit)
        self.n_ranking_features = len(feature_config.ranking)

        # Use provided probabilities or defaults
        self.prob_weight = prob_weight if prob_weight is not None else self.DEFAULT_PROB_WEIGHT
        self.prob_bias = prob_bias if prob_bias is not None else self.DEFAULT_PROB_BIAS
        self.prob_ratio = prob_ratio if prob_ratio is not None else self.DEFAULT_PROB_RATIO
        self.prob_hyperparam = prob_hyperparam if prob_hyperparam is not None else self.DEFAULT_PROB_HYPERPARAM
        self.prob_group_change = prob_group_change if prob_group_change is not None else self.DEFAULT_PROB_GROUP_CHANGE

    def mutate(self, genome: Genome, ctx: EvolutionContext) -> Genome:
        """
        Apply self-adaptive mutation to a weighted sum genome.

        First adapts the mutation sigmas, then applies mutations.

        Args:
            genome: Genome to mutate (will be modified in place)
            ctx: Evolution context

        Returns:
            Mutated genome (same object, modified)
        """
        if genome.mode != "weighted_sum":
            raise ValueError(f"Cannot mutate non-weighted_sum genome: {genome.mode}")

        # Step 1: Adapt sigmas (ES-style)
        if ctx.config.genetic.self_adaptive_mutation:
            genome.mutation_sigmas = genome.mutation_sigmas.adapt()

        sigmas = genome.mutation_sigmas

        # Step 2: Apply mutations based on probabilities
        roll = random.random()

        if roll < self.prob_weight:
            self._mutate_weights(genome, sigmas)
        elif roll < self.prob_weight + self.prob_bias:
            self._mutate_biases(genome, sigmas)
        elif roll < self.prob_weight + self.prob_bias + self.prob_ratio:
            self._mutate_ratios(genome, sigmas)
        elif roll < self.prob_weight + self.prob_bias + self.prob_ratio + self.prob_hyperparam:
            self._mutate_hyperparams(genome, sigmas, ctx)
        else:
            self._mutate_group_count(genome, ctx)

        genome.evaluated = False
        genome.clear_cache()
        return genome

    def _mutate_weights(self, genome: Genome, sigmas: MutationSigmas):
        """Perturb weight values with Gaussian noise.

        Uses torch.randn_like which automatically creates noise on the same
        device as the input tensor, avoiding GPU-CPU handoffs.
        """
        wt = genome.weight_tensors
        sigma = sigmas.weight_sigma

        # Movement weights (randn_like preserves device)
        noise = torch.randn_like(wt.movement_weights) * sigma
        wt.movement_weights = wt.movement_weights + noise

        # Deposit weights
        noise = torch.randn_like(wt.deposit_weights) * sigma
        wt.deposit_weights = wt.deposit_weights + noise

        # Ranking weights
        noise = torch.randn_like(wt.ranking_weights) * sigma
        wt.ranking_weights = wt.ranking_weights + noise

    def _mutate_biases(self, genome: Genome, sigmas: MutationSigmas):
        """Perturb bias values with Gaussian noise.

        Uses torch.randn_like which automatically creates noise on the same
        device as the input tensor, avoiding GPU-CPU handoffs.
        """
        wt = genome.weight_tensors
        sigma = sigmas.bias_sigma

        # Movement biases (randn_like preserves device)
        noise = torch.randn_like(wt.movement_biases) * sigma
        wt.movement_biases = wt.movement_biases + noise

        # Deposit biases
        noise = torch.randn_like(wt.deposit_biases) * sigma
        wt.deposit_biases = wt.deposit_biases + noise

        # Ranking bias (scalar, stays on CPU)
        wt.ranking_bias += random.gauss(0, sigma)

    def _mutate_ratios(self, genome: Genome, sigmas: MutationSigmas):
        """Shift group ratios."""
        if len(genome.group_ratios) <= 1:
            return

        sigma = sigmas.ratio_sigma
        for key in genome.group_ratios:
            genome.group_ratios[key] += random.gauss(0, sigma)
            genome.group_ratios[key] = max(0.05, genome.group_ratios[key])

        genome.normalize_ratios()

    def _mutate_hyperparams(self, genome: Genome, sigmas: MutationSigmas, ctx: EvolutionContext):
        """Mutate swarm hyperparameters."""
        sigma = sigmas.hyperparam_sigma
        evolvable_ranges = SwarmParamRanges().to_evolvable_dict()

        for key, val in genome.params.items():
            # Skip fixed parameters
            if key in FIXED_PARAMS:
                genome.params[key] = FIXED_PARAMS[key]
                continue

            if random.random() < 0.5:  # 50% chance to mutate each param
                if isinstance(val, int):
                    delta = int(round(random.gauss(0, sigma * 5)))
                    new_val = max(1, val + delta)
                    if key in evolvable_ranges:
                        min_v, max_v = evolvable_ranges[key]
                        new_val = max(int(min_v), min(int(max_v), new_val))
                    genome.params[key] = new_val
                elif isinstance(val, float):
                    factor = 1.0 + random.gauss(0, sigma)
                    new_val = val * factor
                    if key in evolvable_ranges:
                        min_v, max_v = evolvable_ranges[key]
                        new_val = max(min_v, min(max_v, new_val))
                    else:
                        new_val = max(0.001, min(0.999, new_val))
                    genome.params[key] = new_val

    def _mutate_group_count(self, genome: Genome, ctx: EvolutionContext):
        """Add or remove an agent group."""
        min_groups, max_groups = ctx.config.genetic.n_agent_groups_range
        current_groups = genome.weight_tensors.n_groups

        if random.random() < 0.5 and current_groups < max_groups:
            # Add a group
            self._add_group(genome)
        elif current_groups > min_groups:
            # Remove a group
            self._remove_group(genome)

    def _add_group(self, genome: Genome):
        """Add a new agent group by duplicating and perturbing existing."""
        wt = genome.weight_tensors
        n_groups = wt.n_groups

        # Get device from existing tensors to avoid GPU-CPU handoffs
        device = wt.movement_weights.device

        # Duplicate last group with small perturbation (create on same device)
        new_mov = wt.movement_weights[-1:] + torch.randn(1, wt.movement_weights.shape[1], device=device) * 0.1
        new_mov_bias = wt.movement_biases[-1:] + torch.randn(1, device=device) * 0.05
        new_dep = wt.deposit_weights[-1:] + torch.randn(1, wt.deposit_weights.shape[1], device=device) * 0.1
        new_dep_bias = wt.deposit_biases[-1:] + torch.randn(1, device=device) * 0.05

        wt.movement_weights = torch.cat([wt.movement_weights, new_mov], dim=0)
        wt.movement_biases = torch.cat([wt.movement_biases, new_mov_bias], dim=0)
        wt.deposit_weights = torch.cat([wt.deposit_weights, new_dep], dim=0)
        wt.deposit_biases = torch.cat([wt.deposit_biases, new_dep_bias], dim=0)

        # Add new group ratio
        new_key = f"g{n_groups}"
        genome.group_ratios[new_key] = 0.2  # Start with small allocation
        genome.normalize_ratios()

    def _remove_group(self, genome: Genome):
        """Remove the last agent group."""
        wt = genome.weight_tensors
        n_groups = wt.n_groups

        if n_groups <= 1:
            return

        # Remove last group's weights
        wt.movement_weights = wt.movement_weights[:-1]
        wt.movement_biases = wt.movement_biases[:-1]
        wt.deposit_weights = wt.deposit_weights[:-1]
        wt.deposit_biases = wt.deposit_biases[:-1]

        # Remove last group ratio
        last_key = f"g{n_groups - 1}"
        if last_key in genome.group_ratios:
            del genome.group_ratios[last_key]

        genome.normalize_ratios()


# =============================================================================
# WeightedSumSeeder
# =============================================================================

# Baseline configuration - balanced starting point
# Target metrics: Hit@1>60%, Hit@5>80%, MRR>80%, Recall@20>85%
BASELINE_CONFIG = {
    "n_agents": 25,
    "steps": 5,
    "decay": 0.5,
    "initial_pool_size": 30,
    "movement_weights": {
        "semantic_similarity_unnormalized": 0.40,  # Core relevance signal
        "stark_centrality": 0.20,                  # STaRK graph structure
        "node_centrality": 0.15,                   # Hub navigation
        "pheromone_repulsion": 0.15,               # Avoid over-visited
        "random_jitter": 0.10,                     # Exploration noise
    },
    "deposit_weights": {
        "flat": 0.30,                              # Baseline deposit
        "semantic_unnormalized": 0.25,             # Relevance-weighted
        "exploration_bonus": 0.20,                 # Reward new nodes
        "hub": 0.15,                               # Reinforce hubs
        "collaborative_amplification": 0.10,      # Consensus paths
    },
    "ranking_weights": {
        "semantic_rank": 0.80,                     # Primary ranking signal
        "percentage_visited": 0.20,                # Swarm consensus
    },
}

# =============================================================================
# Seed Variants - Diverse configurations targeting different metrics
# =============================================================================
# Strategy:
# - Precision seeds (Hit@1, MRR): high semantic, hub deposit, collaborative_amp
# - Recall seeds (Recall@20): high jitter, exploration, many agents
# - Balanced seeds: different trade-offs between precision and recall
SEED_VARIANTS = [
    # --- BASELINE ---
    {"name": "baseline_balanced", "changes": {}},

    # --- PRECISION-FOCUSED (Hit@1, MRR) ---
    # High semantic similarity for precise top-1 results
    {"name": "precision_semantic", "changes": {
        "movement_weights": {
            "semantic_similarity_unnormalized": 0.60, "stark_centrality": 0.15,
            "node_centrality": 0.10, "pheromone_repulsion": 0.10, "random_jitter": 0.05,
        },
        "deposit_weights": {
            "flat": 0.10, "semantic_unnormalized": 0.40, "exploration_bonus": 0.10,
            "hub": 0.20, "collaborative_amplification": 0.20,
        },
        "n_agents": 20, "steps": 6,
    }},

    # Hub highways - fast paths to relevant content
    {"name": "precision_hub_highways", "changes": {
        "movement_weights": {
            "semantic_similarity_unnormalized": 0.35, "stark_centrality": 0.30,
            "node_centrality": 0.20, "pheromone_repulsion": 0.10, "random_jitter": 0.05,
        },
        "deposit_weights": {
            "flat": 0.10, "semantic_unnormalized": 0.20, "exploration_bonus": 0.10,
            "hub": 0.35, "collaborative_amplification": 0.25,
        },
    }},

    # Consensus-driven - multiple agents agree = high quality
    {"name": "precision_consensus", "changes": {
        "movement_weights": {
            "semantic_similarity_unnormalized": 0.45, "stark_centrality": 0.20,
            "node_centrality": 0.15, "pheromone_repulsion": 0.05, "random_jitter": 0.15,
        },
        "deposit_weights": {
            "flat": 0.05, "semantic_unnormalized": 0.25, "exploration_bonus": 0.10,
            "hub": 0.20, "collaborative_amplification": 0.40,
        },
        "ranking_weights": {"semantic_rank": 0.60, "percentage_visited": 0.40},
    }},

    # --- RECALL-FOCUSED (Recall@20) ---
    # High exploration - maximize coverage
    {"name": "recall_explorer", "changes": {
        "movement_weights": {
            "semantic_similarity_unnormalized": 0.25, "stark_centrality": 0.15,
            "node_centrality": 0.15, "pheromone_repulsion": 0.25, "random_jitter": 0.20,
        },
        "deposit_weights": {
            "flat": 0.20, "semantic_unnormalized": 0.15, "exploration_bonus": 0.40,
            "hub": 0.15, "collaborative_amplification": 0.10,
        },
        "n_agents": 40, "steps": 4, "initial_pool_size": 50,
    }},

    # Wide search - many agents, shallow depth
    {"name": "recall_wide_swarm", "changes": {
        "movement_weights": {
            "semantic_similarity_unnormalized": 0.30, "stark_centrality": 0.20,
            "node_centrality": 0.10, "pheromone_repulsion": 0.20, "random_jitter": 0.20,
        },
        "deposit_weights": {
            "flat": 0.30, "semantic_unnormalized": 0.20, "exploration_bonus": 0.30,
            "hub": 0.10, "collaborative_amplification": 0.10,
        },
        "n_agents": 50, "steps": 3, "initial_pool_size": 60,
    }},

    # Repulsion-driven - spread agents across graph
    {"name": "recall_repulsion", "changes": {
        "movement_weights": {
            "semantic_similarity_unnormalized": 0.30, "stark_centrality": 0.15,
            "node_centrality": 0.10, "pheromone_repulsion": 0.35, "random_jitter": 0.10,
        },
        "deposit_weights": {
            "flat": 0.15, "semantic_unnormalized": 0.15, "exploration_bonus": 0.45,
            "hub": 0.15, "collaborative_amplification": 0.10,
        },
        "decay": 0.3,  # Fast decay to encourage spread
    }},

    # --- BALANCED VARIANTS ---
    # Deep search - fewer agents, more steps
    {"name": "balanced_deep", "changes": {
        "movement_weights": {
            "semantic_similarity_unnormalized": 0.45, "stark_centrality": 0.20,
            "node_centrality": 0.15, "pheromone_repulsion": 0.10, "random_jitter": 0.10,
        },
        "n_agents": 18, "steps": 8, "decay": 0.7,
    }},

    # STaRK-optimized - leverage graph structure
    {"name": "balanced_stark_graph", "changes": {
        "movement_weights": {
            "semantic_similarity_unnormalized": 0.30, "stark_centrality": 0.35,
            "node_centrality": 0.15, "pheromone_repulsion": 0.10, "random_jitter": 0.10,
        },
        "deposit_weights": {
            "flat": 0.20, "semantic_unnormalized": 0.20, "exploration_bonus": 0.20,
            "hub": 0.25, "collaborative_amplification": 0.15,
        },
    }},

    # Semantic-deposit synergy
    {"name": "balanced_semantic_deposit", "changes": {
        "movement_weights": {
            "semantic_similarity_unnormalized": 0.50, "stark_centrality": 0.15,
            "node_centrality": 0.10, "pheromone_repulsion": 0.15, "random_jitter": 0.10,
        },
        "deposit_weights": {
            "flat": 0.10, "semantic_unnormalized": 0.45, "exploration_bonus": 0.15,
            "hub": 0.15, "collaborative_amplification": 0.15,
        },
    }},

    # Hybrid precision-recall
    {"name": "balanced_hybrid", "changes": {
        "movement_weights": {
            "semantic_similarity_unnormalized": 0.35, "stark_centrality": 0.20,
            "node_centrality": 0.15, "pheromone_repulsion": 0.15, "random_jitter": 0.15,
        },
        "deposit_weights": {
            "flat": 0.15, "semantic_unnormalized": 0.25, "exploration_bonus": 0.25,
            "hub": 0.20, "collaborative_amplification": 0.15,
        },
        "ranking_weights": {"semantic_rank": 0.70, "percentage_visited": 0.30},
        "n_agents": 30, "steps": 5,
    }},
]


class WeightedSumSeeder:
    """
    Generates seed population for weighted sum evolution.

    Creates baseline + variants + perturbed + wildcard genomes.
    """

    def __init__(self, feature_config: HeuristicFeatureConfig, device: str = "cuda"):
        """
        Initialize seeder with feature configuration.

        Args:
            feature_config: Configuration specifying feature dimensions
            device: Target device for tensors ("cuda" or "cpu")
        """
        self.feature_config = feature_config
        self.device = device
        self.n_movement_features = len(feature_config.movement)
        self.n_deposit_features = len(feature_config.deposit)
        self.n_ranking_features = len(feature_config.ranking)

    def create_seed_population(self, count: int = 18) -> List[Genome]:
        """
        Create seed population with variants.

        Default: 10 predefined + 5 perturbed + 3 wildcard = 18 seeds

        Args:
            count: Number of seeds to create (min 10)

        Returns:
            List of seed genomes
        """
        seeds = []

        # 1. Create predefined variants (up to 10)
        for i, variant in enumerate(SEED_VARIANTS[:min(10, count)]):
            genome = self._create_variant_genome(variant, f"seed_{variant['name']}")
            seeds.append(genome)

        remaining = count - len(seeds)

        # 2. Create perturbed variants (5)
        n_perturbed = min(5, remaining)
        for i in range(n_perturbed):
            genome = self._create_perturbed_genome(f"seed_perturb_{i}")
            seeds.append(genome)

        remaining = count - len(seeds)

        # 3. Create wildcard (random) variants
        for i in range(remaining):
            genome = self._create_wildcard_genome(f"seed_wildcard_{i}")
            seeds.append(genome)

        return seeds

    def _create_variant_genome(self, variant: Dict, genome_id: str) -> Genome:
        """Create genome from a variant specification."""
        # Start with baseline
        config = self._deep_merge(BASELINE_CONFIG.copy(), variant.get("changes", {}))

        return self._config_to_genome(config, genome_id, device=self.device)

    def _create_perturbed_genome(self, genome_id: str) -> Genome:
        """Create genome by perturbing baseline +/- 15%."""
        config = BASELINE_CONFIG.copy()

        # Perturb weights
        perturbed_config = {}
        for key in ["movement_weights", "deposit_weights", "ranking_weights"]:
            if key in config:
                perturbed_config[key] = {}
                for feat, val in config[key].items():
                    perturb = random.uniform(0.85, 1.15)
                    perturbed_config[key][feat] = val * perturb

        # Perturb hyperparams
        for key in ["n_agents", "steps", "decay", "initial_pool_size"]:
            if key in config:
                val = config[key]
                if isinstance(val, int):
                    delta = int(round(val * random.uniform(-0.15, 0.15)))
                    perturbed_config[key] = max(1, val + delta)
                else:
                    perturbed_config[key] = val * random.uniform(0.85, 1.15)

        merged = self._deep_merge(config, perturbed_config)
        return self._config_to_genome(merged, genome_id, device=self.device)

    def _create_wildcard_genome(self, genome_id: str) -> Genome:
        """Create genome with random weights but baseline hyperparams."""
        config = {
            "n_agents": BASELINE_CONFIG["n_agents"],
            "steps": BASELINE_CONFIG["steps"],
            "decay": BASELINE_CONFIG["decay"],
            "initial_pool_size": BASELINE_CONFIG["initial_pool_size"],
            "movement_weights": {f: random.uniform(0.0, 1.0)
                                for f in self.feature_config.movement},
            "deposit_weights": {f: random.uniform(0.0, 1.0)
                               for f in self.feature_config.deposit},
            "ranking_weights": {f: random.uniform(0.0, 1.0)
                               for f in self.feature_config.ranking},
        }
        return self._config_to_genome(config, genome_id, device=self.device)

    def _config_to_genome(self, config: Dict, genome_id: str, device: str = "cuda") -> Genome:
        """Convert a config dictionary to a Genome object.

        Args:
            config: Configuration dictionary with weights and hyperparameters
            genome_id: Unique identifier for the genome
            device: Target device for tensors ("cpu" or "cuda")
        """
        # Build weight tensors directly on target device
        movement_weights = self._weights_dict_to_tensor(
            config.get("movement_weights", {}),
            self.feature_config.movement,
            device=device,
        )
        deposit_weights = self._weights_dict_to_tensor(
            config.get("deposit_weights", {}),
            self.feature_config.deposit,
            device=device,
        )
        ranking_weights = self._weights_dict_to_tensor(
            config.get("ranking_weights", {}),
            self.feature_config.ranking,
            device=device,
        )

        # Single group by default (create biases on same device)
        weight_tensors = WeightTensors(
            movement_weights=movement_weights.unsqueeze(0),  # (1, n_features)
            movement_biases=torch.zeros(1, device=device),
            deposit_weights=deposit_weights.unsqueeze(0),
            deposit_biases=torch.zeros(1, device=device),
            ranking_weights=ranking_weights,
            ranking_bias=0.0,
        )

        # Build params
        params = dict(FIXED_PARAMS)
        for key in ["n_agents", "steps", "decay", "initial_pool_size"]:
            if key in config:
                params[key] = config[key]
            elif key in BASELINE_CONFIG:
                params[key] = BASELINE_CONFIG[key]

        return Genome(
            id=genome_id,
            mode="weighted_sum",
            params=params,
            group_ratios={"g0": 1.0},
            strategies={},  # Empty for weighted_sum mode
            weight_tensors=weight_tensors,
            mutation_sigmas=MutationSigmas(),
            fitness=FitnessResult(),
            evaluated=False,
        )

    def _weights_dict_to_tensor(
        self, weights_dict: Dict[str, float], feature_names: List[str], device: str = "cpu"
    ) -> torch.Tensor:
        """Convert weights dictionary to tensor in correct feature order.

        Args:
            weights_dict: Dictionary mapping feature names to weights
            feature_names: Ordered list of feature names
            device: Target device for the tensor

        Returns:
            Tensor of weights in feature order, on specified device
        """
        weights = []
        for name in feature_names:
            weights.append(weights_dict.get(name, 0.0))
        return torch.as_tensor(weights, dtype=torch.float32, device=device)

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge override into base dictionary."""
        result = base.copy()
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = self._deep_merge(result[key], val)
            else:
                result[key] = val
        return result


# =============================================================================
# Strategy Registration
# =============================================================================

def register_weighted_sum_strategies():
    """Register weighted sum strategies with GeneticRegistry."""
    from .strategies import GeneticRegistry

    @GeneticRegistry.register_mutation("self_adaptive_es")
    def self_adaptive_es_mutation(genome: Genome, ctx: EvolutionContext) -> Genome:
        """Self-adaptive ES-style mutation for weighted sum genomes."""
        if genome.mode != "weighted_sum":
            # Fallback to expression tree mutation
            mutation_fn = GeneticRegistry.get_mutation("guided_mutation")
            return mutation_fn(genome, ctx)

        # Get mutation probabilities from config
        genetic_config = ctx.config.genetic
        mutator = WeightedSumMutator(
            ctx.config.heuristic_features,
            prob_weight=genetic_config.mutation_prob_weight,
            prob_bias=genetic_config.mutation_prob_bias,
            prob_ratio=genetic_config.mutation_prob_ratio,
            prob_hyperparam=genetic_config.mutation_prob_hyperparam,
            prob_group_change=genetic_config.mutation_prob_group_change,
        )
        return mutator.mutate(genome, ctx)

    @GeneticRegistry.register_creation("weighted_sum_seeded")
    def weighted_sum_seeded_creation(
        ctx: EvolutionContext, count: int
    ) -> List[Genome]:
        """Create seed population for weighted sum evolution."""
        device = ctx.device  # Get device from context (cuda or cpu)
        seeder = WeightedSumSeeder(ctx.config.heuristic_features, device=device)
        seeds = seeder.create_seed_population(count)

        # If count > seed population size, generate more random genomes
        while len(seeds) < count:
            genome_id = f"random_ws_{uuid.uuid4().hex[:8]}"
            seeds.append(seeder._create_wildcard_genome(genome_id))

        return seeds[:count]

    @GeneticRegistry.register_crossover("weighted_sum_crossover")
    def weighted_sum_crossover(
        parent1: Genome, parent2: Genome, ctx: EvolutionContext
    ) -> Genome:
        """Crossover for weighted sum genomes."""
        if parent1.mode != "weighted_sum" or parent2.mode != "weighted_sum":
            # Fallback to uniform parameter mix
            crossover_fn = GeneticRegistry.get_crossover("uniform_parameter_mix")
            return crossover_fn(parent1, parent2, ctx)

        child = parent1.copy(new_id=f"child_{uuid.uuid4().hex[:8]}")

        # Crossover weights (uniform selection per weight)
        wt1 = parent1.weight_tensors
        wt2 = parent2.weight_tensors
        child_wt = child.weight_tensors

        # Handle different group counts by using min
        n_groups = min(wt1.n_groups, wt2.n_groups)

        # Movement weights crossover
        mask = torch.rand_like(child_wt.movement_weights[:n_groups]) > 0.5
        child_wt.movement_weights[:n_groups] = torch.where(
            mask, wt1.movement_weights[:n_groups], wt2.movement_weights[:n_groups]
        )

        # Deposit weights crossover
        mask = torch.rand_like(child_wt.deposit_weights[:n_groups]) > 0.5
        child_wt.deposit_weights[:n_groups] = torch.where(
            mask, wt1.deposit_weights[:n_groups], wt2.deposit_weights[:n_groups]
        )

        # Ranking weights crossover
        mask = torch.rand_like(child_wt.ranking_weights) > 0.5
        child_wt.ranking_weights = torch.where(
            mask, wt1.ranking_weights, wt2.ranking_weights
        )

        # Crossover biases
        if random.random() > 0.5:
            child_wt.movement_biases[:n_groups] = wt2.movement_biases[:n_groups]
        if random.random() > 0.5:
            child_wt.deposit_biases[:n_groups] = wt2.deposit_biases[:n_groups]
        if random.random() > 0.5:
            child_wt.ranking_bias = wt2.ranking_bias

        # Crossover hyperparams
        for key in child.params:
            if key in parent2.params and random.random() > 0.5:
                child.params[key] = parent2.params[key]

        # Crossover group ratios
        for key in child.group_ratios:
            if key in parent2.group_ratios and random.random() > 0.5:
                child.group_ratios[key] = parent2.group_ratios[key]

        # Crossover mutation sigmas
        if random.random() > 0.5:
            child.mutation_sigmas = parent2.mutation_sigmas.copy()

        child.normalize_ratios()
        child.evaluated = False
        child.clear_cache()
        return child


# Auto-register on import
register_weighted_sum_strategies()
