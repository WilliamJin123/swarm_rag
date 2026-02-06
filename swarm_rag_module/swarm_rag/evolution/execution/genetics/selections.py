"""
Selection operators for the evolutionary algorithm.

Contains tournament selection and Boltzmann (softmax) selection with
adaptive temperature.
"""
import logging
from typing import List

import torch

from ...types.config import EvolutionContext
from ....interfaces.enums import GeneticKey
from ...types.genome import Genome

from .registry import GeneticRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registered selection operators
# ---------------------------------------------------------------------------

@GeneticRegistry.register_selection(GeneticKey.TOURNAMENT)
def tournament_selection(ctx: EvolutionContext, k: int = 1) -> List[Genome]:
    """
    Selects 'k' parents using Tournament logic.
    """
    tourn_size = ctx.config.genetic.selection_k
    pop_size = len(ctx.population)
    winners = []
    for _ in range(k):
        indices = torch.randint(0, pop_size, (tourn_size,))
        contestants = [ctx.population[i] for i in indices.tolist()]
        # Select winner by FITNESS, not index
        winner = max(contestants, key=lambda g: g.fitness)
        winners.append(winner)

    return winners


@GeneticRegistry.register_selection(GeneticKey.BOLTZMANN)
def boltzmann_selection(ctx: EvolutionContext, k: int = 1) -> List[Genome]:
    """
    Boltzmann (Softmax) Selection with Adaptive Temperature.
    - Probability P(i) ~ exp(Fitness(i) / T)
    - Adapt T based on population diversity:
        * Low diversity -> Increase T (Heat up to explore)
        * High diversity -> Decrease T (Cool down to exploit)

    Optimized to use torch.softmax and float32 for faster computation.
    """
    boltzmann_cfg = ctx.config.genetic.boltzmann

    # Initialize Temperature (if first run)
    if ctx.generation == 0 and ctx.current_temperature == 1.0:
        ctx.current_temperature = boltzmann_cfg.temperature

    # Prepare Scores as float32 tensor for speed (sufficient precision)
    scores = torch.tensor([g.fitness.quality_score for g in ctx.population], dtype=torch.float32)

    # T controls the "pressure".
    # T -> inf: Uniform random
    # T -> 0: Deterministic max
    T = ctx.current_temperature
    T = max(1e-4, T)

    # Use torch.softmax for numerical stability and vectorization
    # softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    # Here we want exp(scores / T) / sum(exp(scores / T))
    probs = torch.softmax(scores / T, dim=0)

    # Select
    selection_indices = torch.multinomial(probs, num_samples=k, replacement=True)
    selected = [ctx.population[i] for i in selection_indices.tolist()]

    # Update Temperature (Adaptive)
    if boltzmann_cfg.adaptive:
        mean_score = scores.mean().item()
        # Calculate Coefficient of Variation (CV) = std / mean
        # Using tensor operations for speed
        if mean_score > 1e-6:
            diversity_cv = (scores.std() / mean_score).item()
        else:
            diversity_cv = 0.0

        cooling_factor = boltzmann_cfg.alpha
        heating_factor = 1.0 / cooling_factor

        min_T = boltzmann_cfg.min_temp
        max_T = boltzmann_cfg.max_temp
        diversity_threshold = boltzmann_cfg.diversity_threshold

        # Heuristic: If relative diversity is low, we are stagnating -> Heat up
        if diversity_cv < diversity_threshold:
            ctx.current_temperature *= heating_factor
        else:
            # Otherwise -> Cool down (Annealing)
            ctx.current_temperature *= cooling_factor

        # Clamp temperature within bounds using standard python math for scalar clamping
        # (faster than wrapping in tensor for a single scalar value)
        ctx.current_temperature = max(min_T, min(max_T, ctx.current_temperature))

    return selected
