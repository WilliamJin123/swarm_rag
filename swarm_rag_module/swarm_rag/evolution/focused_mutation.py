"""
Focused metric-aware mutation for evolution.

This module provides intelligent mutation that targets the weakest
metric in a genome's fitness profile, increasing the likelihood
of meaningful improvements rather than random parameter changes.

The key insight is that different parameters affect different metrics:
- Low recall -> Need more coverage (more agents, steps, larger pools)
- Low MRR -> Need better ranking (tune decay, focus on quality)
- Low precision -> Need better filtering (tune deposit strategies)
"""
import random
from typing import Dict, List, Any, Optional

from .types.genome import Genome, FIXED_PARAMS
from .types.fitness_results import FitnessResult
from .types.config import EvolutionContext, SwarmParamRanges


# =============================================================================
# Metric to Parameter Mapping
# =============================================================================

# Maps weak metrics to parameters that can improve them
# Each entry has:
#   - params: list of parameters to mutate
#   - direction: "increase", "decrease", or "tune" (bidirectional)
#   - description: human-readable explanation

METRIC_TO_PARAM_MAPPING: Dict[str, Dict[str, Any]] = {
    "recall_at_20": {
        # Low recall = not finding enough relevant documents
        # Solution: More coverage through more agents, more steps, larger pools
        "params": ["n_agents", "steps", "initial_pool_size"],
        "direction": "increase",
        "description": "Increase coverage to find more relevant documents",
    },
    "mrr": {
        # Low MRR = relevant documents not ranked highly
        # Solution: Tune decay (pheromone persistence), adjust agent count
        "params": ["decay", "n_agents"],
        "direction": "tune",
        "description": "Improve ranking by tuning pheromone dynamics",
    },
    "precision_at_20": {
        # Low precision = too many irrelevant documents in results
        # Solution: Reduce pool size, fewer steps (less wandering)
        "params": ["initial_pool_size", "steps"],
        "direction": "decrease",
        "description": "Reduce noise by limiting exploration",
    },
}


# =============================================================================
# Core Functions
# =============================================================================

def identify_weakest_metric(fitness: FitnessResult) -> str:
    """
    Identify which metric is the weakest in the fitness profile.

    Compares recall, MRR, and precision to find the lowest performer.
    Uses normalized comparison (all metrics are 0-1 scale).

    Args:
        fitness: FitnessResult with metrics dict containing recall_at_20, mrr, precision_at_20

    Returns:
        Name of the weakest metric
    """
    # Extract metrics from FitnessResult.metrics dict
    if fitness.metrics is None:
        # Default to recall if no metrics available
        return "recall_at_20"

    metrics = {
        "recall_at_20": fitness.metrics.get("recall_at_20", 0.5),
        "mrr": fitness.metrics.get("mrr", 0.5),
        "precision_at_20": fitness.metrics.get("precision_at_20", 0.5),
    }

    # Find metric with lowest value
    weakest = min(metrics, key=lambda k: metrics[k])
    return weakest


def get_mutation_focus(fitness: FitnessResult) -> Dict[str, Any]:
    """
    Determine which parameters to focus mutation on based on fitness.

    Analyzes the fitness profile to identify the weakest metric,
    then returns the parameters that are most likely to improve it.

    Args:
        fitness: FitnessResult from genome evaluation

    Returns:
        Dict with:
            - params: List of parameter names to mutate
            - direction: "increase", "decrease", or "tune"
            - weakest_metric: Name of the identified weak metric
    """
    weakest = identify_weakest_metric(fitness)
    mapping = METRIC_TO_PARAM_MAPPING.get(weakest, {})

    return {
        "params": mapping.get("params", ["n_agents", "steps"]),
        "direction": mapping.get("direction", "tune"),
        "weakest_metric": weakest,
        "description": mapping.get("description", "General parameter tuning"),
    }


def apply_focused_mutation(
    genome: Genome,
    ctx: EvolutionContext,
    fitness: Optional[FitnessResult] = None,
    mutation_rate: float = 0.3,
) -> Genome:
    """
    Apply focused mutation to a genome based on its fitness profile.

    If fitness is provided, mutations are targeted at parameters most
    likely to improve the weakest metric. Otherwise, falls back to
    random parameter selection.

    Args:
        genome: Genome to mutate
        ctx: Evolution context with config
        fitness: Optional fitness result for focused mutation
        mutation_rate: Base probability of mutating each target parameter

    Returns:
        Mutated genome (modified in place)
    """
    # Determine mutation focus
    evolvable_ranges = SwarmParamRanges().to_evolvable_dict()
    if fitness is not None:
        focus = get_mutation_focus(fitness)
        target_params = focus["params"]
        direction = focus["direction"]
    else:
        # Fallback: random selection from evolvable params
        target_params = list(evolvable_ranges.keys())
        direction = "tune"

    # Apply mutations to target parameters
    for param in target_params:
        if param in FIXED_PARAMS:
            continue  # Never mutate fixed params

        if random.random() > mutation_rate:
            continue  # Skip based on rate

        if param not in genome.params:
            continue  # Param not in genome

        current_value = genome.params[param]
        new_value = _mutate_param_focused(param, current_value, direction)
        genome.params[param] = new_value

    return genome


def _mutate_param_focused(
    param: str,
    current_value: Any,
    direction: str,
) -> Any:
    """
    Mutate a single parameter in a focused direction.

    Args:
        param: Parameter name
        current_value: Current parameter value
        direction: "increase", "decrease", or "tune"

    Returns:
        New parameter value, clamped to valid range
    """
    evolvable_ranges = SwarmParamRanges().to_evolvable_dict()
    if param not in evolvable_ranges:
        return current_value

    min_val, max_val = evolvable_ranges[param]
    is_int = isinstance(min_val, int) and isinstance(max_val, int)

    if direction == "increase":
        # Bias toward higher values
        if is_int:
            delta = random.randint(1, 3)
            new_value = current_value + delta
        else:
            factor = random.uniform(1.05, 1.20)
            new_value = current_value * factor

    elif direction == "decrease":
        # Bias toward lower values
        if is_int:
            delta = random.randint(1, 3)
            new_value = current_value - delta
        else:
            factor = random.uniform(0.80, 0.95)
            new_value = current_value * factor

    else:  # "tune" - bidirectional
        if is_int:
            delta = random.randint(-2, 2)
            new_value = current_value + delta
        else:
            factor = random.gauss(1.0, 0.1)
            new_value = current_value * factor

    # Clamp to valid range
    if is_int:
        new_value = max(min_val, min(max_val, int(new_value)))
    else:
        new_value = max(min_val, min(max_val, float(new_value)))

    return new_value
