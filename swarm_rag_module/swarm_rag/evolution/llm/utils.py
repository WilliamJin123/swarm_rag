"""
LLM utilities for genome context serialization and edit application.

Provides enhanced context for LLM-guided mutations including:
- Performance metrics
- Configuration (params, strategies)
- Behavioral analysis from decision tracking
- Evolutionary context
- Available heuristics and parameter bounds

Also provides tiered context builders for the three-tier architecture:
- build_strategic_context: For Strategic Oracle (Tier 1)
- build_tactical_context: For Tactical Advisor (Tier 2)
"""
from typing import Any, Dict, TypedDict, Optional, List
import logging

from ...evolution.llm.strategic_oracle import StrategicContext
from ...evolution.llm.tactical_advisor import TacticalContext
from ...evolution.llm.intents import StrategicDirective

from ..types.genome import Genome, SwarmParams
from .parsers import ExpressionParser

logger = logging.getLogger(__name__)


# Re-export tiered context builders for convenience
def build_strategic_context(
    archive_stats: Dict[str, Any],
    journal: Optional[Any],
    generation: int,
    total_generations: int,
    population: Optional[List[Any]] = None,
) -> "StrategicContext":
    """
    Build context for Strategic Oracle (Tier 1).

    Args:
        archive_stats: Stats from MapElitesArchive.stats()
        journal: EvolutionJournal instance
        generation: Current generation
        total_generations: Total planned generations
        population: Optional list of genomes for diversity calculation

    Returns:
        StrategicContext for the oracle
    """
    from .strategic_oracle import build_strategic_context as _build
    return _build(archive_stats, journal, generation, total_generations, population)


def build_tactical_context(
    genome: Genome,
    directive: "StrategicDirective",
    journal: Optional[Any] = None,
) -> "TacticalContext":
    """
    Build context for Tactical Advisor (Tier 2).

    Args:
        genome: The genome to analyze
        directive: Current strategic directive
        journal: Evolution journal for history

    Returns:
        TacticalContext for the advisor
    """
    from .tactical_advisor import build_tactical_context as _build
    return _build(genome, directive, journal)


class HeuristicStats(TypedDict):
    """Statistics for a single heuristic across all decisions."""
    mean: float
    std: float
    min: float
    max: float
    median: float


class BehavioralMetrics(TypedDict, total=False):
    """Metrics derived from agent decision tracking."""
    unique_nodes_ratio: float  # unique_visited / total_steps
    revisit_rate: float  # revisits / total_steps
    dead_end_rate: float  # dead_ends / total_steps
    avg_branching_factor: float
    convergence_step: Optional[int]  # step where >50% agents converged
    final_dispersion: float  # how spread out agents are (0=clustered, 1=dispersed)
    heuristic_usage: Dict[str, HeuristicStats]
    choice_patterns: Dict[str, float]  # avg_chosen_rank, greedy_match_rate, etc.
    # Enhanced traversal context
    sample_paths: List[Dict[str, Any]]  # Representative agent paths with stuck/revisit info
    node_hotspots: List[Dict[str, Any]]  # Most visited nodes with dead-end flags
    stuck_nodes: Dict[str, List]  # {dead_ends: [...], revisit_traps: [...]}


class EvolutionaryContext(TypedDict, total=False):
    """Context about the evolutionary process."""
    generation: int
    population_size: int
    archive_fill_rate: float  # what % of MAP-Elites cells are occupied
    genome_age: int  # how many generations since created
    mutation_rate: float


class AvailableHeuristics(TypedDict):
    """Lists of available heuristics the LLM can use."""
    movement: List[str]
    ranking: List[str]
    deposit: List[str]


class ParameterBounds(TypedDict, total=False):
    """Valid ranges for parameters."""
    n_agents: tuple
    steps: tuple
    decay: tuple
    initial_pool_size: tuple
    start_subset: tuple
    drop_zone_inc: tuple


class GenomePerformance(TypedDict):
    """Performance metrics for a genome."""
    quality_score: float
    stability_score: float
    recall_at_20: float
    hit_at_1: float
    hit_at_5: float
    mrr: float
    latency: float
    complexity: int


class GenomeConfig(TypedDict):
    """Configuration of a genome."""
    params: SwarmParams
    strategies: Dict[str, str]
    group_ratios: Dict[str, float]


class GenomeLLMContext(TypedDict, total=False):
    """Complete context for LLM genome refinement."""
    id: str
    performance: GenomePerformance
    current_config: GenomeConfig
    # Enhanced fields
    behavioral: Optional[BehavioralMetrics]
    evolutionary: Optional[EvolutionaryContext]
    available_heuristics: AvailableHeuristics
    parameter_bounds: ParameterBounds


def genome_to_json_context(
    genome: Genome,
    decision_context: Optional[Any] = None,  # QueryDecisionContext from decision_tracker
    evolution_context: Optional[Any] = None,  # EvolutionContext
) -> GenomeLLMContext:
    """
    Serializes a Genome into the format expected by the LLM.

    Enhanced version includes:
    - Performance metrics (quality, cost, recall, etc.)
    - Configuration (params, strategies as strings)
    - Behavioral metrics from decision tracking (if provided)
    - Evolutionary context (generation, population stats)
    - Available heuristics list
    - Parameter bounds

    Args:
        genome: The genome to serialize
        decision_context: Optional QueryDecisionContext with agent decision data
        evolution_context: Optional EvolutionContext with evolutionary state

    Returns:
        GenomeLLMContext dict suitable for LLM prompts
    """
    # 1. Performance metrics
    performance: GenomePerformance = {
        "quality_score": genome.fitness.quality_score,
        "stability_score": genome.fitness.stability_score,
        "recall_at_20": genome.metrics.get("Recall@20", 0.0),
        "hit_at_1": genome.metrics.get("Hit@1", 0.0),
        "hit_at_5": genome.metrics.get("Hit@5", 0.0),
        "mrr": genome.metrics.get("MRR", 0.0),
        "latency": genome.latency,
        "complexity": genome.complexity()
    }

    # 2. Configuration (strategies as strings)
    strategy_strings: Dict[str, str] = {}
    for name, tree in genome.strategies.items():
        strategy_strings[name] = tree.to_string()

    config: GenomeConfig = {
        "params": genome.params,
        "strategies": strategy_strings,
        "group_ratios": genome.group_ratios
    }

    # 3. Behavioral metrics from decision tracking
    behavioral: Optional[BehavioralMetrics] = None
    if decision_context is not None:
        behavioral = _extract_behavioral_metrics(decision_context)

    # 4. Evolutionary context
    evolutionary: Optional[EvolutionaryContext] = None
    if evolution_context is not None:
        evolutionary = _extract_evolutionary_context(evolution_context, genome)

    # 5. Available heuristics
    available_heuristics = _get_available_heuristics()

    # 6. Parameter bounds
    parameter_bounds = _get_parameter_bounds(evolution_context)

    return {
        "id": genome.id,
        "performance": performance,
        "current_config": config,
        "behavioral": behavioral,
        "evolutionary": evolutionary,
        "available_heuristics": available_heuristics,
        "parameter_bounds": parameter_bounds,
    }


def _extract_behavioral_metrics(decision_context: Any) -> BehavioralMetrics:
    """Extract behavioral metrics from QueryDecisionContext."""
    behavioral: BehavioralMetrics = {}

    # Handle dict directly (from to_summary_dict output stored on genome)
    if isinstance(decision_context, dict):
        summary = decision_context
        if "trajectory" in summary:
            t = summary["trajectory"]
            behavioral["unique_nodes_ratio"] = t.get("unique_nodes_ratio", 0.0)
            behavioral["revisit_rate"] = t.get("revisit_rate", 0.0)
            behavioral["dead_end_rate"] = t.get("dead_end_rate", 0.0)
            behavioral["avg_branching_factor"] = t.get("avg_branching_factor", 0.0)
            behavioral["convergence_step"] = t.get("convergence_step")
            behavioral["final_dispersion"] = t.get("final_dispersion", 0.0)

        if "heuristic_usage" in summary:
            behavioral["heuristic_usage"] = summary["heuristic_usage"]

        if "choice_patterns" in summary:
            behavioral["choice_patterns"] = summary["choice_patterns"]

        # Extract enhanced traversal context
        if "sample_paths" in summary:
            behavioral["sample_paths"] = summary["sample_paths"]

        if "node_hotspots" in summary:
            behavioral["node_hotspots"] = summary["node_hotspots"]

        if "stuck_nodes" in summary:
            behavioral["stuck_nodes"] = summary["stuck_nodes"]

        return behavioral

    # Use to_summary_dict if available (from DecisionTracker)
    if hasattr(decision_context, 'to_summary_dict'):
        # This is a DecisionTracker
        summary = decision_context.to_summary_dict()
        if "trajectory" in summary:
            t = summary["trajectory"]
            behavioral["unique_nodes_ratio"] = t.get("unique_nodes_ratio", 0.0)
            behavioral["revisit_rate"] = t.get("revisit_rate", 0.0)
            behavioral["dead_end_rate"] = t.get("dead_end_rate", 0.0)
            behavioral["avg_branching_factor"] = t.get("avg_branching_factor", 0.0)
            behavioral["convergence_step"] = t.get("convergence_step")
            behavioral["final_dispersion"] = t.get("final_dispersion", 0.0)

        if "heuristic_usage" in summary:
            behavioral["heuristic_usage"] = summary["heuristic_usage"]

        if "choice_patterns" in summary:
            behavioral["choice_patterns"] = summary["choice_patterns"]

        # Extract enhanced traversal context
        if "sample_paths" in summary:
            behavioral["sample_paths"] = summary["sample_paths"]

        if "node_hotspots" in summary:
            behavioral["node_hotspots"] = summary["node_hotspots"]

        if "stuck_nodes" in summary:
            behavioral["stuck_nodes"] = summary["stuck_nodes"]

    # Handle QueryDecisionContext directly
    elif hasattr(decision_context, 'trajectory_metrics') and decision_context.trajectory_metrics:
        tm = decision_context.trajectory_metrics
        total_steps = max(tm.total_steps, 1)
        behavioral["unique_nodes_ratio"] = tm.unique_nodes_visited / total_steps
        behavioral["revisit_rate"] = tm.revisit_count / total_steps
        behavioral["dead_end_rate"] = tm.dead_end_count / total_steps
        behavioral["avg_branching_factor"] = tm.avg_branching_factor
        behavioral["convergence_step"] = tm.convergence_step
        behavioral["final_dispersion"] = tm.final_dispersion

        if hasattr(decision_context, 'heuristic_stats'):
            behavioral["heuristic_usage"] = decision_context.heuristic_stats

    return behavioral


def _extract_evolutionary_context(
    evolution_context: Any,
    genome: Genome
) -> EvolutionaryContext:
    """Extract evolutionary context from EvolutionContext."""
    evolutionary: EvolutionaryContext = {}

    if hasattr(evolution_context, 'generation'):
        evolutionary["generation"] = evolution_context.generation

    if hasattr(evolution_context, 'population'):
        evolutionary["population_size"] = len(evolution_context.population)

    if hasattr(genome, 'mutation_rate'):
        evolutionary["mutation_rate"] = genome.mutation_rate

    # Archive fill rate would need to be passed from orchestrator
    if hasattr(evolution_context, 'archive_fill_rate'):
        evolutionary["archive_fill_rate"] = evolution_context.archive_fill_rate

    return evolutionary


def _get_available_heuristics() -> AvailableHeuristics:
    """Get list of available heuristics from registry."""
    try:
        from ...core.heuristics import HeuristicRegistry
        # Convert HeuristicKey enums to strings
        return {
            "movement": [str(k) for k in HeuristicRegistry.all_movement().keys()],
            "ranking": [str(k) for k in HeuristicRegistry.all_ranking().keys()],
            "deposit": [str(k) for k in HeuristicRegistry.all_deposit().keys()],
        }
    except ImportError:
        # Fallback to known heuristics
        return {
            "movement": [
                "semantic_similarity",
                "node_centrality",
                "pheromone_repulsion",
                "random_jitter"
            ],
            "ranking": [
                "percentage_visited",
                "semantic_rank"
            ],
            "deposit": [
                "flat",
                "hub",
                "semantic",
                "exploration_bonus",
                "collaborative_amplification"
            ],
        }


def _get_parameter_bounds(evolution_context: Any) -> ParameterBounds:
    """Get valid parameter ranges."""
    # Try to get from config
    if evolution_context is not None and hasattr(evolution_context, 'config'):
        config = evolution_context.config
        if hasattr(config, 'genetic') and hasattr(config.genetic, 'param_ranges'):
            pr = config.genetic.param_ranges
            return {
                "n_agents": getattr(pr, 'n_agents', (5, 30)),
                "steps": getattr(pr, 'steps', (4, 12)),
                "decay": getattr(pr, 'decay', (0.85, 0.99)),
                "initial_pool_size": getattr(pr, 'initial_pool_size', (10, 50)),
                "start_subset": getattr(pr, 'start_subset', (5, 15)),
                "drop_zone_inc": getattr(pr, 'drop_zone_inc', (0.05, 0.2)),
            }

    # Default bounds
    return {
        "n_agents": (5, 30),
        "steps": (4, 12),
        "decay": (0.85, 0.99),
        "initial_pool_size": (10, 50),
        "start_subset": (5, 15),
        "drop_zone_inc": (0.05, 0.2),
    }
