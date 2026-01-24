"""
Mutation intents and intent-to-action mappings for the three-tier architecture.

This module defines:
1. MutationIntent enum - What the LLM wants to achieve
2. EvolutionMode enum - Strategic steering modes
3. Intent-action mappings - How intents translate to parameter/expression changes
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any


class MutationIntent(Enum):
    """
    High-level mutation intents that the Tactical Advisor can prescribe.

    These represent WHAT we want to achieve, not HOW to achieve it.
    The Constrained Executor translates these into actual parameter changes.

    Simplified from 17 intents to 7 core intents:
    - EXPLORE: More agents, steps, coverage, dispersion
    - EXPLOIT: Focus on semantics, reduce exploration
    - REDUCE_LOOPS: Decrease revisit rate, pheromone repulsion
    - REDUCE_COST: Fewer agents/steps, efficient execution
    - IMPROVE_QUALITY: Better recall, semantic focus
    - IMPROVE_CONNECTIVITY: Avoid dead-ends, prefer hubs
    - NO_CHANGE: Genome performing well
    """
    # Exploration (combines: increase_exploration, improve_coverage, increase_dispersion)
    EXPLORE = "explore"

    # Exploitation (combines: reduce_exploration, increase_semantic_focus, speed_convergence)
    EXPLOIT = "exploit"

    # Loop reduction (combines: reduce_loops, reduce_revisits, slow_convergence)
    REDUCE_LOOPS = "reduce_loops"

    # Cost reduction (combines: reduce_cost, reduce_latency)
    REDUCE_COST = "reduce_cost"

    # Quality improvement
    IMPROVE_QUALITY = "improve_quality"

    # Connectivity (combines: avoid_dead_ends, improve_connectivity)
    IMPROVE_CONNECTIVITY = "improve_connectivity"

    # No change needed
    NO_CHANGE = "no_change"

    # Legacy aliases for backwards compatibility
    @classmethod
    def from_legacy(cls, value: str) -> "MutationIntent":
        """Convert legacy intent names to new simplified intents."""
        LEGACY_MAP = {
            "increase_exploration": cls.EXPLORE,
            "improve_coverage": cls.EXPLORE,
            "increase_dispersion": cls.EXPLORE,
            "reduce_exploration": cls.EXPLOIT,
            "increase_semantic_focus": cls.EXPLOIT,
            "speed_convergence": cls.EXPLOIT,
            "reduce_revisits": cls.REDUCE_LOOPS,
            "slow_convergence": cls.REDUCE_LOOPS,
            "reduce_latency": cls.REDUCE_COST,
            "avoid_dead_ends": cls.IMPROVE_CONNECTIVITY,
            "balance_groups": cls.IMPROVE_QUALITY,
            "rebalance_heuristics": cls.IMPROVE_QUALITY,
        }
        if value in LEGACY_MAP:
            return LEGACY_MAP[value]
        try:
            return cls(value)
        except ValueError:
            return cls.NO_CHANGE


class EvolutionMode(Enum):
    """
    Strategic evolution modes set by the Strategic Oracle.

    These determine the overall direction of evolution for a period.
    """
    # Parameter exploration - focus on hyperparameter tuning
    EXPLORE_PARAMS = "explore_params"

    # Strategy exploration - focus on expression/heuristic combinations
    EXPLORE_STRATEGIES = "explore_strategies"

    # Exploitation - refine what's already working
    EXPLOIT_TOP = "exploit_top"

    # Diversity - try to fill more archive cells
    DIVERSIFY = "diversify"

    # Targeted fix - address a specific diagnosed problem
    TARGETED_FIX = "targeted_fix"

    # Balanced - no strong preference
    BALANCED = "balanced"


class TargetComponent(Enum):
    """
    Components that can be targeted for mutation.
    """
    PARAMS = "params"
    MOVEMENT = "movement"
    DEPOSIT = "deposit"
    RANKING = "ranking"
    RATIOS = "ratios"
    ALL = "all"


@dataclass
class StrategicDirective:
    """
    Output from the Strategic Oracle (Tier 1).

    Provides high-level steering for the evolution process.
    """
    mode: EvolutionMode
    focus_component: TargetComponent
    priority_problems: List[str] = field(default_factory=list)
    exploration_temperature: float = 0.5  # 0.0 = conservative, 1.0 = aggressive
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "focus_component": self.focus_component.value,
            "priority_problems": self.priority_problems,
            "exploration_temperature": self.exploration_temperature,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategicDirective":
        return cls(
            mode=EvolutionMode(data.get("mode", "balanced")),
            focus_component=TargetComponent(data.get("focus_component", "all")),
            priority_problems=data.get("priority_problems", []),
            exploration_temperature=data.get("exploration_temperature", 0.5),
            rationale=data.get("rationale", ""),
        )

    @classmethod
    def default(cls) -> "StrategicDirective":
        """Return a default balanced directive."""
        return cls(
            mode=EvolutionMode.BALANCED,
            focus_component=TargetComponent.ALL,
            exploration_temperature=0.5,
        )


@dataclass
class MutationPrescription:
    """
    Output from the Tactical Advisor (Tier 2).

    Describes WHAT to change without specifying exact values.
    """
    diagnosis: str
    primary_intent: MutationIntent
    secondary_intent: Optional[MutationIntent] = None
    target_component: TargetComponent = TargetComponent.ALL
    confidence: float = 0.5  # 0.0-1.0, affects mutation magnitude

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "primary_intent": self.primary_intent.value,
            "secondary_intent": self.secondary_intent.value if self.secondary_intent else None,
            "target_component": self.target_component.value,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MutationPrescription":
        secondary = data.get("secondary_intent")
        return cls(
            diagnosis=data.get("diagnosis", "No diagnosis"),
            primary_intent=MutationIntent(data.get("primary_intent", "no_change")),
            secondary_intent=MutationIntent(secondary) if secondary else None,
            target_component=TargetComponent(data.get("target_component", "all")),
            confidence=data.get("confidence", 0.5),
        )

    @classmethod
    def no_change(cls, diagnosis: str = "Genome is performing adequately") -> "MutationPrescription":
        """Return a prescription that makes no changes."""
        return cls(
            diagnosis=diagnosis,
            primary_intent=MutationIntent.NO_CHANGE,
            confidence=1.0,
        )


@dataclass
class IntentAction:
    """
    Defines how an intent translates to concrete changes.
    """
    # Parameter adjustments: {param_name: delta_value}
    # Positive delta = increase, negative = decrease
    # These are RELATIVE changes, scaled by confidence
    param_adjustments: Dict[str, float] = field(default_factory=dict)

    # Expression template keys to prefer for this intent
    preferred_templates: List[str] = field(default_factory=list)

    # Weight adjustments for specific heuristics
    # {heuristic_name: (min_weight, max_weight)}
    weight_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Description for logging
    description: str = ""


# ============================================================================
# Intent-to-Action Mappings (Simplified - 7 intents)
# ============================================================================

INTENT_ACTIONS: Dict[MutationIntent, IntentAction] = {
    MutationIntent.EXPLORE: IntentAction(
        param_adjustments={
            "n_agents": 4,
            "steps": 1,
            "decay": -0.03,
            "initial_pool_size": 5,
        },
        preferred_templates=["exploration_heavy", "balanced_exploration", "coverage_focused"],
        weight_ranges={
            "pheromone_repulsion": (0.2, 0.4),
            "random_jitter": (0.1, 0.25),
            "semantic_similarity": (0.3, 0.5),
            "node_centrality": (0.1, 0.25),
        },
        description="Increase exploration, coverage, and dispersion",
    ),

    MutationIntent.EXPLOIT: IntentAction(
        param_adjustments={
            "n_agents": -2,
            "steps": -1,
            "decay": 0.03,
        },
        preferred_templates=["semantic_focused", "exploitation"],
        weight_ranges={
            "semantic_similarity": (0.6, 0.85),
            "pheromone_repulsion": (0.05, 0.15),
        },
        description="Focus on semantic exploitation and faster convergence",
    ),

    MutationIntent.REDUCE_LOOPS: IntentAction(
        param_adjustments={
            "decay": -0.05,
        },
        preferred_templates=["anti_loop", "exploration_heavy"],
        weight_ranges={
            "pheromone_repulsion": (0.3, 0.5),
            "random_jitter": (0.1, 0.2),
            "semantic_similarity": (0.3, 0.5),
        },
        description="Reduce revisit loops with stronger pheromone avoidance",
    ),

    MutationIntent.REDUCE_COST: IntentAction(
        param_adjustments={
            "n_agents": -3,
            "steps": -1,
            "initial_pool_size": -5,
        },
        preferred_templates=["efficient", "semantic_focused", "fast"],
        weight_ranges={
            "semantic_similarity": (0.6, 0.8),
        },
        description="Reduce cost and latency by simplifying the search",
    ),

    MutationIntent.IMPROVE_QUALITY: IntentAction(
        param_adjustments={
            "n_agents": 2,
            "steps": 1,
        },
        preferred_templates=["quality_focused", "semantic_focused", "balanced"],
        weight_ranges={
            "semantic_similarity": (0.5, 0.7),
            "node_centrality": (0.1, 0.2),
            "pheromone_repulsion": (0.15, 0.3),
        },
        description="Improve quality through semantic focus and balanced heuristics",
    ),

    MutationIntent.IMPROVE_CONNECTIVITY: IntentAction(
        param_adjustments={
            "initial_pool_size": 5,
            "start_subset": 2,
        },
        preferred_templates=["connectivity_focused", "hub_preferring"],
        weight_ranges={
            "node_centrality": (0.2, 0.45),
            "semantic_similarity": (0.4, 0.6),
        },
        description="Avoid dead-ends by preferring well-connected hub nodes",
    ),

    MutationIntent.NO_CHANGE: IntentAction(
        param_adjustments={},
        preferred_templates=[],
        weight_ranges={},
        description="No changes needed",
    ),
}


def get_intent_action(intent: MutationIntent) -> IntentAction:
    """
    Get the action mapping for an intent.

    Args:
        intent: The mutation intent

    Returns:
        IntentAction with parameter adjustments and template preferences
    """
    return INTENT_ACTIONS.get(intent, IntentAction())


def get_complementary_intents(primary: MutationIntent) -> List[MutationIntent]:
    """
    Get intents that can be safely combined with the primary intent.

    Args:
        primary: The primary mutation intent

    Returns:
        List of complementary intents
    """
    COMPLEMENTARY_MAP = {
        MutationIntent.EXPLORE: [MutationIntent.REDUCE_LOOPS],
        MutationIntent.REDUCE_LOOPS: [MutationIntent.EXPLORE],
        MutationIntent.IMPROVE_QUALITY: [MutationIntent.IMPROVE_CONNECTIVITY],
        MutationIntent.REDUCE_COST: [MutationIntent.EXPLOIT],
        MutationIntent.IMPROVE_CONNECTIVITY: [MutationIntent.EXPLORE],
    }
    return COMPLEMENTARY_MAP.get(primary, [])


def get_conflicting_intents(intent: MutationIntent) -> List[MutationIntent]:
    """
    Get intents that conflict with the given intent.

    Args:
        intent: The mutation intent to check

    Returns:
        List of conflicting intents that should not be combined
    """
    CONFLICT_MAP = {
        MutationIntent.EXPLORE: [MutationIntent.EXPLOIT, MutationIntent.REDUCE_COST],
        MutationIntent.EXPLOIT: [MutationIntent.EXPLORE],
        MutationIntent.REDUCE_COST: [MutationIntent.EXPLORE],
    }
    return CONFLICT_MAP.get(intent, [])
