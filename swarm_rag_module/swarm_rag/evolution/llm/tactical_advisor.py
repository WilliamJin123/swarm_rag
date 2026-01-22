"""
Tactical Advisor (Tier 2) - Per-genome diagnosis and intent prescription.

The Tactical Advisor operates on individual genomes, diagnosing issues and
prescribing mutation intents. It does NOT generate values - it outputs intents
that the Constrained Executor will translate into actual changes.

Input: Simplified genome performance + behavioral signature + strategic directive
Output: Diagnosis + MutationIntent + target component + confidence

LLM calls: get_prescription() -> LLMClient.call()
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from .intents import (
    MutationIntent,
    TargetComponent,
    StrategicDirective,
    MutationPrescription,
    EvolutionMode,
)
from .evolution_journal import EvolutionJournal

if TYPE_CHECKING:
    from .client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class TacticalContext:
    """
    Simplified context for per-genome decisions.

    This is MUCH simpler than the full genome context - only essential metrics.
    """
    genome_id: str

    # Only 3 key metrics (simplified from 10+)
    quality_score: float
    cost_score: float
    recall_at_k: float

    # Aggregated behavioral signature (human-readable)
    behavioral_signature: str  # e.g., "high_revisit,low_dispersion"

    # Current strategy summary (human-readable)
    strategy_summary: str  # e.g., "movement: semantic(0.6) + pheromone(0.3)"

    # Strategic directive from Tier 1
    current_mode: EvolutionMode
    focus_component: TargetComponent
    exploration_temperature: float

    # Historical context (from journal)
    lineage_intents: List[str] = field(default_factory=list)
    lineage_successes: List[bool] = field(default_factory=list)

    # Failed intents to avoid
    failed_intents_global: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "quality_score": self.quality_score,
            "cost_score": self.cost_score,
            "recall_at_k": self.recall_at_k,
            "behavioral_signature": self.behavioral_signature,
            "strategy_summary": self.strategy_summary,
            "current_mode": self.current_mode.value,
            "focus_component": self.focus_component.value,
            "exploration_temperature": self.exploration_temperature,
            "lineage_intents": self.lineage_intents,
            "lineage_successes": self.lineage_successes,
            "failed_intents_global": self.failed_intents_global,
        }


class TacticalAdvisor:
    """
    Tier 2: Per-genome diagnosis and intent prescription.

    Diagnoses individual genomes and prescribes mutation intents.
    Does NOT generate actual values - that's the Executor's job.

    LLM calls made via: get_prescription() -> self.client.call()
    """

    def __init__(self, client: "LLMClient"):
        """
        Initialize the Tactical Advisor.

        Args:
            client: LLMClient instance for API calls
        """
        self.client = client

    def get_prescription(
        self,
        context: TacticalContext,
    ) -> MutationPrescription:
        """
        Get mutation prescription for a genome.

        LLM call: self.client.call(system_prompt, user_prompt)

        Args:
            context: Tactical context with genome metrics

        Returns:
            MutationPrescription with diagnosis and intent
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(context)

        result = self.client.call(system_prompt, user_prompt)

        if not result.success or result.parsed is None:
            logger.warning(
                f"Tactical Advisor LLM call failed for {context.genome_id}: "
                f"{result.error}. Using heuristic fallback."
            )
            return self._heuristic_fallback(context)

        prescription = self._parse_response(result.parsed)

        logger.debug(
            f"Tactical Advisor {context.genome_id}: "
            f"Intent={prescription.primary_intent.value}, "
            f"Confidence={prescription.confidence:.2f}"
        )

        return prescription

    def _build_system_prompt(self) -> str:
        """Build system prompt for tactical decisions."""
        return """You are a Tactical Mutation Advisor for swarm-based retrieval agents.

## Your Role
You diagnose individual genome performance and prescribe mutation INTENTS.
You do NOT specify exact values - only what kind of change is needed.

## Your Task
Given a genome's performance metrics and behavioral signature:
1. Diagnose the root cause of any performance issues
2. Prescribe a mutation intent (what to improve)
3. Specify which component to target
4. Express your confidence (affects change magnitude)

## Available Intents
- `increase_exploration`: More coverage, more agents, more steps
- `reduce_exploration`: Focus on known good areas
- `reduce_loops`: Decrease revisit rate
- `reduce_revisits`: Similar to reduce_loops
- `improve_coverage`: Better spread of exploration
- `increase_dispersion`: Agents end up more spread out
- `reduce_cost`: Fewer agents/steps
- `reduce_latency`: Faster execution
- `improve_quality`: Better recall/precision
- `increase_semantic_focus`: Prioritize semantic similarity
- `avoid_dead_ends`: Don't get stuck at leaf nodes
- `improve_connectivity`: Prefer well-connected nodes
- `balance_groups`: Adjust agent group ratios
- `rebalance_heuristics`: Adjust heuristic weights
- `slow_convergence`: Agents shouldn't cluster too fast
- `speed_convergence`: Cluster faster on good results
- `no_change`: Genome is performing adequately

## Target Components
- `params`: Hyperparameters (n_agents, steps, decay)
- `movement`: Movement strategy expression
- `deposit`: Deposit strategy expression
- `ranking`: Ranking strategy expression
- `ratios`: Agent group ratios
- `all`: No specific preference

## Diagnosis Guidelines

### Behavioral Signatures and Fixes
- `high_revisit` → reduce_loops, reduce_revisits
- `low_dispersion` → increase_dispersion, increase_exploration
- `early_convergence` → slow_convergence, increase_exploration
- `high_dead_end` → avoid_dead_ends, improve_connectivity
- `low_coverage` → improve_coverage, increase_exploration
- `high_cost` → reduce_cost, reduce_exploration
- `low_quality` → improve_quality, increase_semantic_focus

### Confidence Guidelines
- 0.8-1.0: Very clear diagnosis, strong recommendation
- 0.5-0.7: Reasonable diagnosis, moderate change
- 0.2-0.4: Uncertain, try small change
- <0.2: Very uncertain, minimal change or no_change

## IMPORTANT
- Avoid prescribing intents that have historically failed
- Consider the strategic mode (exploit vs explore)
- Don't prescribe multiple conflicting intents
- If genome is performing well (quality > 0.7), consider no_change

## Output Format
Return JSON with exactly these keys:
{
    "diagnosis": "Brief explanation of the root cause",
    "primary_intent": "intent_name",
    "secondary_intent": null | "intent_name",
    "target_component": "params|movement|deposit|ranking|ratios|all",
    "confidence": 0.5
}"""

    def _build_user_prompt(self, context: TacticalContext) -> str:
        """Build user prompt with tactical context."""
        # Build readable metrics
        quality_rating = "GOOD" if context.quality_score > 0.7 else "MODERATE" if context.quality_score > 0.4 else "POOR"
        cost_rating = "LOW" if context.cost_score < 0.3 else "MODERATE" if context.cost_score < 0.6 else "HIGH"

        prompt = f"""## Genome: {context.genome_id}

**Performance** (rating in parentheses):
- Quality Score: {context.quality_score:.4f} ({quality_rating})
- Cost Score: {context.cost_score:.4f} ({cost_rating})
- Recall@K: {context.recall_at_k:.4f}

**Behavioral Signature**: {context.behavioral_signature or "normal"}

**Current Strategy Summary**: {context.strategy_summary}

**Strategic Context**:
- Mode: {context.current_mode.value}
- Focus: {context.focus_component.value}
- Temperature: {context.exploration_temperature:.2f}
"""

        if context.lineage_intents:
            prompt += f"""
**Lineage History** (recent mutations):
- Intents tried: {context.lineage_intents[-3:]}
- Success: {context.lineage_successes[-3:] if context.lineage_successes else 'N/A'}
"""

        if context.failed_intents_global:
            prompt += f"""
**Globally Failed Intents** (avoid these): {context.failed_intents_global}
"""

        prompt += """
**Task**: Diagnose this genome and prescribe a mutation intent.
Return your prescription as JSON."""

        return prompt

    def _parse_response(self, data: Dict[str, Any]) -> MutationPrescription:
        """Parse LLM response into MutationPrescription."""
        primary_str = data.get("primary_intent", "no_change")
        try:
            primary = MutationIntent(primary_str)
        except ValueError:
            logger.warning(f"Unknown intent '{primary_str}', defaulting to no_change")
            primary = MutationIntent.NO_CHANGE

        secondary = None
        secondary_str = data.get("secondary_intent")
        if secondary_str:
            try:
                secondary = MutationIntent(secondary_str)
            except ValueError:
                pass

        target_str = data.get("target_component", "all")
        try:
            target = TargetComponent(target_str)
        except ValueError:
            target = TargetComponent.ALL

        confidence = data.get("confidence", 0.5)
        confidence = max(0.0, min(1.0, float(confidence)))

        return MutationPrescription(
            diagnosis=data.get("diagnosis", "No diagnosis provided"),
            primary_intent=primary,
            secondary_intent=secondary,
            target_component=target,
            confidence=confidence,
        )

    def _heuristic_fallback(self, context: TacticalContext) -> MutationPrescription:
        """
        Heuristic fallback when LLM is unavailable.

        Uses simple rules based on behavioral signature.
        """
        signature = context.behavioral_signature.lower()

        # Parse behavioral issues from signature
        if "high_revisit" in signature or "loop" in signature:
            return MutationPrescription(
                diagnosis="High revisit rate detected",
                primary_intent=MutationIntent.REDUCE_LOOPS,
                target_component=TargetComponent.MOVEMENT,
                confidence=0.6,
            )

        if "low_dispersion" in signature or "clustered" in signature:
            return MutationPrescription(
                diagnosis="Low dispersion - agents clustering",
                primary_intent=MutationIntent.INCREASE_DISPERSION,
                target_component=TargetComponent.MOVEMENT,
                confidence=0.6,
            )

        if "dead_end" in signature:
            return MutationPrescription(
                diagnosis="Dead-end rate too high",
                primary_intent=MutationIntent.AVOID_DEAD_ENDS,
                target_component=TargetComponent.MOVEMENT,
                confidence=0.5,
            )

        if "early_convergence" in signature:
            return MutationPrescription(
                diagnosis="Agents converging too quickly",
                primary_intent=MutationIntent.SLOW_CONVERGENCE,
                target_component=TargetComponent.MOVEMENT,
                confidence=0.5,
            )

        # Quality-based fallback
        if context.quality_score < 0.3:
            return MutationPrescription(
                diagnosis="Low quality score",
                primary_intent=MutationIntent.IMPROVE_QUALITY,
                target_component=TargetComponent.ALL,
                confidence=0.5,
            )

        if context.cost_score > 0.7:
            return MutationPrescription(
                diagnosis="High cost score",
                primary_intent=MutationIntent.REDUCE_COST,
                target_component=TargetComponent.PARAMS,
                confidence=0.5,
            )

        # No clear issue - consider mode
        if context.current_mode == EvolutionMode.EXPLORE_STRATEGIES:
            return MutationPrescription(
                diagnosis="Exploring strategies per directive",
                primary_intent=MutationIntent.REBALANCE_HEURISTICS,
                target_component=TargetComponent.MOVEMENT,
                confidence=0.4,
            )

        if context.current_mode == EvolutionMode.EXPLORE_PARAMS:
            return MutationPrescription(
                diagnosis="Exploring parameters per directive",
                primary_intent=MutationIntent.INCREASE_EXPLORATION,
                target_component=TargetComponent.PARAMS,
                confidence=0.4,
            )

        # Default - small improvement attempt
        return MutationPrescription(
            diagnosis="No clear issue, attempting general improvement",
            primary_intent=MutationIntent.IMPROVE_QUALITY,
            target_component=TargetComponent.ALL,
            confidence=0.3,
        )


def build_tactical_context(
    genome: Any,  # Genome
    directive: StrategicDirective,
    journal: Optional[EvolutionJournal] = None,
) -> TacticalContext:
    """
    Build TacticalContext from a genome.

    Args:
        genome: The genome to analyze
        directive: Current strategic directive
        journal: Evolution journal for history

    Returns:
        TacticalContext for the advisor
    """
    # Build behavioral signature from genome attributes
    behavioral_parts = []

    decision_context = getattr(genome, 'decision_context', None)
    if decision_context:
        if isinstance(decision_context, dict):
            trajectory = decision_context.get('trajectory', {})
            revisit_rate = trajectory.get('revisit_rate', 0)
            dispersion = trajectory.get('final_dispersion', 0.5)
            dead_end_rate = trajectory.get('dead_end_rate', 0)
            convergence_step = trajectory.get('convergence_step')

            if revisit_rate > 0.15:
                behavioral_parts.append("high_revisit")
            if dispersion < 0.3:
                behavioral_parts.append("low_dispersion")
            if dead_end_rate > 0.1:
                behavioral_parts.append("high_dead_end")
            if convergence_step is not None and convergence_step < 2:
                behavioral_parts.append("early_convergence")

    behavioral_signature = ",".join(behavioral_parts) if behavioral_parts else "normal"

    # Build strategy summary
    strategy_parts = []
    for name, tree in genome.strategies.items():
        if hasattr(tree, 'to_string'):
            strategy_str = tree.to_string()
            # Truncate long expressions
            if len(strategy_str) > 50:
                strategy_str = strategy_str[:47] + "..."
            strategy_parts.append(f"{name}: {strategy_str}")

    strategy_summary = "; ".join(strategy_parts[:2])  # Limit to 2 strategies

    # Get journal context
    lineage_intents = []
    lineage_successes = []
    failed_intents = []

    if journal:
        tactical_summary = journal.summarize_for_tactical(genome.id)
        lineage_intents = tactical_summary.get("lineage_intents", [])
        lineage_successes = tactical_summary.get("lineage_successes", [])
        failed_intents = tactical_summary.get("failed_intents_global", [])

    return TacticalContext(
        genome_id=genome.id,
        quality_score=genome.fitness.quality_score,
        cost_score=genome.fitness.cost_score,
        recall_at_k=genome.metrics.get("Recall@20", 0.0),
        behavioral_signature=behavioral_signature,
        strategy_summary=strategy_summary,
        current_mode=directive.mode,
        focus_component=directive.focus_component,
        exploration_temperature=directive.exploration_temperature,
        lineage_intents=lineage_intents,
        lineage_successes=lineage_successes,
        failed_intents_global=failed_intents,
    )
