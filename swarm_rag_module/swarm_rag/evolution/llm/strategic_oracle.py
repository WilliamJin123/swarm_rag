"""
Strategic Oracle (Tier 1) - High-level evolution steering.

The Strategic Oracle operates at the archive/population level, making decisions
about the overall direction of evolution every N generations or when stagnation
is detected.

Input: Archive statistics, QD trends, historical success rates
Output: EvolutionMode + focus area + exploration temperature

LLM calls: get_directive() -> LLMClient.call()
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from .intents import (
    EvolutionMode,
    TargetComponent,
    StrategicDirective,
)
from .evolution_journal import EvolutionJournal

if TYPE_CHECKING:
    from .client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class StrategicContext:
    """
    Aggregated context for strategic decisions.

    This is what Tier 1 receives - archive-level aggregates, not per-genome data.
    """
    # Archive state
    archive_fill_rate: float  # % of cells occupied
    qd_score: float  # Sum of all elite fitnesses
    max_fitness: float  # Best fitness in archive
    filled_cells: int  # Number of occupied cells
    total_cells: int  # Total possible cells

    # Trends
    qd_score_trend: str  # "improving", "stagnant", "declining"
    qd_score_slope: float  # Numerical slope
    stagnation_generations: int  # Consecutive stagnant generations

    # Diversity metrics
    param_diversity: float  # Variance in parameters across elites
    strategy_diversity: float  # Variance in strategy structures

    # Historical success rates
    successful_intents: Dict[str, float]  # intent -> success rate
    failed_intents: Dict[str, float]  # intent -> (low) success rate

    # Recent history
    recent_qd_scores: List[float]  # Last N QD scores
    recent_failed_diagnoses: List[str]  # What hasn't worked

    # Generation info
    current_generation: int
    total_generations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "archive_fill_rate": self.archive_fill_rate,
            "qd_score": self.qd_score,
            "max_fitness": self.max_fitness,
            "filled_cells": self.filled_cells,
            "total_cells": self.total_cells,
            "qd_score_trend": self.qd_score_trend,
            "qd_score_slope": self.qd_score_slope,
            "stagnation_generations": self.stagnation_generations,
            "param_diversity": self.param_diversity,
            "strategy_diversity": self.strategy_diversity,
            "successful_intents": self.successful_intents,
            "failed_intents": self.failed_intents,
            "recent_qd_scores": self.recent_qd_scores,
            "recent_failed_diagnoses": self.recent_failed_diagnoses,
            "current_generation": self.current_generation,
            "total_generations": self.total_generations,
        }


class StrategicOracle:
    """
    Tier 1: Strategic steering of evolution.

    Decides the overall direction of evolution based on archive-level metrics.
    Called periodically (every N generations) or when stagnation is detected.

    LLM calls made via: get_directive() -> self.client.call()
    """

    def __init__(
        self,
        client: "LLMClient",
        call_interval: int = 5,
        stagnation_threshold: int = 3,
    ):
        """
        Initialize the Strategic Oracle.

        Args:
            client: LLMClient instance for API calls
            call_interval: Generations between strategic calls
            stagnation_threshold: Stagnant generations before forced call
        """
        self.client = client
        self.call_interval = call_interval
        self.stagnation_threshold = stagnation_threshold
        self._current_directive: Optional[StrategicDirective] = None
        self._last_call_generation: int = -1

    def should_call(self, generation: int, stagnation_count: int) -> bool:
        """
        Determine if the oracle should be called this generation.

        Args:
            generation: Current generation number
            stagnation_count: Number of stagnant generations

        Returns:
            True if oracle should be invoked
        """
        # First generation
        if self._last_call_generation < 0:
            return True

        # Regular interval
        if generation - self._last_call_generation >= self.call_interval:
            return True

        # Stagnation detected
        if stagnation_count >= self.stagnation_threshold:
            return True

        return False

    def get_directive(
        self,
        context: StrategicContext,
        journal: Optional[EvolutionJournal] = None,
    ) -> StrategicDirective:
        """
        Get strategic directive from the LLM.

        LLM call: self.client.call(system_prompt, user_prompt)

        Args:
            context: Strategic context with archive stats
            journal: Evolution journal for history

        Returns:
            StrategicDirective with mode and focus
        """
        self._last_call_generation = context.current_generation

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(context, journal)

        result = self.client.call(system_prompt, user_prompt)

        if not result.success or result.parsed is None:
            logger.warning(
                f"Strategic Oracle LLM call failed: {result.error}. "
                "Using heuristic fallback."
            )
            return self._heuristic_fallback(context)

        directive = self._parse_response(result.parsed)
        self._current_directive = directive

        logger.info(
            f"Strategic Oracle Gen {context.current_generation}: "
            f"Mode={directive.mode.value}, Focus={directive.focus_component.value}"
        )

        return directive

    def get_current_directive(self) -> StrategicDirective:
        """
        Get the current directive without calling LLM.

        Returns:
            Current directive or default if none set
        """
        if self._current_directive is None:
            return StrategicDirective.default()
        return self._current_directive

    def _build_system_prompt(self) -> str:
        """Build system prompt for strategic decisions."""
        return """You are a Strategic Evolution Advisor for a MAP-Elites quality-diversity optimization system.

## Your Role
You make high-level decisions about evolution direction based on archive-level metrics and trends.
You do NOT make per-genome decisions - that's handled by another system.

## Your Task
Based on the current evolution state, decide:
1. **Evolution Mode** - Overall strategy for the next period
2. **Focus Component** - What to prioritize mutating
3. **Exploration Temperature** - How aggressive to be (0.0-1.0)
4. **Priority Problems** - What issues to address first

## Available Modes
- `explore_params`: Focus on tuning hyperparameters (n_agents, steps, decay, etc.)
- `explore_strategies`: Focus on expression/heuristic combinations
- `exploit_top`: Refine what's already working well
- `diversify`: Try to fill more archive cells, increase coverage
- `targeted_fix`: Address a specific diagnosed problem
- `balanced`: No strong preference, mix of exploration and exploitation

## Available Focus Components
- `params`: Hyperparameters (n_agents, steps, decay, etc.)
- `movement`: Movement strategy expressions
- `deposit`: Deposit strategy expressions
- `ranking`: Ranking strategy expressions
- `ratios`: Agent group ratios
- `all`: No specific focus

## Decision Guidelines

### When to EXPLORE_PARAMS
- QD score improving but slowly
- Strategies seem good but parameters might be suboptimal
- Low stagnation but want faster progress

### When to EXPLORE_STRATEGIES
- Parameters seem fine but quality not improving
- High cost scores suggest inefficient exploration
- Behavioral issues (high revisits, low dispersion)

### When to EXPLOIT_TOP
- Good progress being made
- Archive filling steadily
- Want to maximize current best cells

### When to DIVERSIFY
- Low archive coverage (<30%)
- QD score stagnant
- Top fitness is good but want more diverse solutions

### When to do TARGETED_FIX
- Clear pattern in failed diagnoses
- Specific behavioral issue (revisit loops, dead-ends)
- Recent mutations failing for same reason

### Exploration Temperature
- 0.0-0.3: Conservative, small changes
- 0.4-0.6: Moderate exploration
- 0.7-1.0: Aggressive changes, high diversity

## Output Format
Return JSON with exactly these keys:
{
    "mode": "explore_params|explore_strategies|exploit_top|diversify|targeted_fix|balanced",
    "focus_component": "params|movement|deposit|ranking|ratios|all",
    "exploration_temperature": 0.5,
    "priority_problems": ["problem1", "problem2"],
    "rationale": "Brief explanation of decision"
}"""

    def _build_user_prompt(
        self,
        context: StrategicContext,
        journal: Optional[EvolutionJournal],
    ) -> str:
        """Build user prompt with strategic context."""
        journal_summary = {}
        if journal:
            journal_summary = journal.summarize_for_strategic()

        progress_pct = context.current_generation / max(context.total_generations, 1) * 100

        prompt = f"""## Current Evolution State

**Progress**: Generation {context.current_generation}/{context.total_generations} ({progress_pct:.1f}%)

**Archive Metrics**:
- Fill Rate: {context.archive_fill_rate:.1%} ({context.filled_cells}/{context.total_cells} cells)
- QD Score: {context.qd_score:.4f}
- Max Fitness: {context.max_fitness:.4f}

**Trends**:
- QD Trend: {context.qd_score_trend} (slope: {context.qd_score_slope:.6f})
- Stagnation: {context.stagnation_generations} generations
- Recent QD Scores: {context.recent_qd_scores[-5:] if context.recent_qd_scores else 'N/A'}

**Diversity**:
- Parameter Diversity: {context.param_diversity:.4f}
- Strategy Diversity: {context.strategy_diversity:.4f}

**Historical Success Rates**:
- Successful Intents: {context.successful_intents}
- Failed Intents: {context.failed_intents}

**Recent Failed Diagnoses**:
{chr(10).join('- ' + d for d in context.recent_failed_diagnoses[:5]) if context.recent_failed_diagnoses else '- None'}
"""

        if journal_summary:
            prompt += f"""
**Journal Summary**:
- Total Mutations Tracked: {journal_summary.get('total_mutations', 0)}
- QD Score Trend: {journal_summary.get('qd_score_trend', 'unknown')}
"""

        prompt += """
**Task**: Based on the above state, decide the strategic direction for the next evolution period.
Return your decision as JSON."""

        return prompt

    def _parse_response(self, data: Dict[str, Any]) -> StrategicDirective:
        """Parse LLM response into StrategicDirective."""
        mode_str = data.get("mode", "balanced")
        try:
            mode = EvolutionMode(mode_str)
        except ValueError:
            logger.warning(f"Unknown mode '{mode_str}', defaulting to balanced")
            mode = EvolutionMode.BALANCED

        focus_str = data.get("focus_component", "all")
        try:
            focus = TargetComponent(focus_str)
        except ValueError:
            logger.warning(f"Unknown focus '{focus_str}', defaulting to all")
            focus = TargetComponent.ALL

        return StrategicDirective(
            mode=mode,
            focus_component=focus,
            priority_problems=data.get("priority_problems", []),
            exploration_temperature=data.get("exploration_temperature", 0.5),
            rationale=data.get("rationale", ""),
        )

    def _heuristic_fallback(self, context: StrategicContext) -> StrategicDirective:
        """
        Heuristic fallback when LLM is unavailable.

        Uses simple rules based on evolution state.
        """
        # High stagnation -> diversify
        if context.stagnation_generations >= 5:
            return StrategicDirective(
                mode=EvolutionMode.DIVERSIFY,
                focus_component=TargetComponent.ALL,
                exploration_temperature=0.8,
                rationale="High stagnation detected, diversifying",
            )

        # Low coverage -> diversify
        if context.archive_fill_rate < 0.2:
            return StrategicDirective(
                mode=EvolutionMode.DIVERSIFY,
                focus_component=TargetComponent.MOVEMENT,
                exploration_temperature=0.7,
                rationale="Low coverage, exploring movement strategies",
            )

        # Declining trend -> targeted fix
        if context.qd_score_trend == "declining":
            return StrategicDirective(
                mode=EvolutionMode.TARGETED_FIX,
                focus_component=TargetComponent.PARAMS,
                exploration_temperature=0.4,
                rationale="Declining trend, fixing parameters",
            )

        # Good progress -> exploit
        if context.qd_score_trend == "improving" and context.archive_fill_rate > 0.5:
            return StrategicDirective(
                mode=EvolutionMode.EXPLOIT_TOP,
                focus_component=TargetComponent.ALL,
                exploration_temperature=0.3,
                rationale="Good progress, exploiting current solutions",
            )

        # Default balanced
        return StrategicDirective(
            mode=EvolutionMode.BALANCED,
            focus_component=TargetComponent.ALL,
            exploration_temperature=0.5,
            rationale="Default balanced strategy",
        )


def build_strategic_context(
    archive_stats: Dict[str, Any],
    journal: Optional[EvolutionJournal],
    generation: int,
    total_generations: int,
    population: Optional[List[Any]] = None,
) -> StrategicContext:
    """
    Build StrategicContext from archive stats and journal.

    Args:
        archive_stats: Stats from MapElitesArchive.stats()
        journal: Evolution journal
        generation: Current generation
        total_generations: Total planned generations
        population: Optional list of genomes for diversity calculation

    Returns:
        StrategicContext for the oracle
    """
    # Get journal summary
    journal_summary = journal.summarize_for_strategic() if journal else {}

    # Calculate diversity metrics if population provided
    param_diversity = 0.0
    strategy_diversity = 0.0
    if population and len(population) > 1:
        import numpy as np
        # Simple variance of n_agents and steps as proxy for param diversity
        n_agents_vals = [g.params.get("n_agents", 10) for g in population]
        steps_vals = [g.params.get("steps", 4) for g in population]
        param_diversity = (np.var(n_agents_vals) + np.var(steps_vals)) / 2

        # Strategy diversity as number of unique template structures
        strategy_strs = set()
        for g in population:
            for s in g.strategies.values():
                strategy_strs.add(s.to_string() if hasattr(s, 'to_string') else str(s))
        strategy_diversity = len(strategy_strs) / max(len(population), 1)

    # Get trends from journal
    qd_trend = journal_summary.get("qd_score_trend", "unknown")
    qd_slope = journal_summary.get("qd_score_slope", 0.0)
    stagnation = journal_summary.get("stagnation_generations", 0)

    return StrategicContext(
        archive_fill_rate=archive_stats.get("coverage", 0.0),
        qd_score=archive_stats.get("qd_score", 0.0),
        max_fitness=archive_stats.get("max_fitness", 0.0),
        filled_cells=archive_stats.get("filled_cells", 0),
        total_cells=int(archive_stats.get("coverage", 0.01) ** -1 * archive_stats.get("filled_cells", 1))
        if archive_stats.get("coverage", 0) > 0 else 100,
        qd_score_trend=qd_trend,
        qd_score_slope=qd_slope,
        stagnation_generations=stagnation,
        param_diversity=param_diversity,
        strategy_diversity=strategy_diversity,
        successful_intents=journal_summary.get("successful_intents", {}),
        failed_intents=journal_summary.get("failed_intents", {}),
        recent_qd_scores=journal_summary.get("recent_qd_scores", []),
        recent_failed_diagnoses=journal_summary.get("recent_failed_diagnoses", []),
        current_generation=generation,
        total_generations=total_generations,
    )
