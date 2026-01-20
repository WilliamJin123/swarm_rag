"""
Constrained Executor (Tier 3) - Intent-to-mutation translation.

The Constrained Executor is a DETERMINISTIC component (no LLM calls).
It translates mutation intents into validated parameter and expression changes.

Key responsibilities:
1. Apply parameter adjustments within bounds
2. Build safe expressions from templates
3. Ensure all changes are valid
4. Track what changes were made for the journal
"""
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

from ..types.genome import Genome, SwarmParams
from ..types.expressions import ExpressionNode
from .intents import (
    MutationIntent,
    MutationPrescription,
    TargetComponent,
    StrategicDirective,
    get_intent_action,
    IntentAction,
)
from .expression_builder import SafeExpressionBuilder

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """
    Result of executing a mutation prescription.
    """
    success: bool
    param_changes: Dict[str, Any] = field(default_factory=dict)
    strategy_changes: Dict[str, str] = field(default_factory=dict)
    ratio_changes: Dict[str, float] = field(default_factory=dict)
    templates_used: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def any_changes(self) -> bool:
        """Check if any changes were made."""
        return bool(self.param_changes or self.strategy_changes or self.ratio_changes)


@dataclass
class ParameterBounds:
    """
    Bounds for genome parameters.
    """
    n_agents: Tuple[int, int] = (5, 30)
    steps: Tuple[int, int] = (4, 12)
    decay: Tuple[float, float] = (0.85, 0.99)
    initial_pool_size: Tuple[int, int] = (10, 50)
    start_subset: Tuple[int, int] = (5, 15)
    drop_zone_inc: Tuple[float, float] = (0.05, 0.2)

    def get_bounds(self, param: str) -> Tuple[float, float]:
        """Get bounds for a parameter."""
        return getattr(self, param, (0, 100))

    def clamp(self, param: str, value: float) -> float:
        """Clamp value to parameter bounds."""
        min_val, max_val = self.get_bounds(param)
        return max(min_val, min(max_val, value))


class ConstrainedExecutor:
    """
    Tier 3: Translates intents into validated genome mutations.

    This is a deterministic component - no LLM calls.
    All changes are guaranteed to be valid.
    """

    def __init__(
        self,
        bounds: Optional[ParameterBounds] = None,
    ):
        """
        Initialize the executor.

        Args:
            bounds: Parameter bounds (uses defaults if not provided)
        """
        self.bounds = bounds or ParameterBounds()
        self.expression_builder = SafeExpressionBuilder()

    def execute(
        self,
        genome: Genome,
        prescription: MutationPrescription,
        directive: Optional[StrategicDirective] = None,
    ) -> ExecutionResult:
        """
        Execute a mutation prescription on a genome.

        Args:
            genome: Genome to mutate (will be modified in-place)
            prescription: Mutation prescription from Tactical Advisor
            directive: Strategic directive (for temperature)

        Returns:
            ExecutionResult with changes made
        """
        result = ExecutionResult(success=True)

        # Handle no_change intent
        if prescription.primary_intent == MutationIntent.NO_CHANGE:
            return result

        # Get temperature from directive
        temperature = 0.5
        if directive:
            temperature = directive.exploration_temperature

        # Scale confidence by temperature
        effective_confidence = prescription.confidence * (0.5 + temperature * 0.5)

        # Get action mapping for the intent
        action = get_intent_action(prescription.primary_intent)

        # Apply changes based on target component
        target = prescription.target_component

        if target in (TargetComponent.PARAMS, TargetComponent.ALL):
            self._apply_param_changes(genome, action, effective_confidence, result)

        if target in (TargetComponent.MOVEMENT, TargetComponent.ALL):
            self._apply_strategy_changes(
                genome, action, "movement", prescription.primary_intent,
                effective_confidence, result
            )

        if target in (TargetComponent.DEPOSIT, TargetComponent.ALL):
            self._apply_strategy_changes(
                genome, action, "deposit", prescription.primary_intent,
                effective_confidence, result
            )

        if target in (TargetComponent.RANKING, TargetComponent.ALL):
            self._apply_strategy_changes(
                genome, action, "ranking", prescription.primary_intent,
                effective_confidence, result
            )

        if target in (TargetComponent.RATIOS, TargetComponent.ALL):
            self._apply_ratio_changes(genome, action, effective_confidence, result)

        # Apply secondary intent if present
        if prescription.secondary_intent and prescription.secondary_intent != MutationIntent.NO_CHANGE:
            secondary_action = get_intent_action(prescription.secondary_intent)
            secondary_confidence = effective_confidence * 0.5  # Reduced influence

            if target in (TargetComponent.PARAMS, TargetComponent.ALL):
                self._apply_param_changes(genome, secondary_action, secondary_confidence, result)

        # Clear compiled cache since we modified strategies
        genome.clear_cache()

        return result

    def _apply_param_changes(
        self,
        genome: Genome,
        action: IntentAction,
        confidence: float,
        result: ExecutionResult,
    ):
        """
        Apply parameter adjustments.

        Args:
            genome: Genome to modify
            action: IntentAction with adjustments
            confidence: Confidence level (affects magnitude)
            result: ExecutionResult to update
        """
        for param, delta in action.param_adjustments.items():
            if param not in genome.params:
                continue

            current = genome.params[param]

            # Scale delta by confidence
            scaled_delta = delta * confidence

            # Add some noise for diversity
            noise = random.gauss(0, abs(delta) * 0.2)
            scaled_delta += noise

            # Apply change
            if isinstance(current, int):
                new_value = int(round(current + scaled_delta))
            else:
                new_value = current + scaled_delta

            # Clamp to bounds
            new_value = self.bounds.clamp(param, new_value)

            # Ensure type consistency
            if isinstance(current, int):
                new_value = int(new_value)

            if new_value != current:
                genome.params[param] = new_value
                result.param_changes[param] = {
                    "old": current,
                    "new": new_value,
                    "delta": new_value - current,
                }

    def _apply_strategy_changes(
        self,
        genome: Genome,
        action: IntentAction,
        category: str,
        intent: MutationIntent,
        confidence: float,
        result: ExecutionResult,
    ):
        """
        Apply strategy expression changes.

        Args:
            genome: Genome to modify
            action: IntentAction with preferences
            category: "movement", "deposit", or "ranking"
            intent: The mutation intent
            confidence: Confidence level
            result: ExecutionResult to update
        """
        # Find strategy keys matching the category
        strategy_keys = [
            k for k in genome.strategies.keys()
            if k.endswith(f"_{category}") or k == category
        ]

        if not strategy_keys and category == "ranking":
            # Ranking might just be "ranking"
            if "ranking" in genome.strategies:
                strategy_keys = ["ranking"]

        for key in strategy_keys:
            current_node = genome.strategies.get(key)

            # Build new expression from template
            try:
                new_node, template_name = self.expression_builder.build_from_intent(
                    intent=intent,
                    category=category,
                    confidence=confidence,
                    current_weights=self._extract_current_weights(current_node) if current_node else None,
                )

                # Only update if different
                current_str = current_node.to_string() if current_node else ""
                new_str = new_node.to_string()

                if new_str != current_str:
                    genome.strategies[key] = new_node
                    result.strategy_changes[key] = new_str
                    result.templates_used[key] = template_name

            except Exception as e:
                result.warnings.append(f"Failed to build {category} strategy: {e}")
                logger.warning(f"Strategy build failed for {key}: {e}")

    def _apply_ratio_changes(
        self,
        genome: Genome,
        action: IntentAction,
        confidence: float,
        result: ExecutionResult,
    ):
        """
        Apply agent group ratio changes.

        Args:
            genome: Genome to modify
            action: IntentAction (may specify ratio adjustments)
            confidence: Confidence level
            result: ExecutionResult to update
        """
        if not genome.group_ratios:
            return

        # For now, implement simple ratio balancing
        # More sophisticated ratio adjustments could be added based on action

        n_groups = len(genome.group_ratios)
        if n_groups < 2:
            return

        # Add small random perturbations
        for group_key in genome.group_ratios:
            current = genome.group_ratios[group_key]

            # Small random adjustment
            adjustment = random.gauss(0, 0.05 * confidence)
            new_ratio = max(0.1, min(0.9, current + adjustment))

            if abs(new_ratio - current) > 0.01:
                genome.group_ratios[group_key] = new_ratio
                result.ratio_changes[group_key] = new_ratio

        # Normalize ratios
        genome.normalize_ratios()

    def _extract_current_weights(self, node: ExpressionNode) -> Dict[str, float]:
        """
        Extract weight values from current expression.

        Args:
            node: Current expression node

        Returns:
            Dict of heuristic -> weight
        """
        return self.expression_builder._extract_weights(node)

    def apply_jitter(
        self,
        genome: Genome,
        magnitude: float = 0.1,
    ) -> ExecutionResult:
        """
        Apply small random jitter to parameters (for diversity).

        Args:
            genome: Genome to jitter
            magnitude: How much to perturb (0.0-1.0)

        Returns:
            ExecutionResult with changes
        """
        result = ExecutionResult(success=True)

        # Jitter numeric parameters
        param_jitter = {
            "n_agents": 2,
            "steps": 1,
            "decay": 0.02,
            "initial_pool_size": 3,
        }

        for param, base_delta in param_jitter.items():
            if param not in genome.params:
                continue

            current = genome.params[param]
            delta = random.gauss(0, base_delta * magnitude)

            if isinstance(current, int):
                new_value = int(round(current + delta))
            else:
                new_value = current + delta

            new_value = self.bounds.clamp(param, new_value)

            if isinstance(current, int):
                new_value = int(new_value)

            if new_value != current:
                genome.params[param] = new_value
                result.param_changes[param] = {
                    "old": current,
                    "new": new_value,
                }

        return result


class ThreeTierMutator:
    """
    Orchestrates the three-tier mutation process.

    Combines:
    - Strategic Oracle (Tier 1): High-level steering
    - Tactical Advisor (Tier 2): Per-genome diagnosis
    - Constrained Executor (Tier 3): Validated mutations
    """

    def __init__(
        self,
        llm_wrapper: Any,
        model: str,
        bounds: Optional[ParameterBounds] = None,
    ):
        """
        Initialize the three-tier mutator.

        Args:
            llm_wrapper: LLM wrapper for API calls
            model: Model ID to use
            bounds: Parameter bounds
        """
        from .strategic_oracle import StrategicOracle
        from .tactical_advisor import TacticalAdvisor

        self.oracle = StrategicOracle(llm_wrapper, model)
        self.advisor = TacticalAdvisor(llm_wrapper, model)
        self.executor = ConstrainedExecutor(bounds)

        self._current_directive: Optional[StrategicDirective] = None

    def update_directive(
        self,
        context: Any,  # StrategicContext
        journal: Optional[Any] = None,  # EvolutionJournal
    ) -> StrategicDirective:
        """
        Update the strategic directive (call periodically).

        Args:
            context: Strategic context
            journal: Evolution journal

        Returns:
            New strategic directive
        """
        self._current_directive = self.oracle.get_directive(context, journal)
        return self._current_directive

    def get_directive(self) -> StrategicDirective:
        """Get current directive without calling LLM."""
        if self._current_directive is None:
            return StrategicDirective.default()
        return self._current_directive

    def mutate(
        self,
        genome: Genome,
        journal: Optional[Any] = None,  # EvolutionJournal
    ) -> Tuple[ExecutionResult, MutationPrescription]:
        """
        Perform three-tier mutation on a genome.

        Args:
            genome: Genome to mutate (modified in-place)
            journal: Evolution journal for history

        Returns:
            Tuple of (ExecutionResult, MutationPrescription)
        """
        from .tactical_advisor import build_tactical_context

        directive = self.get_directive()

        # Build tactical context
        tactical_ctx = build_tactical_context(genome, directive, journal)

        # Get prescription from Tier 2
        prescription = self.advisor.get_prescription(tactical_ctx)

        # Execute prescription with Tier 3
        result = self.executor.execute(genome, prescription, directive)

        return result, prescription

    def should_update_oracle(
        self,
        generation: int,
        stagnation_count: int,
    ) -> bool:
        """
        Check if oracle should be updated.

        Args:
            generation: Current generation
            stagnation_count: Stagnant generations

        Returns:
            True if oracle should be called
        """
        return self.oracle.should_call(generation, stagnation_count)
