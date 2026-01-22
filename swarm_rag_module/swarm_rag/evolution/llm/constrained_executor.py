"""
Constrained Executor (Tier 3) - Intent-to-mutation translation.

The Constrained Executor is a DETERMINISTIC component (no LLM calls).
It translates mutation intents into validated parameter and expression changes.

Key responsibilities:
1. Apply parameter adjustments within bounds
2. Build safe expressions from templates
3. Ensure all changes are valid
4. Track what changes were made for the journal

Creative Mode Extension (Tier 2.5):
When creative mode is enabled and triggered, the executor can use
LLM-generated custom expressions validated through AST parsing.
"""
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING

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

if TYPE_CHECKING:
    from .client import LLMClient
    from .creative_synthesizer import CreativeSynthesizer, CreativeProposal

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

    # Creative mode tracking
    used_creative_mode: bool = False
    creative_categories: List[str] = field(default_factory=list)  # Which categories used creative
    creative_rationale: str = ""
    creative_expressions: Dict[str, str] = field(default_factory=dict)  # category -> expression

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

    Creative Mode Extension:
    When a CreativeSynthesizer is provided and creative mode is triggered,
    the executor can use LLM-generated expressions instead of templates.
    """

    def __init__(
        self,
        bounds: Optional[ParameterBounds] = None,
        creative_synthesizer: Optional["CreativeSynthesizer"] = None,
    ):
        """
        Initialize the executor.

        Args:
            bounds: Parameter bounds (uses defaults if not provided)
            creative_synthesizer: Optional creative synthesizer for custom expressions
        """
        self.bounds = bounds or ParameterBounds()
        self.expression_builder = SafeExpressionBuilder()
        self.creative_synthesizer = creative_synthesizer

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

    def execute_with_creative(
        self,
        genome: Genome,
        prescription: MutationPrescription,
        directive: Optional[StrategicDirective] = None,
        creative_context: Optional[Any] = None,  # CreativeModeContext
        use_creative: bool = False,
    ) -> ExecutionResult:
        """
        Execute a mutation prescription with optional creative mode.

        Args:
            genome: Genome to mutate (will be modified in-place)
            prescription: Mutation prescription from Tactical Advisor
            directive: Strategic directive (for temperature)
            creative_context: Context for creative synthesis
            use_creative: Whether to attempt creative mode

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
        target = prescription.target_component

        # Try creative mode if enabled and context provided
        creative_success = False
        if use_creative and self.creative_synthesizer and creative_context:
            creative_success = self._try_creative_mode(
                genome, prescription, creative_context, result
            )

        # Apply standard changes if creative mode wasn't used or failed
        if not creative_success:
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

        # Apply secondary intent if present (always use standard path)
        if prescription.secondary_intent and prescription.secondary_intent != MutationIntent.NO_CHANGE:
            secondary_action = get_intent_action(prescription.secondary_intent)
            secondary_confidence = effective_confidence * 0.5

            if target in (TargetComponent.PARAMS, TargetComponent.ALL):
                self._apply_param_changes(genome, secondary_action, secondary_confidence, result)

        # Clear compiled cache since we modified strategies
        genome.clear_cache()

        return result

    def _try_creative_mode(
        self,
        genome: Genome,
        prescription: MutationPrescription,
        creative_context: Any,
        result: ExecutionResult,
    ) -> bool:
        """
        Attempt to use creative mode for expression generation.

        Args:
            genome: Genome to mutate
            prescription: Mutation prescription
            creative_context: Context for creative synthesis
            result: ExecutionResult to update

        Returns:
            True if creative mode was successfully applied
        """
        if not self.creative_synthesizer:
            return False

        # Determine which categories to target
        target = prescription.target_component
        categories = []

        if target in (TargetComponent.MOVEMENT, TargetComponent.ALL):
            categories.append("movement")
        if target in (TargetComponent.DEPOSIT, TargetComponent.ALL):
            categories.append("deposit")
        if target in (TargetComponent.RANKING, TargetComponent.ALL):
            categories.append("ranking")

        if not categories:
            return False

        # Try to synthesize expressions
        try:
            proposals = self.creative_synthesizer.synthesize_multiple(
                creative_context, categories
            )
        except Exception as e:
            logger.warning(f"Creative synthesis failed: {e}")
            result.warnings.append(f"Creative synthesis failed: {e}")
            return False

        # Apply valid proposals
        any_applied = False
        rationales = []

        for category, proposal in proposals.items():
            if proposal.valid and proposal.parsed_node:
                # Apply the creative expression
                applied = self._apply_creative_expression(
                    genome, category, proposal, result
                )
                if applied:
                    any_applied = True
                    result.creative_categories.append(category)
                    result.creative_expressions[category] = proposal.expression
                    if proposal.rationale:
                        rationales.append(f"{category}: {proposal.rationale}")
            else:
                # Log validation failure
                error = proposal.validation_error or "Unknown error"
                logger.debug(f"Creative proposal for {category} invalid: {error}")
                result.warnings.append(f"Creative {category} validation failed: {error}")

        if any_applied:
            result.used_creative_mode = True
            result.creative_rationale = " | ".join(rationales)

        return any_applied

    def _apply_creative_expression(
        self,
        genome: Genome,
        category: str,
        proposal: "CreativeProposal",
        result: ExecutionResult,
    ) -> bool:
        """
        Apply a validated creative expression to the genome.

        Args:
            genome: Genome to mutate
            category: Expression category (movement/deposit/ranking)
            proposal: Validated creative proposal
            result: ExecutionResult to update

        Returns:
            True if expression was applied
        """
        if not proposal.parsed_node:
            return False

        # Find strategy keys matching the category
        strategy_keys = [
            k for k in genome.strategies.keys()
            if k.endswith(f"_{category}") or k == category
        ]

        if not strategy_keys and category == "ranking":
            if "ranking" in genome.strategies:
                strategy_keys = ["ranking"]

        if not strategy_keys:
            # No existing strategies for this category
            result.warnings.append(f"No {category} strategies found in genome")
            return False

        # Apply to all matching strategy keys
        applied_any = False
        for key in strategy_keys:
            current_node = genome.strategies.get(key)
            current_str = current_node.to_string() if current_node else ""
            new_str = proposal.parsed_node.to_string()

            if new_str != current_str:
                genome.strategies[key] = proposal.parsed_node.copy()
                result.strategy_changes[key] = new_str
                result.templates_used[key] = f"creative:{category}"
                applied_any = True

        return applied_any


class ThreeTierMutator:
    """
    Orchestrates the three-tier mutation process.

    Combines:
    - Strategic Oracle (Tier 1): High-level steering
    - Tactical Advisor (Tier 2): Per-genome diagnosis
    - Creative Synthesizer (Tier 2.5, optional): Custom LLM expressions
    - Constrained Executor (Tier 3): Validated mutations

    All LLM calls flow through LLMClient.
    """

    def __init__(
        self,
        client: "LLMClient",
        bounds: Optional[ParameterBounds] = None,
        creative_mode_config: Optional[Any] = None,  # CreativeModeConfig
    ):
        """
        Initialize the three-tier mutator.

        Args:
            client: LLMClient for all LLM API calls
            bounds: Parameter bounds
            creative_mode_config: Optional creative mode configuration
        """
        from .strategic_oracle import StrategicOracle
        from .tactical_advisor import TacticalAdvisor

        self.client = client
        self.oracle = StrategicOracle(client)
        self.advisor = TacticalAdvisor(client)

        # Initialize creative synthesizer if creative mode is enabled
        self.creative_synthesizer: Optional["CreativeSynthesizer"] = None
        self.creative_mode_config = creative_mode_config

        if creative_mode_config and creative_mode_config.enabled:
            from .creative_synthesizer import CreativeSynthesizer
            self.creative_synthesizer = CreativeSynthesizer(
                client=client,
                complexity_limit=creative_mode_config.complexity_limit,
            )
            logger.info("Creative mode enabled in ThreeTierMutator")

        self.executor = ConstrainedExecutor(
            bounds=bounds,
            creative_synthesizer=self.creative_synthesizer,
        )

        self._current_directive: Optional[StrategicDirective] = None

        # Creative mode tracking
        self._creative_mutations_this_gen = 0
        self._stagnation_count = 0
        self._archive_fill_rate = 0.0
        self._top_fitness_unchanged = 0

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

        # Check if creative mode should be used
        use_creative = self._should_use_creative()
        creative_context = None

        if use_creative and self.creative_synthesizer:
            from .creative_synthesizer import build_creative_context
            creative_context = build_creative_context(
                genome=genome,
                prescription=prescription,
                stagnation_count=self._stagnation_count,
                archive_fill_rate=self._archive_fill_rate,
                top_fitness_unchanged=self._top_fitness_unchanged,
                generation=0,  # Will be set by caller if needed
            )

            # Use creative execution path
            result = self.executor.execute_with_creative(
                genome, prescription, directive, creative_context, use_creative=True
            )

            # Track creative mutation usage
            if result.used_creative_mode:
                self._creative_mutations_this_gen += 1
        else:
            # Execute prescription with standard Tier 3
            result = self.executor.execute(genome, prescription, directive)

        return result, prescription

    def mutate_with_context(
        self,
        genome: Genome,
        journal: Optional[Any] = None,
        stagnation_count: int = 0,
        archive_fill_rate: float = 0.5,
        top_fitness_unchanged: int = 0,
        generation: int = 0,
    ) -> Tuple[ExecutionResult, MutationPrescription]:
        """
        Perform mutation with explicit context for creative mode triggering.

        Args:
            genome: Genome to mutate (modified in-place)
            journal: Evolution journal for history
            stagnation_count: Generations without improvement
            archive_fill_rate: Current archive fill rate
            top_fitness_unchanged: Generations where top fitness unchanged
            generation: Current generation number

        Returns:
            Tuple of (ExecutionResult, MutationPrescription)
        """
        # Update internal tracking
        self._stagnation_count = stagnation_count
        self._archive_fill_rate = archive_fill_rate
        self._top_fitness_unchanged = top_fitness_unchanged

        from .tactical_advisor import build_tactical_context

        directive = self.get_directive()
        tactical_ctx = build_tactical_context(genome, directive, journal)
        prescription = self.advisor.get_prescription(tactical_ctx)

        # Check if creative mode should be used
        use_creative = self._should_use_creative()
        creative_context = None

        if use_creative and self.creative_synthesizer:
            from .creative_synthesizer import build_creative_context
            creative_context = build_creative_context(
                genome=genome,
                prescription=prescription,
                stagnation_count=stagnation_count,
                archive_fill_rate=archive_fill_rate,
                top_fitness_unchanged=top_fitness_unchanged,
                generation=generation,
            )

            result = self.executor.execute_with_creative(
                genome, prescription, directive, creative_context, use_creative=True
            )

            if result.used_creative_mode:
                self._creative_mutations_this_gen += 1
        else:
            result = self.executor.execute(genome, prescription, directive)

        return result, prescription

    def _should_use_creative(self) -> bool:
        """
        Determine if creative mode should be used for this mutation.

        Returns:
            True if creative mode should be triggered
        """
        if not self.creative_mode_config or not self.creative_mode_config.enabled:
            return False

        if not self.creative_synthesizer:
            return False

        # Check per-generation limit
        max_per_gen = self.creative_mode_config.max_creative_per_generation
        if self._creative_mutations_this_gen >= max_per_gen:
            return False

        # Check trigger conditions
        from .creative_synthesizer import should_use_creative_mode
        return should_use_creative_mode(
            creative_mode_enabled=True,
            stagnation_count=self._stagnation_count,
            archive_fill_rate=self._archive_fill_rate,
            top_fitness_unchanged=self._top_fitness_unchanged,
            generation=0,  # Periodic interval handled separately
            stagnation_threshold=self.creative_mode_config.trigger_stagnation,
            fill_rate_threshold=self.creative_mode_config.trigger_fill_rate,
            periodic_interval=self.creative_mode_config.periodic_interval,
        )

    def update_archive_state(
        self,
        stagnation_count: int,
        archive_fill_rate: float,
        top_fitness_unchanged: int,
    ):
        """
        Update archive state for creative mode triggering.

        Args:
            stagnation_count: Generations without improvement
            archive_fill_rate: Current archive fill rate
            top_fitness_unchanged: Generations where top fitness unchanged
        """
        self._stagnation_count = stagnation_count
        self._archive_fill_rate = archive_fill_rate
        self._top_fitness_unchanged = top_fitness_unchanged

    def reset_generation_counters(self):
        """Reset per-generation counters (call at start of each generation)."""
        self._creative_mutations_this_gen = 0

    def get_creative_stats(self) -> Dict[str, Any]:
        """Get creative mode statistics."""
        stats = {
            "creative_mutations_this_gen": self._creative_mutations_this_gen,
            "creative_mode_enabled": self.creative_mode_config.enabled if self.creative_mode_config else False,
        }

        if self.creative_synthesizer:
            stats.update(self.creative_synthesizer.get_stats())

        return stats

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
