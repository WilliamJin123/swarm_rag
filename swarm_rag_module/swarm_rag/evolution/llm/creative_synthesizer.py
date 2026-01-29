"""
Creative Synthesizer (Tier 2.5) - LLM-generated custom expressions.

The Creative Synthesizer is an OPTIONAL component that allows the LLM to
generate custom heuristic expressions beyond the predefined templates.
It sits between the Tactical Advisor (Tier 2) and Constrained Executor (Tier 3).

Key responsibilities:
1. Generate custom expressions when archive is stagnant
2. Validate all generated expressions through AST parsing
3. Enforce feature/operator whitelists
4. Fall back to templates on validation failure

Trigger conditions:
- Stagnation detected (5+ generations without improvement)
- Archive fill rate below threshold (<30%)
- Top fitness unchanged for 3+ generations
- Periodic experimentation (every 10 generations)

LLM calls: synthesize() -> LLMClient.call()
"""
import logging
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, TYPE_CHECKING

from ..types.expressions import ExpressionNode
from .parsers import ExpressionParser
from .intents import MutationPrescription, TargetComponent

if TYPE_CHECKING:
    from .client import LLMClient

logger = logging.getLogger(__name__)


# =============================================================================
# Creative Proposal Data Types
# =============================================================================

@dataclass
class CreativeProposal:
    """
    A creative expression proposal from the LLM.

    Contains the raw expression string, validation status, and parsed result.
    """
    expression: str                         # Raw expression string from LLM
    category: str                           # movement/deposit/ranking
    rationale: str                          # LLM explanation for this expression
    confidence: float                       # LLM confidence (0.0-1.0)

    # Validation results (populated after validation)
    parsed_node: Optional[ExpressionNode] = None
    valid: bool = False
    validation_error: Optional[str] = None

    # Metadata
    features_used: List[str] = field(default_factory=list)
    complexity: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "expression": self.expression,
            "category": self.category,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "valid": self.valid,
            "validation_error": self.validation_error,
            "features_used": self.features_used,
            "complexity": self.complexity,
        }


@dataclass
class CreativeModeContext:
    """
    Context passed to the creative synthesizer for expression generation.
    """
    genome_id: str
    quality_score: float
    stability_score: float
    recall: float

    # Performance issues identified
    issues: List[str] = field(default_factory=list)

    # Tactical diagnosis from Tier 2
    diagnosis: str = ""
    intent: str = ""

    # Archive state
    stagnation_count: int = 0
    archive_fill_rate: float = 0.0
    top_fitness_unchanged: int = 0
    generation: int = 0

    # Current expression (for reference)
    current_expression: Optional[str] = None


# =============================================================================
# Heuristic Whitelists
# =============================================================================

AVAILABLE_HEURISTICS: Dict[str, List[str]] = {
    "movement": [
        "semantic_similarity",
        "pheromone_repulsion",
        "node_centrality",
        "random_jitter",
    ],
    "deposit": [
        "semantic_similarity",
        "node_centrality",
        "pheromone_repulsion",
    ],
    "ranking": [
        "semantic_similarity",
        "percentage_visited",
    ],
}

# Allowed operators (AST operators)
ALLOWED_OPERATORS = ["+", "-", "*", "/", "max", "min"]

# Allowed functions
ALLOWED_FUNCTIONS = ["sqrt", "abs", "tanh", "sigmoid", "exp", "log"]

# Complexity limit
MAX_EXPRESSION_COMPLEXITY = 30


# =============================================================================
# Creative Synthesizer
# =============================================================================

class CreativeSynthesizer:
    """
    Tier 2.5: Generates and validates custom LLM expressions.

    The Creative Synthesizer allows the LLM to generate expressions that
    go beyond predefined templates while maintaining safety through:

    1. AST parsing validation
    2. Feature whitelist enforcement
    3. Operator whitelist enforcement
    4. Complexity limits
    5. Test evaluation with dummy features

    If validation fails, the system falls back to template-based generation.

    LLM calls made via: synthesize() -> self.client.call()
    """

    def __init__(
        self,
        client: "LLMClient",
        complexity_limit: int = MAX_EXPRESSION_COMPLEXITY,
    ):
        """
        Initialize the creative synthesizer.

        Args:
            client: LLMClient instance for API calls
            complexity_limit: Maximum expression node count
        """
        self.client = client
        self.complexity_limit = complexity_limit

        # Track success/failure for circuit breaker
        self._success_count = 0
        self._failure_count = 0
        self._consecutive_failures = 0
        self._disabled = False

    def synthesize(
        self,
        context: CreativeModeContext,
        category: str,
    ) -> CreativeProposal:
        """
        Synthesize a creative expression for the given category.

        Args:
            context: Creative mode context with genome info
            category: movement/deposit/ranking

        Returns:
            CreativeProposal with validation status
        """
        if self._disabled:
            return self._create_failed_proposal(
                category,
                "Creative mode temporarily disabled due to consecutive failures"
            )

        # Build prompt
        user_prompt = self._build_creative_prompt(context, category)
        system_prompt = (
            "You are an expert in designing heuristic expressions for swarm "
            "intelligence systems. Generate valid, safe expressions using only "
            "the allowed components."
        )

        # Call LLM
        result = self.client.call(
            system_prompt, user_prompt, json_mode=True, temperature=0.7
        )
        if not result.success:
            logger.warning(f"Creative synthesis LLM call failed: {result.error}")
            return self._create_failed_proposal(category, f"LLM call failed: {result.error}")

        proposal = self._parse_llm_response(result.content, category)

        # Validate the proposal
        proposal = self._validate_proposal(proposal)

        # Update circuit breaker stats
        self._update_stats(proposal.valid)

        return proposal

    def synthesize_multiple(
        self,
        context: CreativeModeContext,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, CreativeProposal]:
        """
        Synthesize expressions for multiple categories in one call.

        Args:
            context: Creative mode context
            categories: List of categories (defaults to all)

        Returns:
            Dict of category -> CreativeProposal
        """
        categories = categories or ["movement", "deposit", "ranking"]

        if self._disabled:
            return {
                cat: self._create_failed_proposal(cat, "Creative mode disabled")
                for cat in categories
            }

        # Build multi-category prompt
        user_prompt = self._build_multi_category_prompt(context, categories)
        system_prompt = (
            "You are an expert in designing heuristic expressions for swarm "
            "intelligence systems. Generate valid, safe expressions using only "
            "the allowed components."
        )

        result = self.client.call(
            system_prompt, user_prompt, json_mode=True, temperature=0.7
        )
        if not result.success:
            logger.warning(f"Multi-category synthesis failed: {result.error}")
            return {
                cat: self._create_failed_proposal(cat, f"LLM call failed: {result.error}")
                for cat in categories
            }

        proposals = self._parse_multi_response(result.content, categories)

        # Validate each proposal
        validated = {}
        for cat, proposal in proposals.items():
            validated[cat] = self._validate_proposal(proposal)
            self._update_stats(validated[cat].valid)

        return validated

    def _build_creative_prompt(
        self,
        context: CreativeModeContext,
        category: str,
    ) -> str:
        """Build the creative synthesis prompt."""
        heuristics = AVAILABLE_HEURISTICS.get(category, [])
        heuristic_list = "\n".join(f"  - {h}" for h in heuristics)

        return f"""You are designing a {category} heuristic for a swarm intelligence retrieval system.

## Current Genome Performance
- Quality: {context.quality_score:.3f}
- Stability: {context.stability_score:.3f}
- Recall: {context.recall:.3f}
- Issues: {', '.join(context.issues) if context.issues else 'None identified'}

## Tactical Diagnosis
{context.diagnosis}
Intent: {context.intent}

## Stagnation Context
- Generations without improvement: {context.stagnation_count}
- Archive fill rate: {context.archive_fill_rate:.1%}
- Current generation: {context.generation}

## Available Building Blocks

Heuristics (use exactly these names):
{heuristic_list}

Operators: +, -, *, /
Functions: sqrt(x), abs(x), tanh(x), sigmoid(x), exp(x), log(x), max(a,b), min(a,b)
Constants: any float between 0.0 and 2.0

## Constraints
- Expression must evaluate to a single numeric value
- All heuristics return values in [0, 1] range
- Keep expressions simple (under {self.complexity_limit} nodes when parsed)
- Weights should generally sum to ~1.0 for weighted combinations
- Use ONLY the heuristics listed above for {category}

## Examples
Good: "semantic_similarity * 0.6 + pheromone_repulsion * 0.3 + random_jitter * 0.1"
Good: "sqrt(semantic_similarity) * 0.7 + tanh(pheromone_repulsion * 2) * 0.3"
Good: "max(semantic_similarity, node_centrality * 0.5) + random_jitter * 0.1"
Bad: "unknown_feature * 0.5" (unknown heuristic)
Bad: "semantic_similarity ** 2" (power operator not allowed)

## Output Format (JSON)
{{
  "expression": "<your expression using ONLY the allowed heuristics>",
  "rationale": "<brief explanation of why this expression might work better>",
  "confidence": <0.0-1.0>
}}

Generate a creative {category} expression that addresses the diagnosed issues."""

    def _build_multi_category_prompt(
        self,
        context: CreativeModeContext,
        categories: List[str],
    ) -> str:
        """Build prompt for multi-category synthesis."""
        category_blocks = []
        for cat in categories:
            heuristics = AVAILABLE_HEURISTICS.get(cat, [])
            category_blocks.append(f"  {cat}: {', '.join(heuristics)}")

        return f"""You are designing heuristics for a swarm intelligence retrieval system.

## Current Genome Performance
- Quality: {context.quality_score:.3f}
- Stability: {context.stability_score:.3f}
- Recall: {context.recall:.3f}
- Issues: {', '.join(context.issues) if context.issues else 'None identified'}

## Tactical Diagnosis
{context.diagnosis}
Intent: {context.intent}

## Available Heuristics by Category
{chr(10).join(category_blocks)}

## Operators & Functions
Operators: +, -, *, /
Functions: sqrt(x), abs(x), tanh(x), sigmoid(x), exp(x), log(x), max(a,b), min(a,b)
Constants: floats between 0.0 and 2.0

## Output Format (JSON)
{{
  "movement": {{
    "expression": "<expression>",
    "rationale": "<why>",
    "confidence": <0.0-1.0>
  }},
  "deposit": {{
    "expression": "<expression>",
    "rationale": "<why>",
    "confidence": <0.0-1.0>
  }},
  "ranking": {{
    "expression": "<expression>",
    "rationale": "<why>",
    "confidence": <0.0-1.0>
  }}
}}

Generate creative expressions for each category. Use ONLY the heuristics listed for each category."""

    def _parse_llm_response(
        self,
        response: str,
        category: str,
    ) -> CreativeProposal:
        """Parse single-category LLM response."""
        try:
            data = json.loads(response)
            return CreativeProposal(
                expression=data.get("expression", ""),
                category=category,
                rationale=data.get("rationale", ""),
                confidence=float(data.get("confidence", 0.5)),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._create_failed_proposal(category, f"Parse error: {e}")

    def _parse_multi_response(
        self,
        response: str,
        categories: List[str],
    ) -> Dict[str, CreativeProposal]:
        """Parse multi-category LLM response."""
        proposals = {}

        try:
            data = json.loads(response)

            for cat in categories:
                if cat in data and isinstance(data[cat], dict):
                    cat_data = data[cat]
                    proposals[cat] = CreativeProposal(
                        expression=cat_data.get("expression", ""),
                        category=cat,
                        rationale=cat_data.get("rationale", ""),
                        confidence=float(cat_data.get("confidence", 0.5)),
                    )
                else:
                    proposals[cat] = self._create_failed_proposal(
                        cat, f"Missing {cat} in response"
                    )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse multi-response: {e}")
            for cat in categories:
                proposals[cat] = self._create_failed_proposal(cat, f"Parse error: {e}")

        return proposals

    def _validate_proposal(self, proposal: CreativeProposal) -> CreativeProposal:
        """
        Validate a creative proposal through multiple checks.

        Validation pipeline:
        1. AST parse (syntax check)
        2. Feature whitelist check
        3. Operator whitelist check
        4. Complexity limit
        5. Test evaluation with dummy features
        """
        if not proposal.expression:
            proposal.valid = False
            proposal.validation_error = "Empty expression"
            return proposal

        try:
            # 1. AST parse
            node = ExpressionParser.parse(proposal.expression)

            # 2. Feature whitelist check
            features = self._extract_features(node)
            allowed = set(AVAILABLE_HEURISTICS.get(proposal.category, []))
            invalid = [f for f in features if f not in allowed]

            if invalid:
                raise ValueError(f"Invalid features for {proposal.category}: {invalid}")

            # 3. Operator/function whitelist check
            self._check_operators(node)

            # 4. Complexity limit
            complexity = node.size()
            if complexity > self.complexity_limit:
                raise ValueError(
                    f"Expression too complex: {complexity} nodes > {self.complexity_limit}"
                )

            # 5. Test evaluation with dummy features
            dummy_features = {h: 0.5 for h in allowed}
            result = node.evaluate(dummy_features)

            result_tensor = torch.as_tensor(result) if not isinstance(result, torch.Tensor) else result
            if not torch.isfinite(result_tensor).all():
                raise ValueError("Expression produces non-finite values")

            # Validation passed
            proposal.parsed_node = node
            proposal.valid = True
            proposal.features_used = list(features)
            proposal.complexity = complexity

        except Exception as e:
            proposal.valid = False
            proposal.validation_error = str(e)
            logger.debug(f"Validation failed for '{proposal.expression}': {e}")

        return proposal

    def _extract_features(self, node: ExpressionNode) -> Set[str]:
        """Recursively extract feature names from expression."""
        features = set()

        if node.type == 'feature':
            features.add(str(node.value))

        for child in node.children:
            features.update(self._extract_features(child))

        return features

    def _check_operators(self, node: ExpressionNode):
        """Check that only allowed operators/functions are used."""
        if node.type == 'op':
            if node.value not in ALLOWED_OPERATORS:
                raise ValueError(f"Disallowed operator: {node.value}")
        elif node.type == 'func':
            if node.value not in ALLOWED_FUNCTIONS:
                raise ValueError(f"Disallowed function: {node.value}")

        for child in node.children:
            self._check_operators(child)

    def _create_failed_proposal(
        self,
        category: str,
        error: str,
    ) -> CreativeProposal:
        """Create a failed proposal with error message."""
        return CreativeProposal(
            expression="",
            category=category,
            rationale="",
            confidence=0.0,
            valid=False,
            validation_error=error,
        )

    def _update_stats(self, success: bool):
        """Update success/failure stats for circuit breaker."""
        if success:
            self._success_count += 1
            self._consecutive_failures = 0
        else:
            self._failure_count += 1
            self._consecutive_failures += 1

            # Circuit breaker: disable after 5 consecutive failures
            if self._consecutive_failures >= 5:
                logger.warning(
                    "Creative mode disabled due to 5 consecutive failures. "
                    "Call reset_circuit_breaker() to re-enable."
                )
                self._disabled = True

    def reset_circuit_breaker(self):
        """Reset the circuit breaker to re-enable creative mode."""
        self._disabled = False
        self._consecutive_failures = 0
        logger.info("Creative mode circuit breaker reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get synthesis statistics."""
        total = self._success_count + self._failure_count
        return {
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "success_rate": self._success_count / total if total > 0 else 0.0,
            "consecutive_failures": self._consecutive_failures,
            "disabled": self._disabled,
        }


# =============================================================================
# Trigger Logic
# =============================================================================

def should_use_creative_mode(
    creative_mode_enabled: bool,
    stagnation_count: int,
    archive_fill_rate: float,
    top_fitness_unchanged: int,
    generation: int,
    stagnation_threshold: int = 5,
    fill_rate_threshold: float = 0.3,
    periodic_interval: int = 10,
) -> bool:
    """
    Determine if creative mode should be triggered.

    Args:
        creative_mode_enabled: Global enable flag
        stagnation_count: Generations without improvement
        archive_fill_rate: Fraction of archive cells filled
        top_fitness_unchanged: Generations where top fitness hasn't improved
        generation: Current generation number
        stagnation_threshold: Stagnation trigger threshold
        fill_rate_threshold: Archive fill rate trigger threshold
        periodic_interval: Generation interval for periodic experimentation

    Returns:
        True if creative mode should be used
    """
    if not creative_mode_enabled:
        return False

    return (
        stagnation_count >= stagnation_threshold or
        archive_fill_rate < fill_rate_threshold or
        top_fitness_unchanged >= 3 or
        (generation > 0 and generation % periodic_interval == 0)
    )


def build_creative_context(
    genome: Any,  # Genome
    prescription: MutationPrescription,
    stagnation_count: int = 0,
    archive_fill_rate: float = 0.5,
    top_fitness_unchanged: int = 0,
    generation: int = 0,
) -> CreativeModeContext:
    """
    Build context for creative synthesis from genome and prescription.

    Args:
        genome: The genome to enhance
        prescription: Mutation prescription from Tactical Advisor
        stagnation_count: Generations without improvement
        archive_fill_rate: Archive fill rate
        top_fitness_unchanged: Generations where top fitness unchanged
        generation: Current generation

    Returns:
        CreativeModeContext for the synthesizer
    """
    # Extract performance metrics
    quality = genome.fitness.quality_score if genome.fitness else 0.0
    stability = genome.fitness.stability_score if genome.fitness else 0.0
    recall = genome.metrics.get("recall_at_20", 0.0)

    # Build issue list from diagnosis
    issues = []
    diagnosis_lower = prescription.diagnosis.lower()

    if "loop" in diagnosis_lower or "revisit" in diagnosis_lower:
        issues.append("high_revisit_rate")
    if "dead" in diagnosis_lower or "stuck" in diagnosis_lower:
        issues.append("dead_end_issues")
    if "cost" in diagnosis_lower:
        issues.append("high_cost")
    if "quality" in diagnosis_lower or "recall" in diagnosis_lower:
        issues.append("low_quality")
    if "convergence" in diagnosis_lower:
        issues.append("convergence_issues")
    if "dispersion" in diagnosis_lower or "coverage" in diagnosis_lower:
        issues.append("poor_coverage")

    return CreativeModeContext(
        genome_id=genome.id,
        quality_score=quality,
        stability_score=stability,
        recall=recall,
        issues=issues,
        diagnosis=prescription.diagnosis,
        intent=prescription.primary_intent.value,
        stagnation_count=stagnation_count,
        archive_fill_rate=archive_fill_rate,
        top_fitness_unchanged=top_fitness_unchanged,
        generation=generation,
    )
