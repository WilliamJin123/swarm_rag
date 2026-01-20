"""
Safe Expression Builder - Template-based expression generation.

Instead of having the LLM generate raw expression strings (error-prone),
this module provides validated templates that are guaranteed to parse correctly.
The LLM only needs to select a template and provide weight preferences.
"""
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from ..types.expressions import ExpressionNode
from .parsers import ExpressionParser
from .intents import MutationIntent, get_intent_action

logger = logging.getLogger(__name__)


@dataclass
class ExpressionTemplate:
    """
    A validated expression template with fillable weight slots.
    """
    name: str
    category: str  # 'movement', 'deposit', 'ranking'
    template: str  # Template string with {w1}, {w2}, etc. placeholders
    weight_slots: List[str]  # Names of heuristics for each weight slot
    default_weights: Dict[str, float]  # Default values if not specified
    description: str = ""

    def fill(self, weights: Optional[Dict[str, float]] = None) -> str:
        """
        Fill the template with weights.

        Args:
            weights: Dict of heuristic_name -> weight value

        Returns:
            Filled expression string
        """
        weights = weights or {}
        filled = self.template

        for i, slot_name in enumerate(self.weight_slots):
            placeholder = f"{{w{i + 1}}}"
            weight = weights.get(slot_name, self.default_weights.get(slot_name, 0.5))
            # Round to 2 decimal places
            weight = round(max(0.0, min(1.0, weight)), 2)
            filled = filled.replace(placeholder, str(weight))

        return filled

    def parse(self, weights: Optional[Dict[str, float]] = None) -> ExpressionNode:
        """
        Fill and parse the template into an ExpressionNode.

        Args:
            weights: Dict of heuristic_name -> weight value

        Returns:
            Parsed ExpressionNode (guaranteed valid)

        Raises:
            ValueError: If template is malformed (should never happen with validated templates)
        """
        filled = self.fill(weights)
        return ExpressionParser.parse(filled)


# ============================================================================
# Movement Strategy Templates
# ============================================================================

MOVEMENT_TEMPLATES: Dict[str, ExpressionTemplate] = {
    # Basic templates
    "semantic_only": ExpressionTemplate(
        name="semantic_only",
        category="movement",
        template="semantic_similarity",
        weight_slots=[],
        default_weights={},
        description="Pure semantic similarity - maximum exploitation",
    ),

    "semantic_focused": ExpressionTemplate(
        name="semantic_focused",
        category="movement",
        template="semantic_similarity * {w1} + pheromone_repulsion * {w2}",
        weight_slots=["semantic_similarity", "pheromone_repulsion"],
        default_weights={"semantic_similarity": 0.75, "pheromone_repulsion": 0.15},
        description="Semantic-heavy with light exploration",
    ),

    "balanced": ExpressionTemplate(
        name="balanced",
        category="movement",
        template="semantic_similarity * {w1} + pheromone_repulsion * {w2} + node_centrality * {w3}",
        weight_slots=["semantic_similarity", "pheromone_repulsion", "node_centrality"],
        default_weights={"semantic_similarity": 0.5, "pheromone_repulsion": 0.25, "node_centrality": 0.15},
        description="Balanced mix of exploitation and exploration",
    ),

    "balanced_exploration": ExpressionTemplate(
        name="balanced_exploration",
        category="movement",
        template="semantic_similarity * {w1} + pheromone_repulsion * {w2} + random_jitter * {w3}",
        weight_slots=["semantic_similarity", "pheromone_repulsion", "random_jitter"],
        default_weights={"semantic_similarity": 0.45, "pheromone_repulsion": 0.35, "random_jitter": 0.1},
        description="Balanced with randomness for diversity",
    ),

    "exploration_heavy": ExpressionTemplate(
        name="exploration_heavy",
        category="movement",
        template="semantic_similarity * {w1} + pheromone_repulsion * {w2} + random_jitter * {w3}",
        weight_slots=["semantic_similarity", "pheromone_repulsion", "random_jitter"],
        default_weights={"semantic_similarity": 0.35, "pheromone_repulsion": 0.4, "random_jitter": 0.15},
        description="Exploration-focused with strong repulsion",
    ),

    "anti_loop": ExpressionTemplate(
        name="anti_loop",
        category="movement",
        template="semantic_similarity * {w1} + pheromone_repulsion * {w2} + random_jitter * {w3}",
        weight_slots=["semantic_similarity", "pheromone_repulsion", "random_jitter"],
        default_weights={"semantic_similarity": 0.4, "pheromone_repulsion": 0.45, "random_jitter": 0.1},
        description="Strong loop avoidance",
    ),

    "hub_preferring": ExpressionTemplate(
        name="hub_preferring",
        category="movement",
        template="semantic_similarity * {w1} + node_centrality * {w2} + pheromone_repulsion * {w3}",
        weight_slots=["semantic_similarity", "node_centrality", "pheromone_repulsion"],
        default_weights={"semantic_similarity": 0.45, "node_centrality": 0.35, "pheromone_repulsion": 0.15},
        description="Prefers well-connected hub nodes",
    ),

    "connectivity_focused": ExpressionTemplate(
        name="connectivity_focused",
        category="movement",
        template="node_centrality * {w1} + semantic_similarity * {w2} + pheromone_repulsion * {w3}",
        weight_slots=["node_centrality", "semantic_similarity", "pheromone_repulsion"],
        default_weights={"node_centrality": 0.4, "semantic_similarity": 0.4, "pheromone_repulsion": 0.15},
        description="Prioritizes connectivity to avoid dead-ends",
    ),

    "dispersion_focused": ExpressionTemplate(
        name="dispersion_focused",
        category="movement",
        template="pheromone_repulsion * {w1} + semantic_similarity * {w2} + random_jitter * {w3}",
        weight_slots=["pheromone_repulsion", "semantic_similarity", "random_jitter"],
        default_weights={"pheromone_repulsion": 0.45, "semantic_similarity": 0.35, "random_jitter": 0.15},
        description="Maximizes agent spread/dispersion",
    ),

    "coverage_focused": ExpressionTemplate(
        name="coverage_focused",
        category="movement",
        template="pheromone_repulsion * {w1} + node_centrality * {w2} + semantic_similarity * {w3}",
        weight_slots=["pheromone_repulsion", "node_centrality", "semantic_similarity"],
        default_weights={"pheromone_repulsion": 0.35, "node_centrality": 0.3, "semantic_similarity": 0.3},
        description="Optimizes for broad coverage",
    ),

    "exploitation": ExpressionTemplate(
        name="exploitation",
        category="movement",
        template="semantic_similarity * {w1} + node_centrality * {w2}",
        weight_slots=["semantic_similarity", "node_centrality"],
        default_weights={"semantic_similarity": 0.8, "node_centrality": 0.15},
        description="Pure exploitation strategy",
    ),

    "efficient": ExpressionTemplate(
        name="efficient",
        category="movement",
        template="semantic_similarity * {w1} + node_centrality * {w2}",
        weight_slots=["semantic_similarity", "node_centrality"],
        default_weights={"semantic_similarity": 0.7, "node_centrality": 0.2},
        description="Cost-efficient movement",
    ),

    "quality_focused": ExpressionTemplate(
        name="quality_focused",
        category="movement",
        template="semantic_similarity * {w1} + pheromone_repulsion * {w2} + node_centrality * {w3}",
        weight_slots=["semantic_similarity", "pheromone_repulsion", "node_centrality"],
        default_weights={"semantic_similarity": 0.6, "pheromone_repulsion": 0.2, "node_centrality": 0.15},
        description="Optimizes for result quality",
    ),

    "fast": ExpressionTemplate(
        name="fast",
        category="movement",
        template="semantic_similarity * {w1}",
        weight_slots=["semantic_similarity"],
        default_weights={"semantic_similarity": 0.9},
        description="Minimal computation strategy",
    ),
}

# ============================================================================
# Deposit Strategy Templates
# ============================================================================

DEPOSIT_TEMPLATES: Dict[str, ExpressionTemplate] = {
    "flat": ExpressionTemplate(
        name="flat",
        category="deposit",
        template="1.0",
        weight_slots=[],
        default_weights={},
        description="Uniform deposit",
    ),

    "semantic_deposit": ExpressionTemplate(
        name="semantic_deposit",
        category="deposit",
        template="semantic_similarity * {w1} + {w2}",
        weight_slots=["semantic_factor", "base"],
        default_weights={"semantic_factor": 0.8, "base": 0.2},
        description="Deposit proportional to semantic relevance",
    ),

    "hub_deposit": ExpressionTemplate(
        name="hub_deposit",
        category="deposit",
        template="node_centrality * {w1} + {w2}",
        weight_slots=["centrality_factor", "base"],
        default_weights={"centrality_factor": 0.7, "base": 0.3},
        description="More deposit on hub nodes",
    ),

    "exploration_bonus": ExpressionTemplate(
        name="exploration_bonus",
        category="deposit",
        template="pheromone_repulsion * {w1} + {w2}",
        weight_slots=["exploration_factor", "base"],
        default_weights={"exploration_factor": 0.6, "base": 0.4},
        description="Bonus for unvisited areas",
    ),

    "balanced_deposit": ExpressionTemplate(
        name="balanced_deposit",
        category="deposit",
        template="semantic_similarity * {w1} + node_centrality * {w2} + {w3}",
        weight_slots=["semantic_factor", "centrality_factor", "base"],
        default_weights={"semantic_factor": 0.4, "centrality_factor": 0.3, "base": 0.3},
        description="Balanced deposit strategy",
    ),
}

# ============================================================================
# Ranking Strategy Templates
# ============================================================================

RANKING_TEMPLATES: Dict[str, ExpressionTemplate] = {
    "semantic_rank": ExpressionTemplate(
        name="semantic_rank",
        category="ranking",
        template="semantic_similarity",
        weight_slots=[],
        default_weights={},
        description="Rank by semantic similarity",
    ),

    "visit_weighted": ExpressionTemplate(
        name="visit_weighted",
        category="ranking",
        template="semantic_similarity * {w1} + percentage_visited * {w2}",
        weight_slots=["semantic_weight", "visit_weight"],
        default_weights={"semantic_weight": 0.7, "visit_weight": 0.3},
        description="Weight visits in ranking",
    ),

    "pure_visits": ExpressionTemplate(
        name="pure_visits",
        category="ranking",
        template="percentage_visited",
        weight_slots=[],
        default_weights={},
        description="Rank purely by visit frequency",
    ),
}


# ============================================================================
# Template Registry
# ============================================================================

ALL_TEMPLATES: Dict[str, Dict[str, ExpressionTemplate]] = {
    "movement": MOVEMENT_TEMPLATES,
    "deposit": DEPOSIT_TEMPLATES,
    "ranking": RANKING_TEMPLATES,
}


def get_template(category: str, name: str) -> Optional[ExpressionTemplate]:
    """
    Get a template by category and name.

    Args:
        category: 'movement', 'deposit', or 'ranking'
        name: Template name

    Returns:
        ExpressionTemplate or None if not found
    """
    return ALL_TEMPLATES.get(category, {}).get(name)


def get_templates_for_category(category: str) -> Dict[str, ExpressionTemplate]:
    """
    Get all templates for a category.

    Args:
        category: 'movement', 'deposit', or 'ranking'

    Returns:
        Dict of template_name -> ExpressionTemplate
    """
    return ALL_TEMPLATES.get(category, {})


def get_templates_for_intent(
    intent: MutationIntent,
    category: str = "movement"
) -> List[ExpressionTemplate]:
    """
    Get templates suitable for a given intent.

    Args:
        intent: The mutation intent
        category: Template category

    Returns:
        List of suitable templates
    """
    action = get_intent_action(intent)
    preferred = action.preferred_templates

    templates = get_templates_for_category(category)

    # Get preferred templates that exist
    result = []
    for name in preferred:
        if name in templates:
            result.append(templates[name])

    # If no preferred templates, return a reasonable default
    if not result:
        if category == "movement":
            result = [templates.get("balanced", templates.get("semantic_focused"))]
        elif category == "deposit":
            result = [templates.get("flat")]
        elif category == "ranking":
            result = [templates.get("semantic_rank")]
        result = [t for t in result if t is not None]

    return result


class SafeExpressionBuilder:
    """
    Builds safe, validated expressions from templates and intents.

    This is the key component that eliminates LLM expression parsing errors
    by only allowing template-based generation.
    """

    def __init__(self):
        self.templates = ALL_TEMPLATES

    def build_from_intent(
        self,
        intent: MutationIntent,
        category: str,
        confidence: float = 0.5,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[ExpressionNode, str]:
        """
        Build an expression based on mutation intent.

        Args:
            intent: The mutation intent
            category: 'movement', 'deposit', or 'ranking'
            confidence: How aggressively to apply the intent (0.0-1.0)
            current_weights: Current weight values (for incremental changes)

        Returns:
            Tuple of (ExpressionNode, template_name)
        """
        action = get_intent_action(intent)
        templates = get_templates_for_intent(intent, category)

        if not templates:
            # Fallback to a default template
            default_templates = {
                "movement": "balanced",
                "deposit": "flat",
                "ranking": "semantic_rank",
            }
            template = get_template(category, default_templates.get(category, "balanced"))
            if template is None:
                raise ValueError(f"No templates available for category: {category}")
        else:
            # Select template (higher confidence = more specialized template)
            if confidence > 0.7 and len(templates) > 1:
                template = templates[0]  # Most specialized
            else:
                template = random.choice(templates)

        # Compute weights based on intent action and confidence
        weights = self._compute_weights(
            template=template,
            action=action,
            confidence=confidence,
            current_weights=current_weights,
        )

        try:
            node = template.parse(weights)
            return node, template.name
        except Exception as e:
            logger.warning(f"Template parsing failed for {template.name}: {e}")
            # Fallback to simplest expression
            return ExpressionParser.parse("semantic_similarity"), "fallback"

    def build_from_template(
        self,
        category: str,
        template_name: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> ExpressionNode:
        """
        Build an expression from a specific template.

        Args:
            category: 'movement', 'deposit', or 'ranking'
            template_name: Name of the template
            weights: Weight values for slots

        Returns:
            Parsed ExpressionNode
        """
        template = get_template(category, template_name)
        if template is None:
            raise ValueError(f"Template not found: {category}/{template_name}")

        return template.parse(weights)

    def mutate_expression(
        self,
        current_node: ExpressionNode,
        intent: MutationIntent,
        category: str,
        confidence: float = 0.5,
    ) -> Tuple[ExpressionNode, str]:
        """
        Mutate an existing expression based on intent.

        Args:
            current_node: Current expression (may be used for weight extraction)
            intent: Mutation intent
            category: Expression category
            confidence: Mutation aggressiveness

        Returns:
            Tuple of (new_ExpressionNode, template_name)
        """
        # Extract current weights from node if possible
        current_weights = self._extract_weights(current_node)

        # Build new expression
        return self.build_from_intent(
            intent=intent,
            category=category,
            confidence=confidence,
            current_weights=current_weights,
        )

    def _compute_weights(
        self,
        template: ExpressionTemplate,
        action: Any,  # IntentAction
        confidence: float,
        current_weights: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Compute weight values based on intent action.

        Args:
            template: Target template
            action: IntentAction with weight ranges
            confidence: How much to push toward intent-optimal values
            current_weights: Current weight values

        Returns:
            Dict of slot_name -> weight value
        """
        weights = {}
        current_weights = current_weights or {}

        for slot_name in template.weight_slots:
            # Get range from intent action
            min_w, max_w = action.weight_ranges.get(slot_name, (0.1, 0.9))

            # Get current value or default
            current = current_weights.get(
                slot_name,
                template.default_weights.get(slot_name, 0.5)
            )

            # Interpolate toward target based on confidence
            # High confidence = move strongly toward intent-optimal range
            target = (min_w + max_w) / 2

            if confidence > 0.7:
                # Strong move toward target
                new_weight = current + (target - current) * 0.8
            elif confidence > 0.4:
                # Moderate move
                new_weight = current + (target - current) * 0.5
            else:
                # Light adjustment
                new_weight = current + (target - current) * 0.2

            # Add some noise for diversity
            noise = random.gauss(0, 0.05 * confidence)
            new_weight = new_weight + noise

            # Clamp to valid range
            new_weight = max(min_w * 0.8, min(max_w * 1.2, new_weight))
            new_weight = max(0.0, min(1.0, new_weight))

            weights[slot_name] = round(new_weight, 2)

        return weights

    def _extract_weights(self, node: ExpressionNode) -> Dict[str, float]:
        """
        Extract weight values from an existing expression node.

        This is a best-effort extraction - expressions may have
        complex structures that don't map cleanly to weights.

        Args:
            node: Expression node to analyze

        Returns:
            Dict of heuristic_name -> weight value
        """
        weights = {}
        self._extract_weights_recursive(node, weights, 1.0)
        return weights

    def _extract_weights_recursive(
        self,
        node: ExpressionNode,
        weights: Dict[str, float],
        multiplier: float,
    ):
        """
        Recursively extract weights from an expression tree.

        Looks for patterns like: feature * constant
        """
        if node.type == 'feature':
            # Standalone feature gets the current multiplier as weight
            name = str(node.value)
            weights[name] = weights.get(name, 0) + multiplier
            return

        if node.type == 'const':
            return

        if node.type == 'op' and node.value == '*':
            # Multiplication: look for feature * const
            if len(node.children) >= 2:
                left, right = node.children[0], node.children[1]

                if left.type == 'feature' and right.type == 'const':
                    name = str(left.value)
                    weights[name] = weights.get(name, 0) + float(right.value) * multiplier
                    return
                elif right.type == 'feature' and left.type == 'const':
                    name = str(right.value)
                    weights[name] = weights.get(name, 0) + float(left.value) * multiplier
                    return

        # Recurse into children
        for child in node.children:
            self._extract_weights_recursive(child, weights, multiplier)

    def list_templates(self, category: Optional[str] = None) -> List[str]:
        """
        List available template names.

        Args:
            category: Optional category filter

        Returns:
            List of template names
        """
        if category:
            return list(self.templates.get(category, {}).keys())
        else:
            result = []
            for cat, templates in self.templates.items():
                result.extend(f"{cat}/{name}" for name in templates.keys())
            return result
