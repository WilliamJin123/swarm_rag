"""
LLM-guided evolution module.

Provides a three-tier architecture for LLM-guided genome evolution:

Tier 1 - Strategic Oracle:
    High-level evolution steering based on archive statistics.
    Called periodically or on stagnation detection.

Tier 2 - Tactical Advisor:
    Per-genome diagnosis and mutation intent prescription.
    Determines WHAT to change, not HOW.

Tier 3 - Constrained Executor:
    Deterministic translation of intents to validated mutations.
    Uses safe expression templates to avoid parsing errors.

Key Components:
- EvolutionJournal: Tracks mutation history for learning loops
- SafeExpressionBuilder: Template-based expression generation
- MutationIntent: Enum of possible mutation goals
- StrategicDirective: Output of Strategic Oracle
- MutationPrescription: Output of Tactical Advisor
"""

# Intents and data classes
from .intents import (
    MutationIntent,
    EvolutionMode,
    TargetComponent,
    StrategicDirective,
    MutationPrescription,
    IntentAction,
    get_intent_action,
    get_complementary_intents,
    get_conflicting_intents,
    INTENT_ACTIONS,
)

# Evolution Journal
from .evolution_journal import (
    EvolutionJournal,
    MutationRecord,
    GenerationSummary,
)

# Expression Builder
from .expression_builder import (
    SafeExpressionBuilder,
    ExpressionTemplate,
    get_template,
    get_templates_for_category,
    get_templates_for_intent,
    MOVEMENT_TEMPLATES,
    DEPOSIT_TEMPLATES,
    RANKING_TEMPLATES,
)

# Strategic Oracle (Tier 1)
from .strategic_oracle import (
    StrategicOracle,
    StrategicContext,
    build_strategic_context,
)

# Tactical Advisor (Tier 2)
from .tactical_advisor import (
    TacticalAdvisor,
    TacticalContext,
    build_tactical_context,
)

# Constrained Executor (Tier 3)
from .constrained_executor import (
    ConstrainedExecutor,
    ExecutionResult,
    ParameterBounds,
    ThreeTierMutator,
)

# Legacy components
from .provider import BaseLLMProvider, LLMResponse
from .parsers import ExpressionParser
from .utils import (
    genome_to_json_context,
    apply_llm_edits,
    GenomeLLMContext,
    GenomePerformance,
    GenomeConfig,
    BehavioralMetrics,
    EvolutionaryContext,
)

__all__ = [
    # Intents
    "MutationIntent",
    "EvolutionMode",
    "TargetComponent",
    "StrategicDirective",
    "MutationPrescription",
    "IntentAction",
    "get_intent_action",
    "get_complementary_intents",
    "get_conflicting_intents",
    "INTENT_ACTIONS",
    # Journal
    "EvolutionJournal",
    "MutationRecord",
    "GenerationSummary",
    # Expression Builder
    "SafeExpressionBuilder",
    "ExpressionTemplate",
    "get_template",
    "get_templates_for_category",
    "get_templates_for_intent",
    "MOVEMENT_TEMPLATES",
    "DEPOSIT_TEMPLATES",
    "RANKING_TEMPLATES",
    # Strategic Oracle
    "StrategicOracle",
    "StrategicContext",
    "build_strategic_context",
    # Tactical Advisor
    "TacticalAdvisor",
    "TacticalContext",
    "build_tactical_context",
    # Constrained Executor
    "ConstrainedExecutor",
    "ExecutionResult",
    "ParameterBounds",
    "ThreeTierMutator",
    # Legacy
    "BaseLLMProvider",
    "LLMResponse",
    "ExpressionParser",
    "genome_to_json_context",
    "apply_llm_edits",
    "GenomeLLMContext",
    "GenomePerformance",
    "GenomeConfig",
    "BehavioralMetrics",
    "EvolutionaryContext",
]
