"""
LLM-guided evolution module.

All LLM API calls flow through LLMClient (client.py).

Three-tier architecture:
    Tier 1 - Strategic Oracle: Archive-level steering (every N generations)
    Tier 2 - Tactical Advisor: Per-genome diagnosis and intent prescription
    Tier 2.5 - Creative Synthesizer: Optional custom expression generation
    Tier 3 - Constrained Executor: Deterministic intent-to-mutation (NO LLM)

Key Components:
- LLMClient: Single source of truth for all LLM API calls
- EvolutionJournal: Tracks mutation history for learning loops
- SafeExpressionBuilder: Template-based expression generation
- MutationIntent: Enum of possible mutation goals
"""

# LLM Client (single source of truth for all API calls)
from .client import LLMClient, LLMCallResult, SUPPORTED_PROVIDERS
from .factory import LLMClientFactory

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

# Creative Synthesizer (Tier 2.5)
from .creative_synthesizer import (
    CreativeSynthesizer,
    CreativeProposal,
    CreativeModeContext,
    should_use_creative_mode,
    build_creative_context,
    AVAILABLE_HEURISTICS,
    ALLOWED_OPERATORS,
    ALLOWED_FUNCTIONS,
    MAX_EXPRESSION_COMPLEXITY,
)

# Utils
from .parsers import ExpressionParser
from .utils import genome_to_json_context

__all__ = [
    # LLM Client
    "LLMClient",
    "LLMCallResult",
    "LLMClientFactory",
    "SUPPORTED_PROVIDERS",
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
    # Creative Synthesizer
    "CreativeSynthesizer",
    "CreativeProposal",
    "CreativeModeContext",
    "should_use_creative_mode",
    "build_creative_context",
    "AVAILABLE_HEURISTICS",
    "ALLOWED_OPERATORS",
    "ALLOWED_FUNCTIONS",
    "MAX_EXPRESSION_COMPLEXITY",
    # Utils
    "ExpressionParser",
    "genome_to_json_context",
]
