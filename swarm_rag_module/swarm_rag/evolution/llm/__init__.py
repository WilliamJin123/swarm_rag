"""
LLM-guided evolution module.

All LLM API calls flow through LLMClient (client.py).

Architecture:
- LLMMutationEngine: Main interface for LLM-guided mutations (orchestrates all tiers)
- LLMClient: Single source of truth for all LLM API calls
- EvolutionJournal: Tracks mutation history for learning loops
- DecisionTracker: Captures agent decision data for behavioral analysis
"""

# LLM Client (single source of truth for all API calls)
from .client import LLMClient, LLMCallResult, SUPPORTED_PROVIDERS
from .factory import LLMClientFactory

# Decision Tracker
from .decision_tracker import (
    DecisionTracker,
    SmartDecisionTracker,  # Alias for backwards compatibility
    AgentDecision,
    TrajectoryMetrics,
    QueryDecisionContext,
)

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

# Constrained Executor and LLM Mutation Engine
from .constrained_executor import (
    ConstrainedExecutor,
    ExecutionResult,
    ParameterBounds,
    ThreeTierMutator,
)

# LLMMutationEngine is the main interface (alias for ThreeTierMutator)
LLMMutationEngine = ThreeTierMutator

# Internal tier components (used by LLMMutationEngine)
from .strategic_oracle import StrategicOracle, StrategicContext, build_strategic_context
from .tactical_advisor import TacticalAdvisor, TacticalContext, build_tactical_context
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
    # Main interface
    "LLMMutationEngine",
    "ThreeTierMutator",  # Alias for backwards compatibility
    # LLM Client
    "LLMClient",
    "LLMCallResult",
    "LLMClientFactory",
    "SUPPORTED_PROVIDERS",
    # Decision Tracker
    "DecisionTracker",
    "SmartDecisionTracker",
    "AgentDecision",
    "TrajectoryMetrics",
    "QueryDecisionContext",
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
    # Executor
    "ConstrainedExecutor",
    "ExecutionResult",
    "ParameterBounds",
    # Tier components (internal)
    "StrategicOracle",
    "StrategicContext",
    "build_strategic_context",
    "TacticalAdvisor",
    "TacticalContext",
    "build_tactical_context",
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
