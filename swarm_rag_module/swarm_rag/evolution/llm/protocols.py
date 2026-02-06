"""
Protocol definitions for LLM components.

Defines interfaces (Protocols) for LLM components that can be used
without importing the actual implementations. Includes null implementations
for when LLM is disabled.
"""
from typing import Protocol, Dict, List, Optional, Any, Tuple, TypedDict

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


# =============================================================================
# Return TypedDicts for protocol methods
# =============================================================================

class StrategicSummary(TypedDict, total=False):
    """Return type for JournalProtocol.summarize_for_strategic()."""
    qd_score_trend: str
    qd_score_slope: float
    stagnation_generations: int
    total_mutations: int
    recent_qd_scores: List[float]
    successful_intents: Dict[str, float]
    failed_intents: Dict[str, float]
    recent_failed_diagnoses: List[str]


class TacticalSummary(TypedDict, total=False):
    """Return type for JournalProtocol.summarize_for_tactical()."""
    lineage_intents: List[str]
    lineage_successes: List[bool]
    successful_intents_global: List[str]
    failed_intents_global: List[str]


class JournalDict(TypedDict):
    """Return type for JournalProtocol.to_dict()."""
    records: List[Dict[str, Any]]
    generation_summaries: List[Dict[str, Any]]
    qd_score_history: List[float]


class TrackerSummary(TypedDict, total=False):
    """Return type for TrackerProtocol.to_summary_dict()."""
    n_agents: int
    n_steps: int
    decisions_captured: int
    sampling_mode: NotRequired[str]
    trajectory: NotRequired[Dict[str, Any]]
    heuristic_usage: NotRequired[Dict[str, Dict[str, float]]]
    choice_patterns: NotRequired[Dict[str, float]]
    sample_paths: NotRequired[List[Dict[str, Any]]]
    node_hotspots: NotRequired[List[Dict[str, Any]]]
    stuck_nodes: NotRequired[Dict[str, List[Any]]]
    edge_case_summary: NotRequired[Dict[str, Any]]


class _NullMutationRecord:
    """Sentinel returned by NullJournal.record_mutation(). Evaluates as falsy."""
    __slots__ = ()
    def __repr__(self) -> str:
        return 'NullMutationRecord()'
    def __bool__(self) -> bool:
        return False

_NULL_MUTATION_RECORD = _NullMutationRecord()


class JournalProtocol(Protocol):
    """Protocol for evolution journal (mutation history tracking)."""

    def record_mutation(
        self,
        generation: int,
        genome_id: str,
        parent_genome_id: Optional[str],
        diagnosis: str,
        intent: Any,  # MutationIntent
        target_component: Any,  # TargetComponent
        confidence: float,
        param_changes: Dict[str, Any],
        strategy_changes: Dict[str, str],
        fitness_before: float,
    ) -> Any:
        """Record a mutation attempt."""
        ...

    def update_outcome(
        self,
        record: Any,
        fitness_after: float,
        added_to_archive: bool,
        replaced_existing: bool = False,
    ) -> None:
        """Update a mutation record with its outcome."""
        ...

    def finalize_generation(
        self,
        generation: int,
        qd_score: float,
        coverage: float,
    ) -> None:
        """Finalize statistics for a generation."""
        ...

    def summarize_for_strategic(self) -> StrategicSummary:
        """Build summary for strategic decisions."""
        ...

    def summarize_for_tactical(self, genome_id: str) -> TacticalSummary:
        """Build summary for tactical decisions."""
        ...

    def to_dict(self) -> JournalDict:
        """Serialize journal for checkpointing."""
        ...


class TrackerProtocol(Protocol):
    """Protocol for decision tracking (agent behavior analysis)."""

    def start_query(self, query_id: Any, n_agents: int, n_steps: int) -> None:
        """Begin tracking for a new query."""
        ...

    def record_decision(
        self,
        agent_id: int,
        step: int,
        current_node: int,
        candidates: List[int],
        heuristic_scores: Dict[str, Any],
        final_scores: Any,
        chosen_node: int,
        chosen_index: int,
        deposit: float
    ) -> None:
        """Record a single agent decision."""
        ...

    def clear(self) -> None:
        """Reset tracker for reuse."""
        ...

    def to_summary_dict(self, agent_trajectories: Optional[List[List[int]]] = None) -> TrackerSummary:
        """Convert tracked data to a summary dict."""
        ...

    @property
    def decision_count(self) -> int:
        """Number of decisions captured."""
        ...


class MutatorProtocol(Protocol):
    """Protocol for LLM-guided mutation."""

    def mutate(
        self,
        genome: Any,
        journal: Optional[JournalProtocol] = None,
    ) -> Tuple[Any, Any]:
        """Perform mutation on a genome."""
        ...

    def update_directive(
        self,
        context: Any,
        journal: Optional[JournalProtocol] = None,
    ) -> Any:
        """Update the strategic directive."""
        ...

    def should_update_oracle(
        self,
        generation: int,
        stagnation_count: int,
    ) -> bool:
        """Check if oracle should be updated."""
        ...


class NullJournal:
    """No-op journal implementation for when LLM is disabled."""

    def record_mutation(
        self,
        generation: int,
        genome_id: str,
        parent_genome_id: Optional[str],
        diagnosis: str,
        intent: Any,
        target_component: Any,
        confidence: float,
        param_changes: Dict[str, Any],
        strategy_changes: Dict[str, str],
        fitness_before: float,
    ) -> _NullMutationRecord:
        """No-op: returns a falsy sentinel that satisfies the protocol contract."""
        return _NULL_MUTATION_RECORD

    def update_outcome(
        self,
        record: Any,
        fitness_after: float,
        added_to_archive: bool,
        replaced_existing: bool = False,
    ) -> None:
        """No-op."""
        pass

    def finalize_generation(
        self,
        generation: int,
        qd_score: float,
        coverage: float,
    ) -> None:
        """No-op."""
        pass

    def summarize_for_strategic(self) -> StrategicSummary:
        """Return empty summary."""
        return {}

    def summarize_for_tactical(self, genome_id: str) -> TacticalSummary:
        """Return empty summary."""
        return {}

    def to_dict(self) -> JournalDict:
        """Return empty dict."""
        return {"records": [], "generation_summaries": [], "qd_score_history": []}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NullJournal":
        """Create from dict (ignores data)."""
        return cls()


class NullTracker:
    """No-op tracker implementation for when LLM is disabled.

    Accepts and ignores arbitrary keyword arguments (e.g. sample_rate,
    sampling_mode, max_decisions) so callers can pass DecisionTracker
    constructor args without error.
    """

    def __init__(self, enabled: bool = False, **kwargs: Any):
        self.enabled = False
        self._decision_count = 0
        # kwargs intentionally ignored -- this stub does not track decisions.

    def start_query(self, query_id: Any, n_agents: int, n_steps: int) -> None:
        """No-op."""
        pass

    def record_decision(
        self,
        agent_id: int,
        step: int,
        current_node: int,
        candidates: List[int],
        heuristic_scores: Dict[str, Any],
        final_scores: Any,
        chosen_node: int,
        chosen_index: int,
        deposit: float
    ) -> None:
        """No-op."""
        pass

    def clear(self) -> None:
        """No-op."""
        pass

    def to_summary_dict(self, agent_trajectories: Optional[List[List[int]]] = None) -> TrackerSummary:
        """Return empty summary."""
        return {"n_agents": 0, "n_steps": 0, "decisions_captured": 0}

    @property
    def decision_count(self) -> int:
        """Always returns 0."""
        return 0
