"""
Evolution Journal - History tracking for mutation decisions.

Tracks what was tried and what worked, enabling learning loops where
the LLM can avoid repeating failed experiments.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import json
import logging
from datetime import datetime

from .intents import MutationIntent, EvolutionMode, TargetComponent

logger = logging.getLogger(__name__)


@dataclass
class MutationRecord:
    """
    Records a single mutation event with its outcome.
    """
    generation: int
    genome_id: str
    parent_genome_id: Optional[str]

    # What was diagnosed/intended
    diagnosis: str
    intent: MutationIntent
    target_component: TargetComponent
    confidence: float

    # What was actually changed
    param_changes: Dict[str, Any] = field(default_factory=dict)
    strategy_changes: Dict[str, str] = field(default_factory=dict)

    # Outcome (filled in after evaluation)
    added_to_archive: bool = False
    replaced_existing: bool = False
    fitness_before: float = 0.0
    fitness_after: float = 0.0
    fitness_delta: float = 0.0

    # Timestamp for ordering
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def was_successful(self) -> bool:
        """Whether the mutation improved fitness or found a new niche."""
        return self.added_to_archive or self.fitness_delta > 0.01

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "genome_id": self.genome_id,
            "parent_genome_id": self.parent_genome_id,
            "diagnosis": self.diagnosis,
            "intent": self.intent.value,
            "target_component": self.target_component.value,
            "confidence": self.confidence,
            "param_changes": self.param_changes,
            "strategy_changes": self.strategy_changes,
            "added_to_archive": self.added_to_archive,
            "replaced_existing": self.replaced_existing,
            "fitness_before": self.fitness_before,
            "fitness_after": self.fitness_after,
            "fitness_delta": self.fitness_delta,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MutationRecord":
        return cls(
            generation=data.get("generation", 0),
            genome_id=data.get("genome_id", ""),
            parent_genome_id=data.get("parent_genome_id"),
            diagnosis=data.get("diagnosis", ""),
            intent=MutationIntent(data.get("intent", "no_change")),
            target_component=TargetComponent(data.get("target_component", "all")),
            confidence=data.get("confidence", 0.5),
            param_changes=data.get("param_changes", {}),
            strategy_changes=data.get("strategy_changes", {}),
            added_to_archive=data.get("added_to_archive", False),
            replaced_existing=data.get("replaced_existing", False),
            fitness_before=data.get("fitness_before", 0.0),
            fitness_after=data.get("fitness_after", 0.0),
            fitness_delta=data.get("fitness_delta", 0.0),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class GenerationSummary:
    """
    Summary statistics for a single generation.
    """
    generation: int
    total_mutations: int
    successful_mutations: int
    archive_additions: int
    avg_fitness_delta: float
    intent_distribution: Dict[str, int]
    qd_score: float = 0.0
    coverage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "total_mutations": self.total_mutations,
            "successful_mutations": self.successful_mutations,
            "archive_additions": self.archive_additions,
            "avg_fitness_delta": self.avg_fitness_delta,
            "intent_distribution": self.intent_distribution,
            "qd_score": self.qd_score,
            "coverage": self.coverage,
        }


class EvolutionJournal:
    """
    Maintains history of mutation decisions and their outcomes.

    Provides:
    1. Success rate tracking by intent
    2. Failed experiment avoidance
    3. Trend analysis for strategic decisions
    4. Serialization for checkpointing
    """

    def __init__(self, max_records: int = 10000):
        """
        Initialize the journal.

        Args:
            max_records: Maximum records to keep (FIFO eviction)
        """
        self.max_records = max_records
        self.records: List[MutationRecord] = []
        self.generation_summaries: List[GenerationSummary] = []

        # Quick lookup indexes
        self._intent_counts: Dict[MutationIntent, int] = defaultdict(int)
        self._intent_successes: Dict[MutationIntent, int] = defaultdict(int)
        self._recent_diagnoses: List[str] = []
        self._qd_score_history: List[float] = []

    def record_mutation(
        self,
        generation: int,
        genome_id: str,
        parent_genome_id: Optional[str],
        diagnosis: str,
        intent: MutationIntent,
        target_component: TargetComponent,
        confidence: float,
        param_changes: Dict[str, Any],
        strategy_changes: Dict[str, str],
        fitness_before: float,
    ) -> MutationRecord:
        """
        Record a mutation attempt (outcome filled in later).

        Args:
            generation: Current generation
            genome_id: ID of the mutated genome
            parent_genome_id: ID of the parent genome
            diagnosis: LLM's diagnosis
            intent: The mutation intent
            target_component: What was targeted
            confidence: LLM's confidence
            param_changes: Parameter changes made
            strategy_changes: Strategy expression changes
            fitness_before: Parent's fitness

        Returns:
            MutationRecord to be updated with outcome
        """
        record = MutationRecord(
            generation=generation,
            genome_id=genome_id,
            parent_genome_id=parent_genome_id,
            diagnosis=diagnosis,
            intent=intent,
            target_component=target_component,
            confidence=confidence,
            param_changes=param_changes,
            strategy_changes=strategy_changes,
            fitness_before=fitness_before,
        )

        self.records.append(record)
        self._intent_counts[intent] += 1
        self._recent_diagnoses.append(diagnosis)

        # Trim to max size
        if len(self.records) > self.max_records:
            removed = self.records.pop(0)
            self._intent_counts[removed.intent] -= 1
            if removed.was_successful:
                self._intent_successes[removed.intent] -= 1

        if len(self._recent_diagnoses) > 100:
            self._recent_diagnoses.pop(0)

        return record

    def update_outcome(
        self,
        record: MutationRecord,
        fitness_after: float,
        added_to_archive: bool,
        replaced_existing: bool = False,
    ):
        """
        Update a mutation record with its outcome.

        Args:
            record: The record to update
            fitness_after: Fitness after evaluation
            added_to_archive: Whether it was added to archive
            replaced_existing: Whether it replaced an existing elite
        """
        record.fitness_after = fitness_after
        record.fitness_delta = fitness_after - record.fitness_before
        record.added_to_archive = added_to_archive
        record.replaced_existing = replaced_existing

        if record.was_successful:
            self._intent_successes[record.intent] += 1

    def finalize_generation(
        self,
        generation: int,
        qd_score: float,
        coverage: float,
    ):
        """
        Finalize statistics for a generation.

        Args:
            generation: Generation number
            qd_score: Current QD score
            coverage: Current archive coverage
        """
        gen_records = [r for r in self.records if r.generation == generation]

        if not gen_records:
            return

        intent_dist = defaultdict(int)
        for r in gen_records:
            intent_dist[r.intent.value] += 1

        successful = sum(1 for r in gen_records if r.was_successful)
        archive_adds = sum(1 for r in gen_records if r.added_to_archive)
        avg_delta = sum(r.fitness_delta for r in gen_records) / len(gen_records)

        summary = GenerationSummary(
            generation=generation,
            total_mutations=len(gen_records),
            successful_mutations=successful,
            archive_additions=archive_adds,
            avg_fitness_delta=avg_delta,
            intent_distribution=dict(intent_dist),
            qd_score=qd_score,
            coverage=coverage,
        )

        self.generation_summaries.append(summary)
        self._qd_score_history.append(qd_score)

        # Keep only recent history
        if len(self._qd_score_history) > 100:
            self._qd_score_history.pop(0)

    def get_success_rate_by_intent(self) -> Dict[MutationIntent, float]:
        """
        Get success rate for each intent type.

        Returns:
            Dict mapping intent to success rate (0.0-1.0)
        """
        rates = {}
        for intent in MutationIntent:
            count = self._intent_counts.get(intent, 0)
            if count > 0:
                rates[intent] = self._intent_successes.get(intent, 0) / count
            else:
                rates[intent] = 0.5  # Unknown, assume neutral
        return rates

    def get_successful_intents(self, min_rate: float = 0.3, min_count: int = 5) -> List[MutationIntent]:
        """
        Get intents that have been successful.

        Args:
            min_rate: Minimum success rate
            min_count: Minimum number of attempts

        Returns:
            List of successful intent types
        """
        rates = self.get_success_rate_by_intent()
        successful = []
        for intent, rate in rates.items():
            if rate >= min_rate and self._intent_counts.get(intent, 0) >= min_count:
                successful.append(intent)
        return successful

    def get_failed_intents(self, max_rate: float = 0.1, min_count: int = 5) -> List[MutationIntent]:
        """
        Get intents that have consistently failed.

        Args:
            max_rate: Maximum success rate to be considered failed
            min_count: Minimum number of attempts

        Returns:
            List of failed intent types
        """
        rates = self.get_success_rate_by_intent()
        failed = []
        for intent, rate in rates.items():
            if rate <= max_rate and self._intent_counts.get(intent, 0) >= min_count:
                failed.append(intent)
        return failed

    def get_qd_score_trend(self, window: int = 5) -> Tuple[float, str]:
        """
        Get QD score trend over recent generations.

        Args:
            window: Number of generations to consider

        Returns:
            Tuple of (slope, trend_description)
        """
        if len(self._qd_score_history) < 2:
            return 0.0, "insufficient_data"

        recent = self._qd_score_history[-window:]
        if len(recent) < 2:
            return 0.0, "insufficient_data"

        # Simple linear trend
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0, "flat"

        slope = numerator / denominator

        if slope > 0.01:
            return slope, "improving"
        elif slope < -0.01:
            return slope, "declining"
        else:
            return slope, "stagnant"

    def get_stagnation_count(self, threshold: float = 0.001) -> int:
        """
        Count consecutive generations with minimal QD improvement.

        Args:
            threshold: Minimum improvement to not be considered stagnant

        Returns:
            Number of stagnant generations
        """
        if len(self._qd_score_history) < 2:
            return 0

        count = 0
        for i in range(len(self._qd_score_history) - 1, 0, -1):
            delta = self._qd_score_history[i] - self._qd_score_history[i - 1]
            if abs(delta) < threshold:
                count += 1
            else:
                break
        return count

    def get_recent_failed_diagnoses(self, limit: int = 10) -> List[str]:
        """
        Get diagnoses from recent failed mutations.

        Args:
            limit: Maximum number to return

        Returns:
            List of diagnosis strings
        """
        failed = [
            r.diagnosis
            for r in reversed(self.records[-100:])
            if not r.was_successful
        ]
        return failed[:limit]

    def summarize_for_strategic(self) -> Dict[str, Any]:
        """
        Build summary for Strategic Oracle (Tier 1).

        Returns:
            Dict with aggregated information for strategic decisions
        """
        success_rates = self.get_success_rate_by_intent()
        slope, trend = self.get_qd_score_trend()
        stagnation = self.get_stagnation_count()

        # Intent success summary (for LLM consumption)
        successful_intents = {
            intent.value: rate
            for intent, rate in success_rates.items()
            if rate > 0.3 and self._intent_counts.get(intent, 0) >= 3
        }
        failed_intents = {
            intent.value: rate
            for intent, rate in success_rates.items()
            if rate < 0.15 and self._intent_counts.get(intent, 0) >= 3
        }

        return {
            "qd_score_trend": trend,
            "qd_score_slope": slope,
            "stagnation_generations": stagnation,
            "total_mutations": len(self.records),
            "recent_qd_scores": self._qd_score_history[-10:],
            "successful_intents": successful_intents,
            "failed_intents": failed_intents,
            "recent_failed_diagnoses": self.get_recent_failed_diagnoses(5),
        }

    def summarize_for_tactical(self, genome_id: str) -> Dict[str, Any]:
        """
        Build summary for Tactical Advisor (Tier 2).

        Args:
            genome_id: The genome being mutated

        Returns:
            Dict with relevant history for this genome's lineage
        """
        # Find parent mutations
        lineage_records = []
        current_id = genome_id
        for record in reversed(self.records):
            if record.genome_id == current_id:
                lineage_records.append(record)
                current_id = record.parent_genome_id
                if current_id is None or len(lineage_records) >= 5:
                    break

        recent_intents = [r.intent.value for r in lineage_records]
        recent_successes = [r.was_successful for r in lineage_records]

        return {
            "lineage_intents": recent_intents,
            "lineage_successes": recent_successes,
            "successful_intents_global": [
                i.value for i in self.get_successful_intents()
            ],
            "failed_intents_global": [
                i.value for i in self.get_failed_intents()
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize journal for checkpointing."""
        return {
            "records": [r.to_dict() for r in self.records],
            "generation_summaries": [s.to_dict() for s in self.generation_summaries],
            "qd_score_history": self._qd_score_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], max_records: int = 10000) -> "EvolutionJournal":
        """Restore journal from checkpoint."""
        journal = cls(max_records=max_records)

        for rec_data in data.get("records", []):
            record = MutationRecord.from_dict(rec_data)
            journal.records.append(record)
            journal._intent_counts[record.intent] += 1
            if record.was_successful:
                journal._intent_successes[record.intent] += 1

        journal._qd_score_history = data.get("qd_score_history", [])

        # Rebuild generation summaries
        for sum_data in data.get("generation_summaries", []):
            journal.generation_summaries.append(
                GenerationSummary(
                    generation=sum_data["generation"],
                    total_mutations=sum_data["total_mutations"],
                    successful_mutations=sum_data["successful_mutations"],
                    archive_additions=sum_data["archive_additions"],
                    avg_fitness_delta=sum_data["avg_fitness_delta"],
                    intent_distribution=sum_data["intent_distribution"],
                    qd_score=sum_data.get("qd_score", 0.0),
                    coverage=sum_data.get("coverage", 0.0),
                )
            )

        return journal

    def save_to_file(self, path: str):
        """Save journal to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, path: str, max_records: int = 10000) -> "EvolutionJournal":
        """Load journal from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, max_records)
