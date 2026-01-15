"""
Decision tracking for agent-level behavioral analysis.

Captures detailed decision data from agent steps for LLM-guided mutations.
Provides trajectory analysis and heuristic score statistics that help the
LLM understand WHY agents are performing a certain way.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np


@dataclass
class AgentDecision:
    """Single agent decision at one step."""
    agent_id: int
    step: int
    current_node: int
    candidates: List[int]  # neighbor node IDs considered
    scores: Dict[str, np.ndarray]  # {heuristic_name: scores per candidate}
    final_scores: np.ndarray  # combined weighted scores
    probabilities: np.ndarray  # softmax of final_scores
    chosen_node: int
    chosen_index: int
    deposit_amount: float


@dataclass
class TrajectoryMetrics:
    """Aggregated trajectory analysis across all agents."""
    total_steps: int
    unique_nodes_visited: int
    revisit_count: int  # times an agent returned to previously visited node
    dead_end_count: int  # steps where agent couldn't move
    avg_branching_factor: float  # avg number of candidates per step
    convergence_step: Optional[int]  # step where >50% agents at same node
    final_dispersion: float  # how spread out agents are at end (0=clustered, 1=dispersed)


@dataclass
class QueryDecisionContext:
    """Complete decision context for one query evaluation."""
    query_id: Any
    n_agents: int
    n_steps: int
    decisions: List[AgentDecision] = field(default_factory=list)
    trajectory_metrics: Optional[TrajectoryMetrics] = None

    # Heuristic score statistics (across all decisions)
    # Format: {"semantic_similarity": {"mean": 0.7, "std": 0.1, "min": 0.2, "max": 0.95}}
    heuristic_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


class DecisionTracker:
    """
    Captures agent decision-making data during retrieval.

    This tracker records individual agent decisions (which node was chosen,
    why it was chosen based on heuristic scores) and computes trajectory-level
    statistics. The captured data is used to provide rich context to the LLM
    during genome mutation, enabling smarter, more targeted improvements.

    Usage:
        tracker = DecisionTracker()
        tracker.start_query(query_id="q1", n_agents=20, n_steps=4)

        # During retrieval, record each decision
        tracker.record_decision(
            agent_id=0, step=0, current_node=100,
            candidates=[101, 102, 103],
            heuristic_scores={"semantic_similarity": np.array([0.8, 0.6, 0.4])},
            final_scores=np.array([0.5, 0.3, 0.2]),
            chosen_node=101, chosen_index=0, deposit=0.8
        )

        # After retrieval completes
        context = tracker.get_context(agent_trajectories)

        # Use context in LLM mutation
        llm_context = genome_to_json_context(genome, decision_context=context)
    """

    def __init__(self, enabled: bool = True, sample_rate: float = 1.0):
        """
        Args:
            enabled: Whether tracking is active
            sample_rate: Fraction of decisions to capture (0.0-1.0).
                         1.0 = capture all, 0.1 = capture 10%
        """
        self.enabled = enabled
        self.sample_rate = sample_rate
        self._decisions: List[AgentDecision] = []
        self._query_id: Any = None
        self._n_agents: int = 0
        self._n_steps: int = 0
        self._rng = np.random.default_rng()

    def should_capture(self) -> bool:
        """Check if this decision should be captured based on sample rate."""
        if not self.enabled:
            return False
        if self.sample_rate >= 1.0:
            return True
        return self._rng.random() < self.sample_rate

    def start_query(self, query_id: Any, n_agents: int, n_steps: int):
        """Begin tracking for a new query."""
        self._decisions = []
        self._query_id = query_id
        self._n_agents = n_agents
        self._n_steps = n_steps

    def record_decision(
        self,
        agent_id: int,
        step: int,
        current_node: int,
        candidates: List[int],
        heuristic_scores: Dict[str, np.ndarray],
        final_scores: np.ndarray,
        chosen_node: int,
        chosen_index: int,
        deposit: float
    ):
        """
        Record a single agent decision.

        Args:
            agent_id: Index of the agent (0 to n_agents-1)
            step: Current step in traversal (0 to n_steps-1)
            current_node: Node ID where agent is currently located
            candidates: List of neighbor node IDs considered for next move
            heuristic_scores: Dict mapping heuristic name to score array
                             (one score per candidate)
            final_scores: Combined weighted scores used for selection
            chosen_node: The node ID that was selected
            chosen_index: Index of chosen_node in candidates list
            deposit: Amount of pheromone deposited at chosen_node
        """
        if not self.should_capture():
            return

        # Compute probabilities from final scores
        total = np.sum(final_scores) + 1e-10
        probabilities = final_scores / total

        decision = AgentDecision(
            agent_id=agent_id,
            step=step,
            current_node=current_node,
            candidates=list(candidates),
            scores={k: v.copy() if isinstance(v, np.ndarray) else np.array(v)
                   for k, v in heuristic_scores.items()},
            final_scores=final_scores.copy() if isinstance(final_scores, np.ndarray)
                        else np.array(final_scores),
            probabilities=probabilities,
            chosen_node=chosen_node,
            chosen_index=chosen_index,
            deposit_amount=deposit
        )
        self._decisions.append(decision)

    def compute_trajectory_metrics(
        self,
        agent_trajectories: List[List[int]]
    ) -> TrajectoryMetrics:
        """
        Analyze completed trajectories to compute behavioral metrics.

        Args:
            agent_trajectories: List of node ID lists, one per agent

        Returns:
            TrajectoryMetrics with aggregated statistics
        """
        if not agent_trajectories:
            return TrajectoryMetrics(
                total_steps=0,
                unique_nodes_visited=0,
                revisit_count=0,
                dead_end_count=0,
                avg_branching_factor=0.0,
                convergence_step=None,
                final_dispersion=0.0
            )

        # Total steps (excluding starting positions)
        total_steps = sum(max(0, len(t) - 1) for t in agent_trajectories)

        # Unique nodes visited across all agents
        all_visited = [node for traj in agent_trajectories for node in traj]
        unique_visited = len(set(all_visited))

        # Count revisits per agent (visiting same node multiple times)
        revisits = 0
        for traj in agent_trajectories:
            seen = set()
            for node in traj:
                if node in seen:
                    revisits += 1
                seen.add(node)

        # Count dead ends (steps with no movement - agent stayed in place)
        dead_ends = sum(
            1 for d in self._decisions
            if d.chosen_node == d.current_node
        )

        # Average branching factor (number of options per decision)
        if self._decisions:
            avg_branch = np.mean([len(d.candidates) for d in self._decisions])
        else:
            avg_branch = 0.0

        # Convergence detection - find first step where >50% agents at same node
        convergence_step = None
        n_agents = len(agent_trajectories)
        max_steps = max(len(t) for t in agent_trajectories) if agent_trajectories else 0

        for step in range(max_steps):
            step_positions = [
                traj[min(step, len(traj) - 1)]
                for traj in agent_trajectories
            ]
            position_counts = Counter(step_positions)
            most_common_count = position_counts.most_common(1)[0][1]
            if most_common_count >= n_agents * 0.5:
                convergence_step = step
                break

        # Final dispersion - entropy-based measure of how spread out agents are
        final_positions = [traj[-1] for traj in agent_trajectories if traj]
        if final_positions:
            counts = Counter(final_positions)
            probs = np.array(list(counts.values())) / len(final_positions)
            max_entropy = np.log(len(agent_trajectories)) if len(agent_trajectories) > 1 else 1.0
            actual_entropy = -np.sum(probs * np.log(probs + 1e-10))
            dispersion = actual_entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            dispersion = 0.0

        return TrajectoryMetrics(
            total_steps=total_steps,
            unique_nodes_visited=unique_visited,
            revisit_count=revisits,
            dead_end_count=dead_ends,
            avg_branching_factor=float(avg_branch),
            convergence_step=convergence_step,
            final_dispersion=float(dispersion)
        )

    def compute_heuristic_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for each heuristic across all decisions.

        Returns:
            Dict mapping heuristic name to stats dict with:
            - mean: Average score
            - std: Standard deviation
            - min: Minimum score
            - max: Maximum score
            - median: Median score
        """
        if not self._decisions:
            return {}

        # Collect all scores by heuristic name
        all_scores: Dict[str, List[float]] = {}
        for decision in self._decisions:
            for name, scores in decision.scores.items():
                if name not in all_scores:
                    all_scores[name] = []
                all_scores[name].extend(scores.tolist())

        # Compute stats
        stats = {}
        for name, scores in all_scores.items():
            if not scores:
                continue
            arr = np.array(scores)
            stats[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr))
            }
        return stats

    def compute_choice_analysis(self) -> Dict[str, Any]:
        """
        Analyze choice patterns - did agents make "good" or "bad" choices?

        Returns:
            Dict with:
            - avg_chosen_rank: Average rank of chosen node among candidates
            - greedy_match_rate: How often agents chose the highest-scored candidate
            - exploration_rate: How often agents chose lower-ranked candidates
        """
        if not self._decisions:
            return {
                "avg_chosen_rank": 0.0,
                "greedy_match_rate": 0.0,
                "exploration_rate": 0.0
            }

        ranks = []
        greedy_matches = 0

        for d in self._decisions:
            if len(d.final_scores) == 0:
                continue

            # Rank of chosen node (0 = best, higher = worse)
            sorted_indices = np.argsort(d.final_scores)[::-1]
            rank = np.where(sorted_indices == d.chosen_index)[0]
            if len(rank) > 0:
                ranks.append(rank[0])
                if rank[0] == 0:
                    greedy_matches += 1

        n_decisions = len(self._decisions)
        return {
            "avg_chosen_rank": float(np.mean(ranks)) if ranks else 0.0,
            "greedy_match_rate": greedy_matches / n_decisions if n_decisions > 0 else 0.0,
            "exploration_rate": 1.0 - (greedy_matches / n_decisions) if n_decisions > 0 else 0.0
        }

    def get_context(
        self,
        agent_trajectories: Optional[List[List[int]]] = None
    ) -> QueryDecisionContext:
        """
        Build complete decision context for LLM consumption.

        Args:
            agent_trajectories: Optional list of agent paths for trajectory analysis

        Returns:
            QueryDecisionContext with all captured decisions and computed metrics
        """
        trajectory_metrics = None
        if agent_trajectories:
            trajectory_metrics = self.compute_trajectory_metrics(agent_trajectories)

        return QueryDecisionContext(
            query_id=self._query_id,
            n_agents=self._n_agents,
            n_steps=self._n_steps,
            decisions=self._decisions.copy(),
            trajectory_metrics=trajectory_metrics,
            heuristic_stats=self.compute_heuristic_stats()
        )

    def clear(self):
        """Reset tracker for reuse."""
        self._decisions = []
        self._query_id = None
        self._n_agents = 0
        self._n_steps = 0

    @property
    def decision_count(self) -> int:
        """Number of decisions captured."""
        return len(self._decisions)

    def to_summary_dict(
        self,
        agent_trajectories: Optional[List[List[int]]] = None
    ) -> Dict[str, Any]:
        """
        Convert tracked data to a summary dict for LLM context.

        This is a convenience method that returns a JSON-serializable dict
        with behavioral metrics suitable for inclusion in LLM prompts.
        """
        ctx = self.get_context(agent_trajectories)

        summary = {
            "n_agents": ctx.n_agents,
            "n_steps": ctx.n_steps,
            "decisions_captured": len(ctx.decisions),
        }

        if ctx.trajectory_metrics:
            tm = ctx.trajectory_metrics
            summary["trajectory"] = {
                "unique_nodes_ratio": tm.unique_nodes_visited / max(tm.total_steps, 1),
                "revisit_rate": tm.revisit_count / max(tm.total_steps, 1),
                "dead_end_rate": tm.dead_end_count / max(tm.total_steps, 1),
                "avg_branching_factor": tm.avg_branching_factor,
                "convergence_step": tm.convergence_step,
                "final_dispersion": tm.final_dispersion,
            }

        if ctx.heuristic_stats:
            summary["heuristic_usage"] = ctx.heuristic_stats

        choice_analysis = self.compute_choice_analysis()
        summary["choice_patterns"] = choice_analysis

        return summary
