"""
Decision tracking for agent-level behavioral analysis.

Captures detailed decision data from agent steps for LLM-guided mutations.
Provides trajectory analysis and heuristic score statistics that help the
LLM understand WHY agents are performing a certain way.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
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
class TraversalPath:
    """Single agent traversal path with stuck/revisit info."""
    agent_id: int
    path: List[int]
    revisit_nodes: List[int]  # nodes visited more than once
    stuck_at: Optional[int]  # node where agent got stuck (dead-end or loop)


@dataclass
class StuckNodeSummary:
    """Summary of problematic nodes across all agents."""
    dead_end_nodes: List[tuple]  # [(node_id, count), ...] - nodes where agents couldn't move
    revisit_hotspots: List[tuple]  # [(node_id, count), ...] - most revisited nodes


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

    Supports two sampling modes:
    - "uniform": Captures decisions with uniform probability (sample_rate)
    - "priority": Prioritizes edge cases (dead ends, revisits, low confidence)

    Usage:
        tracker = DecisionTracker(sampling_mode="priority")
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
        summary = tracker.to_summary_dict(agent_trajectories)
    """

    def __init__(
        self,
        enabled: bool = True,
        sample_rate: float = 1.0,
        sampling_mode: Literal["uniform", "priority"] = "uniform",
        priority_sample_rate: float = 0.5,
        max_decisions: int = 500
    ):
        """
        Args:
            enabled: Whether tracking is active
            sample_rate: Fraction of decisions to capture in uniform mode (0.0-1.0)
            sampling_mode: "uniform" for random sampling, "priority" for edge-case focus
            priority_sample_rate: Sample rate for edge cases in priority mode
            max_decisions: Maximum decisions to capture (prevents memory bloat)
        """
        self.enabled = enabled
        self.sample_rate = sample_rate
        self.sampling_mode = sampling_mode
        self.priority_sample_rate = priority_sample_rate
        self.max_decisions = max_decisions

        self._decisions: List[AgentDecision] = []
        self._query_id: Any = None
        self._n_agents: int = 0
        self._n_steps: int = 0
        self._rng = np.random.default_rng()
        self._visited_nodes: Dict[int, set] = {}  # For priority mode

    def start_query(self, query_id: Any, n_agents: int, n_steps: int):
        """Begin tracking for a new query."""
        self._decisions = []
        self._query_id = query_id
        self._n_agents = n_agents
        self._n_steps = n_steps
        self._visited_nodes = {i: set() for i in range(n_agents)}

    def _should_capture_uniform(self) -> bool:
        """Check if decision should be captured using uniform sampling."""
        if not self.enabled:
            return False
        if len(self._decisions) >= self.max_decisions:
            return False
        if self.sample_rate >= 1.0:
            return True
        return self._rng.random() < self.sample_rate

    def _should_capture_priority(
        self,
        agent_id: int,
        step: int,
        current_node: int,
        chosen_node: int,
        final_scores: np.ndarray
    ) -> bool:
        """
        Determine if decision should be captured using priority sampling.

        Prioritizes edge cases that are most informative:
        - Dead ends: chosen_node == current_node (couldn't move)
        - Revisits: chosen_node already visited by this agent
        - Low confidence: top 2 scores are within 10% of each other
        - Boundary steps: first or last step
        """
        if not self.enabled:
            return False
        if len(self._decisions) >= self.max_decisions:
            return False

        is_edge_case = False

        # 1. Dead end detection
        if chosen_node == current_node:
            is_edge_case = True

        # 2. Revisit detection
        if agent_id in self._visited_nodes:
            if chosen_node in self._visited_nodes[agent_id]:
                is_edge_case = True
            self._visited_nodes[agent_id].add(chosen_node)

        # 3. Low confidence detection
        if len(final_scores) >= 2:
            sorted_scores = np.sort(final_scores)[::-1]
            if sorted_scores[0] > 0:
                score_ratio = sorted_scores[1] / (sorted_scores[0] + 1e-10)
                if score_ratio > 0.9:
                    is_edge_case = True

        # 4. Boundary step detection
        if step == 0 or step == self._n_steps - 1:
            is_edge_case = True

        # Apply appropriate sample rate
        if is_edge_case:
            return self._rng.random() < self.priority_sample_rate
        else:
            return self._rng.random() < self.sample_rate

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
            final_scores: Combined weighted scores used for selection
            chosen_node: The node ID that was selected
            chosen_index: Index of chosen_node in candidates list
            deposit: Amount of pheromone deposited at chosen_node
        """
        # Determine if we should capture based on sampling mode
        if self.sampling_mode == "priority":
            should_capture = self._should_capture_priority(
                agent_id, step, current_node, chosen_node, final_scores
            )
        else:
            should_capture = self._should_capture_uniform()

        if not should_capture:
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
        """Analyze completed trajectories to compute behavioral metrics."""
        if not agent_trajectories:
            return TrajectoryMetrics(
                total_steps=0, unique_nodes_visited=0, revisit_count=0,
                dead_end_count=0, avg_branching_factor=0.0,
                convergence_step=None, final_dispersion=0.0
            )

        total_steps = sum(max(0, len(t) - 1) for t in agent_trajectories)
        all_visited = [node for traj in agent_trajectories for node in traj]
        unique_visited = len(set(all_visited))

        revisits = 0
        for traj in agent_trajectories:
            seen = set()
            for node in traj:
                if node in seen:
                    revisits += 1
                seen.add(node)

        dead_ends = sum(1 for d in self._decisions if d.chosen_node == d.current_node)

        avg_branch = np.mean([len(d.candidates) for d in self._decisions]) if self._decisions else 0.0

        # Convergence detection
        convergence_step = None
        n_agents = len(agent_trajectories)
        max_steps = max(len(t) for t in agent_trajectories) if agent_trajectories else 0

        for step in range(max_steps):
            step_positions = [traj[min(step, len(traj) - 1)] for traj in agent_trajectories]
            position_counts = Counter(step_positions)
            most_common_count = position_counts.most_common(1)[0][1]
            if most_common_count >= n_agents * 0.5:
                convergence_step = step
                break

        # Final dispersion
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
        """Compute statistics for each heuristic across all decisions."""
        if not self._decisions:
            return {}

        all_scores: Dict[str, List[float]] = {}
        for decision in self._decisions:
            for name, scores in decision.scores.items():
                if name not in all_scores:
                    all_scores[name] = []
                all_scores[name].extend(scores.tolist())

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
        """Analyze choice patterns - how agents chose candidates."""
        if not self._decisions:
            return {"avg_chosen_rank": 0.0, "greedy_match_rate": 0.0, "exploration_rate": 0.0}

        ranks = []
        greedy_matches = 0

        for d in self._decisions:
            if len(d.final_scores) == 0:
                continue
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

    def get_context(self, agent_trajectories: Optional[List[List[int]]] = None) -> QueryDecisionContext:
        """Build complete decision context for LLM consumption."""
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
        self._visited_nodes = {}

    @property
    def decision_count(self) -> int:
        """Number of decisions captured."""
        return len(self._decisions)

    def compute_traversal_paths(self, agent_trajectories: List[List[int]]) -> List[TraversalPath]:
        """Extract per-agent traversal paths with revisit and stuck info."""
        paths = []
        for agent_id, traj in enumerate(agent_trajectories):
            if not traj:
                continue

            seen = set()
            revisit_nodes = []
            for node in traj:
                if node in seen:
                    revisit_nodes.append(node)
                seen.add(node)

            stuck_at = None
            if len(traj) >= 2:
                if traj[-1] == traj[-2]:
                    stuck_at = traj[-1]
                elif len(traj) >= 4 and traj[-1] == traj[-3] and traj[-2] == traj[-4]:
                    stuck_at = traj[-1]

            paths.append(TraversalPath(
                agent_id=agent_id, path=traj,
                revisit_nodes=list(set(revisit_nodes)), stuck_at=stuck_at
            ))
        return paths

    def compute_node_visitation_stats(self, agent_trajectories: List[List[int]]) -> Dict[int, Dict[str, Any]]:
        """Compute node-level visitation statistics."""
        node_stats: Dict[int, Dict[str, Any]] = {}

        for agent_id, traj in enumerate(agent_trajectories):
            for i, node in enumerate(traj):
                if node not in node_stats:
                    node_stats[node] = {
                        "visits": 0, "unique_agents": set(),
                        "is_dead_end": False, "is_terminal": 0
                    }
                node_stats[node]["visits"] += 1
                node_stats[node]["unique_agents"].add(agent_id)
                if i == len(traj) - 1:
                    node_stats[node]["is_terminal"] += 1
                if i > 0 and traj[i] == traj[i-1]:
                    node_stats[node]["is_dead_end"] = True

        for node in node_stats:
            node_stats[node]["unique_agents"] = len(node_stats[node]["unique_agents"])
        return node_stats

    def identify_stuck_nodes(self, agent_trajectories: List[List[int]]) -> StuckNodeSummary:
        """Identify nodes where agents frequently get stuck."""
        dead_end_counts: Counter = Counter()
        revisit_counts: Counter = Counter()

        for traj in agent_trajectories:
            if not traj:
                continue
            for i in range(1, len(traj)):
                if traj[i] == traj[i-1]:
                    dead_end_counts[traj[i]] += 1
            seen = set()
            for node in traj:
                if node in seen:
                    revisit_counts[node] += 1
                seen.add(node)

        return StuckNodeSummary(
            dead_end_nodes=dead_end_counts.most_common(5),
            revisit_hotspots=revisit_counts.most_common(5)
        )

    def sample_representative_paths(self, agent_trajectories: List[List[int]], max_samples: int = 5) -> List[TraversalPath]:
        """Sample diverse, representative paths for LLM context."""
        all_paths = self.compute_traversal_paths(agent_trajectories)
        if not all_paths:
            return []

        stuck_paths = [p for p in all_paths if p.stuck_at is not None]
        revisit_paths = [p for p in all_paths if p.revisit_nodes and p.stuck_at is None]
        clean_paths = [p for p in all_paths if not p.revisit_nodes and p.stuck_at is None]

        samples = []
        for p in stuck_paths[:max(1, max_samples // 2)]:
            samples.append(p)

        revisit_paths.sort(key=lambda p: len(p.revisit_nodes), reverse=True)
        for p in revisit_paths[:max(1, (max_samples - len(samples)) // 2)]:
            if len(samples) < max_samples:
                samples.append(p)

        for p in clean_paths:
            if len(samples) >= max_samples:
                break
            samples.append(p)

        return samples[:max_samples]

    def reconstruct_trajectories_from_decisions(self) -> List[List[int]]:
        """Reconstruct agent trajectories from recorded decisions."""
        if not self._decisions:
            return []

        agent_decisions: Dict[int, List[AgentDecision]] = {}
        for d in self._decisions:
            if d.agent_id not in agent_decisions:
                agent_decisions[d.agent_id] = []
            agent_decisions[d.agent_id].append(d)

        trajectories = []
        for agent_id in sorted(agent_decisions.keys()):
            decisions = sorted(agent_decisions[agent_id], key=lambda x: x.step)
            if not decisions:
                continue
            path = [decisions[0].current_node]
            for d in decisions:
                path.append(d.chosen_node)
            trajectories.append(path)
        return trajectories

    def to_summary_dict(self, agent_trajectories: Optional[List[List[int]]] = None) -> Dict[str, Any]:
        """Convert tracked data to a summary dict for LLM context."""
        if agent_trajectories is None and self._decisions:
            agent_trajectories = self.reconstruct_trajectories_from_decisions()

        ctx = self.get_context(agent_trajectories)

        summary = {
            "n_agents": ctx.n_agents,
            "n_steps": ctx.n_steps,
            "decisions_captured": len(ctx.decisions),
            "sampling_mode": self.sampling_mode,
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

        summary["choice_patterns"] = self.compute_choice_analysis()

        if agent_trajectories:
            sample_paths = self.sample_representative_paths(agent_trajectories, max_samples=5)
            summary["sample_paths"] = [
                {"agent_id": p.agent_id, "path": p.path, "revisit_nodes": p.revisit_nodes, "stuck_at": p.stuck_at}
                for p in sample_paths
            ]

            node_stats = self.compute_node_visitation_stats(agent_trajectories)
            sorted_nodes = sorted(node_stats.items(), key=lambda x: x[1]["visits"], reverse=True)[:10]
            summary["node_hotspots"] = [
                {"node_id": node_id, "visits": stats["visits"],
                 "unique_agents": stats["unique_agents"], "is_dead_end": stats["is_dead_end"]}
                for node_id, stats in sorted_nodes
            ]

            stuck_summary = self.identify_stuck_nodes(agent_trajectories)
            summary["stuck_nodes"] = {
                "dead_ends": [list(item) for item in stuck_summary.dead_end_nodes],
                "revisit_traps": [list(item) for item in stuck_summary.revisit_hotspots]
            }

        # Edge case summary for priority mode
        if self.sampling_mode == "priority":
            summary["edge_case_summary"] = self._get_edge_case_summary()

        return summary

    def _get_edge_case_summary(self) -> Dict[str, Any]:
        """Get a summary focused on edge cases."""
        if not self._decisions:
            return {}

        dead_ends = sum(1 for d in self._decisions if d.chosen_node == d.current_node)
        boundary_decisions = sum(1 for d in self._decisions if d.step == 0 or d.step == self._n_steps - 1)
        low_confidence = 0

        for d in self._decisions:
            if len(d.final_scores) >= 2:
                sorted_scores = np.sort(d.final_scores)[::-1]
                if sorted_scores[0] > 0 and sorted_scores[1] / (sorted_scores[0] + 1e-10) > 0.9:
                    low_confidence += 1

        total = len(self._decisions)
        return {
            "total_captured": total,
            "dead_end_rate": dead_ends / max(1, total),
            "low_confidence_rate": low_confidence / max(1, total),
            "boundary_decisions": boundary_decisions,
            "issues_detected": {
                "dead_ends": dead_ends > total * 0.1,
                "uncertain_decisions": low_confidence > total * 0.3,
            }
        }


# Backwards compatibility alias
SmartDecisionTracker = DecisionTracker
