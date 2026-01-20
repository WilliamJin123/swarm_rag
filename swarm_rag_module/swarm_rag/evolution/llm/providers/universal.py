"""
Universal LLM Provider - provider-agnostic implementation.

Uses keycycle's MultiProviderWrapper to support any LLM provider
(cerebras, openai, groq, anthropic, together, etc.) through a unified API.

Enhanced with behavioral context for smarter mutations.

Note: The `wrapper` and `model` attributes are public and used by the
three-tier LLM architecture (Strategic Oracle, Tactical Advisor).
"""
import json
import logging
from typing import Optional, Any

from ..provider import BaseLLMProvider, LLMResponse
from ..utils import genome_to_json_context
from ...types.genome import Genome
from ...types.config import EvolutionContext

logger = logging.getLogger(__name__)


class UniversalLLMProvider(BaseLLMProvider):
    """
    Provider-agnostic LLM implementation using keycycle's MultiProviderWrapper.

    Supports any provider that keycycle supports (cerebras, openai, groq,
    anthropic, together, etc.) through the same interface.

    Enhanced with behavioral analysis from decision tracking for smarter mutations.

    Args:
        provider: LLM provider name (e.g., "cerebras", "openai", "groq")
        model: Model ID for the provider (e.g., "zai-glm-4.7", "gpt-4o-mini")
        env_path: Path to .env file containing API keys
        **kwargs: Additional arguments for BaseLLMProvider (max_retries,
                  retry_delay, circuit_threshold)
    """

    def __init__(
        self,
        provider: str = "cerebras",
        model: str = "zai-glm-4.7",
        env_path: str = ".env",
        **kwargs
    ):
        super().__init__(**kwargs)

        try:
            from keycycle import MultiProviderWrapper
        except ImportError:
            raise ImportError(
                "keycycle is required for UniversalLLMProvider. "
                "Install it with: pip install keycycle"
            )

        from dotenv import load_dotenv
        load_dotenv(env_path)

        self.provider = provider
        self.model = model
        self.wrapper = MultiProviderWrapper.from_env(
            provider=provider,
            default_model_id=model,
            env_file=env_path
        )
        logger.info(f"Initialized UniversalLLMProvider: {provider}/{model}")

    def _call_llm(
        self,
        genome: Genome,
        context: EvolutionContext,
    ) -> LLMResponse:
        """
        Call the configured LLM provider to refine genome.

        Args:
            genome: Genome to refine
            context: Evolution context

        Returns:
            LLMResponse with diagnosis and proposed changes
        """
        # Get decision context from genome if available (set by PopulationEvaluator)
        decision_context = getattr(genome, 'decision_context', None)

        # Build enhanced context with behavioral data
        context_data = genome_to_json_context(
            genome,
            decision_context=decision_context,
            evolution_context=context
        )

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(context_data)

        client = self.wrapper.get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        diagnosis = data.get("diagnosis", "No diagnosis provided")
        proposed_changes = data.get("proposed_changes", {})

        return LLMResponse(
            diagnosis=diagnosis,
            proposed_changes=proposed_changes,
            raw_response=content,
            success=True
        )

    def _build_system_prompt(self) -> str:
        """Build enhanced system prompt for genome refinement."""
        return """You are an expert AI Geneticist optimizing Swarm-based Retrieval Agents.

## Your Role
Analyze agent performance and behavior to propose targeted improvements. You receive:
1. **Performance Metrics** - Quality, Cost, Recall, MRR, Latency
2. **Behavioral Analysis** - How agents explore the graph (revisits, dead-ends, convergence, dispersion)
3. **Heuristic Usage** - Which decision factors are influencing agent choices
4. **Configuration** - Current parameters and strategy expressions

## Diagnosis Guidelines

### High Cost + Low Recall
Problem: Agents explore too aggressively without finding relevant nodes
Solutions:
- Increase `semantic_similarity` weight in movement strategy
- Decrease `pheromone_repulsion` weight (less forced exploration)
- Reduce `n_agents` or `steps` to cut cost
- Increase `decay` to make pheromone trails fade faster

### High Revisit Rate (>15%)
Problem: Agents getting stuck in loops, revisiting same nodes
Solutions:
- Increase `pheromone_repulsion` weight significantly
- Add `random_jitter` to movement strategy for randomness
- Decrease `decay` to make pheromone trails persist longer (stronger avoidance)

### Early Convergence (convergence_step < 2)
Problem: All agents clustering on same nodes too quickly
Solutions:
- Increase `pheromone_repulsion` weight
- Add `random_jitter` for diversity
- Increase `n_agents` for broader coverage
- Decrease `semantic_similarity` to reduce herding

### High Dead-End Rate (>10%)
Problem: Agents reaching graph boundaries or isolated nodes
Solutions:
- Increase `initial_pool_size` for better starting points
- Increase `node_centrality` weight to prefer connected nodes
- Increase `start_subset` to diversify agent spawning

### Low Dispersion (<0.3)
Problem: Agents end up clustered, not covering enough ground
Solutions:
- Increase `pheromone_repulsion` significantly
- Decrease `semantic_similarity` to reduce convergence
- Increase `n_agents` and `steps`

### Greedy Match Rate > 80%
Problem: Agents always choosing highest-scored candidate (no exploration)
Solutions:
- Add `random_jitter` to movement
- Increase `pheromone_repulsion`
- This may be fine if quality is high

## Path-Based Diagnosis (use sample_paths and stuck_nodes data)

### Agent Stuck at Same Node (stuck_at != null)
Problem: Agent reached a dead-end or got trapped in a loop
Solutions:
- Increase `node_centrality` weight to prefer hub nodes with more neighbors
- Increase `pheromone_repulsion` to force exploration away from visited nodes
- Increase `initial_pool_size` to start from better-connected nodes

### Oscillation Paths (e.g., A->B->A->B pattern in path)
Problem: Agent bouncing between two nodes without progress
Solutions:
- Increase `pheromone_repulsion` significantly (0.3+)
- Add `random_jitter` to break symmetry
- Decrease `decay` to make pheromone trails persist longer

### Dead-End Traps (high counts in stuck_nodes.dead_ends)
Problem: Many agents getting stuck at specific nodes with no neighbors
Solutions:
- Increase `initial_pool_size` to diversify starting points
- Increase `node_centrality` to prefer well-connected hub nodes
- Reduce `steps` if agents are walking into graph boundaries

### Revisit Traps (high counts in stuck_nodes.revisit_traps)
Problem: Specific nodes are being revisited excessively across agents
Solutions:
- Increase `pheromone_repulsion` weight significantly
- Decrease `decay` so pheromone trails persist longer
- Check if these nodes are semantically attractive but non-productive

### Node Hotspots (is_dead_end=True in node_hotspots)
Problem: High-traffic nodes that are actually dead-ends
Solutions:
- These nodes might be attracting agents due to high semantic similarity
- Increase `node_centrality` to counterbalance semantic attraction
- Consider if `initial_pool_size` is too small (agents all starting near same node)

## Available Heuristics (use exact names in strategies)

Movement Heuristics:
- `semantic_similarity`: Cosine similarity to query (exploitation)
- `node_centrality`: Prefer highly-connected hub nodes
- `pheromone_repulsion`: Avoid recently-visited nodes (exploration)
- `random_jitter`: Pure random exploration

Deposit Heuristics:
- `flat`: Uniform pheromone deposit
- `hub`: More pheromone on central nodes
- `semantic`: Deposit proportional to similarity
- `exploration_bonus`: Extra deposit on fresh nodes
- `collaborative_amplification`: "Rich get richer" effect

Ranking Heuristics:
- `percentage_visited`: Weight by how many agents visited
- `semantic_rank`: Final semantic similarity score

## Output Format

Return a JSON object with exactly these keys:
{
    "diagnosis": "Brief analysis explaining the root cause of poor performance",
    "proposed_changes": {
        "params": {
            // Only include params you want to change
            "n_agents": 15,
            "steps": 6
        },
        "strategies": {
            // Only include strategies you want to change
            // Use format: "heuristic_name * weight + heuristic_name * weight"
            "g0_movement": "semantic_similarity * 0.5 + node_centrality * 0.3 + pheromone_repulsion * 0.2"
        }
    }
}

## Critical Rules
1. Only propose changes that directly address the diagnosed issue
2. Respect parameter bounds provided in the context
3. Keep strategy expressions simple (2-4 heuristics max)
4. Don't over-correct - make incremental changes
5. If behavioral data shows a specific problem, address THAT problem specifically"""

    def _build_user_prompt(self, context_data: dict) -> str:
        """Build user prompt with full context."""
        # Performance metrics
        perf = context_data['performance']
        metrics_str = f"""- Quality Score: {perf['quality_score']:.4f} (Target: 1.0)
- Cost Score: {perf['cost_score']:.4f} (Target: 0.0 - Lower is better)
- Recall@20: {perf['recall_at_20']:.4f}
- Hit@1: {perf['hit_at_1']:.4f}
- Hit@5: {perf.get('hit_at_5', 0.0):.4f}
- MRR: {perf.get('mrr', 0.0):.4f}
- Latency: {perf['latency']:.4f}s
- Complexity: {perf['complexity']}"""

        # Behavioral analysis (if available)
        behavioral_str = ""
        if context_data.get('behavioral'):
            b = context_data['behavioral']
            behavioral_str = f"""

**Behavioral Analysis** (from agent decision tracking):
- Unique Nodes Ratio: {b.get('unique_nodes_ratio', 0):.2%} (higher = better coverage)
- Revisit Rate: {b.get('revisit_rate', 0):.2%} (lower = less wasted steps)
- Dead-End Rate: {b.get('dead_end_rate', 0):.2%} (lower = better navigation)
- Avg Branching Factor: {b.get('avg_branching_factor', 0):.1f} (options per decision)
- Convergence Step: {b.get('convergence_step', 'Never')} (earlier = faster but riskier)
- Final Dispersion: {b.get('final_dispersion', 0):.2%} (higher = better spread)"""

            # Heuristic usage stats
            if b.get('heuristic_usage'):
                behavioral_str += "\n\n**Heuristic Score Distributions**:"
                for name, stats in b['heuristic_usage'].items():
                    behavioral_str += f"\n  - {name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}"

            # Choice patterns
            if b.get('choice_patterns'):
                cp = b['choice_patterns']
                behavioral_str += f"""

**Choice Patterns**:
- Greedy Match Rate: {cp.get('greedy_match_rate', 0):.1%} (how often agents pick top candidate)
- Avg Chosen Rank: {cp.get('avg_chosen_rank', 0):.2f} (0 = always best, higher = more exploration)
- Exploration Rate: {cp.get('exploration_rate', 0):.1%}"""

            # Sample agent paths (enhanced traversal context)
            if b.get('sample_paths'):
                behavioral_str += "\n\n**Sample Agent Paths**:"
                for p in b['sample_paths'][:5]:  # Limit to 5 paths
                    path_str = " -> ".join(str(n) for n in p.get('path', [])[:8])  # Truncate long paths
                    if len(p.get('path', [])) > 8:
                        path_str += " ..."
                    status = ""
                    if p.get('stuck_at') is not None:
                        status = f" [STUCK at {p['stuck_at']}]"
                    elif p.get('revisit_nodes'):
                        status = f" [Revisited: {', '.join(str(n) for n in p['revisit_nodes'][:3])}]"
                    behavioral_str += f"\n  Agent {p.get('agent_id', '?')}: {path_str}{status}"

            # Node hotspots
            if b.get('node_hotspots'):
                behavioral_str += "\n\n**Node Hotspots** (most visited):"
                for h in b['node_hotspots'][:5]:  # Top 5
                    flags = []
                    if h.get('is_dead_end'):
                        flags.append("DEAD-END")
                    if h.get('visits', 0) > h.get('unique_agents', 0) * 1.5:
                        flags.append("HIGH-REVISIT")
                    flag_str = f" [{', '.join(flags)}]" if flags else ""
                    behavioral_str += f"\n  Node {h.get('node_id', '?')}: {h.get('visits', 0)} visits, {h.get('unique_agents', 0)} agents{flag_str}"

            # Problem nodes summary
            if b.get('stuck_nodes'):
                sn = b['stuck_nodes']
                if sn.get('dead_ends') or sn.get('revisit_traps'):
                    behavioral_str += "\n\n**PROBLEM NODES**:"
                    if sn.get('dead_ends'):
                        behavioral_str += f"\n  Dead-end traps: {sn['dead_ends']}"
                    if sn.get('revisit_traps'):
                        behavioral_str += f"\n  Revisit loops: {sn['revisit_traps']}"

        # Evolutionary context (if available)
        evo_str = ""
        if context_data.get('evolutionary'):
            e = context_data['evolutionary']
            evo_str = f"""

**Evolutionary Context**:
- Generation: {e.get('generation', 'N/A')}
- Population Size: {e.get('population_size', 'N/A')}
- Mutation Rate: {e.get('mutation_rate', 'N/A')}"""

        # Parameter bounds
        bounds_str = ""
        if context_data.get('parameter_bounds'):
            pb = context_data['parameter_bounds']
            bounds_str = f"""

**Parameter Bounds** (respect these limits):
- n_agents: {pb.get('n_agents', (5, 30))}
- steps: {pb.get('steps', (4, 12))}
- decay: {pb.get('decay', (0.85, 0.99))}
- initial_pool_size: {pb.get('initial_pool_size', (10, 50))}"""

        # Available heuristics
        heuristics_str = ""
        if context_data.get('available_heuristics'):
            ah = context_data['available_heuristics']
            heuristics_str = f"""

**Available Heuristics**:
- Movement: {', '.join(ah.get('movement', []))}
- Deposit: {', '.join(ah.get('deposit', []))}
- Ranking: {', '.join(ah.get('ranking', []))}"""

        return f"""**Agent ID**: {context_data['id']}

**Performance Metrics**:
{metrics_str}
{behavioral_str}
{evo_str}

**Current Configuration**:
{json.dumps(context_data['current_config'], indent=2)}
{bounds_str}
{heuristics_str}

**Task**: This agent is underperforming. Based on the metrics and behavioral analysis above:
1. Diagnose the root cause of poor performance
2. Propose specific, targeted changes to fix the identified issue

Return JSON with 'diagnosis' and 'proposed_changes' keys."""
