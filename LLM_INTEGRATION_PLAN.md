# LLM-Guided Smart Evolution Integration Plan

## Objective
To replace the random mutation operators with an **LLM-based Individual Refinement Strategy**. Instead of breeding agents together (crossover), the LLM will act as an intelligent "coach" for each agent. It will analyze a specific agent's performance metrics (e.g., high cost, low recall) and "edit" that agent's parameters and logic to address its specific weaknesses.

## Architecture

### 1. New Component: `LLMEvolutionLoop`
We will introduce a new loop implementation `LLMEvolutionLoop` that adheres to the same contract as the existing `EvolutionLoop`.

- **Responsibility**: Manages the transition from Generation `N` to `N+1`.
- **Logic**: 
    1. Selects survivors/parents based on standard selection (e.g., Elitism).
    2. Instead of random mutation, it iterates through selected genomes.
    3. Calls `LLMOptimizer.refine_genome(genome, context)` for each candidate.
    4. The LLM returns a *modified version* of that specific genome.

### 2. Interface: `LLMOptimizer` (The "Black Box")
An abstract base class (or Protocol) defining the interface for the LLM interaction.

**Crucial Note**: The actual API call to an LLM provider (OpenAI, Anthropic, Gemini) will **NOT** be implemented in this phase. The `refine_genome` method will be implemented with a "Black Box" placeholder or mock that returns structured data consistent with the prompt design. The focus is on the *architecture*, *prompt engineering*, and *parsing* logic.

```python
class LLMOptimizer(ABC):
    @abstractmethod
    def refine_genome(
        self, 
        genome: Genome, 
        evolution_context: EvolutionContext,
        history: List[str] = None # Optional: logs of previous edits
    ) -> GenomeData: 
        """
        Analyzes a SINGLE genome's performance and returns a modified version.
        
        NOTE: This method's internal implementation will currently be a STUB or MOCK.
        It should construct the full prompt, but then skip the network call and 
        return a pre-defined or random valid response for testing the pipeline.
        """
        pass
```

- **Input**: A single `Genome` object with its `fitness` and `metrics` (Recall, Cost, etc.) populated.
- **Output**: A `GenomeData` dictionary representing the *improved* version of that agent.

### 3. Data Serialization (Context Window Optimization)
We focus on the *individual's* narrative.

**Input Format (to LLM):**
```json
{
  "id": "gen_5_agent_12",
  "performance": {
    "quality_score": 0.45,
    "cost_score": 0.9,
    "recall": 0.6,
    "average_steps": 12
  },
  "current_config": {
    "params": { "n_agents": 20, "decay": 0.5 },
    "strategies": {
      "ranking": "(semantic_similarity * 0.8) + (pagerank * 0.2)",
      "g0_movement": "semantic_similarity * pheromone_repulsion"
    }
  }
}
```

**Output Format (from LLM):**
The LLM will be instructed to return a JSON containing the *edits*.

```json
{
  "diagnosis": "The agent has high cost (too many steps) but decent recall. It is exploring too aggressively.",
  "proposed_changes": {
    "params": { "steps": 8 }, 
    "strategies": {
      "g0_movement": "semantic_similarity * (pheromone_repulsion + 0.1)" 
    }
  }
}
```
*Note: The LLM only needs to return what CHANGED, or the full object if easier for parsing.*

## Prompt Engineering

### System Prompt
> You are an expert AI Geneticist. Your job is to optimize individual Retrieval Agents.
> 
> You will be given an agent's **Code** (parameters & logic) and its **Report Card** (metrics).
> - **High Cost**: Reduce `steps`, `n_agents`, or make movement more focused (less random/exploration).
> - **Low Recall**: Increase exploration (`pheromone_repulsion`), add `n_agents`, or relax thresholds.
> 
> You must output a valid JSON object representing the **refined** agent.

### User Instruction
> **Agent ID**: {id}
> **Metrics**: 
> - Quality: {quality} (Target: 1.0)
> - Cost: {cost} (Target: 0.0 - Lower is better)
> - Latency: {latency}ms
>
> **Current Logic**:
> {current_config_json}
>
> **Task**: This agent is underperforming. Analyze the metrics above.
> 1. Diagnosis: Why is the score low?
> 2. Action: Edit the `params` or `strategies` to fix the specific weakness identified in the diagnosis.
> 
> Return JSON format.

## Implementation Steps

### Phase 1: Core Logic (Skeleton)
1.  **Create Directory**: `swarm_rag_module/swarm_rag/evolution/llm/`
2.  **Define Interface**: `optimizer.py` with `BaseLLMOptimizer` having `refine_genome`.
    - **Deliverable**: A class where `refine_genome` constructs the full prompts but returns a Mock response.
3.  **Implement Logic**: `LLMEvolutionLoop` in `loop.py`. It should parallelize LLM calls (using `ThreadPoolExecutor`) since we are processing N agents individually.

### Phase 2: Parsing & Serialization
1.  **Serialization**: `genome_to_json(genome)` helper.
2.  **Expression Parser**: Implement `ExpressionParser` in `parsers.py` to convert string formulas (e.g., `"A * B"`) back into `ExpressionNode` objects. This is critical for interpreting the LLM's strategy edits.

### Phase 3: Integration
1.  **Switch Logic**: Modify `EvolutionEngine` to use `LLMEvolutionLoop` when `use_llm_evolution=True`.
2.  **Mock Optimizer**: Ensure the `refine_genome` implementation is robust enough to handle the Mock data correctly (i.e., it successfully "updates" the genome).

## Approval
Please review this updated plan for **Individual Genome Refinement**. Upon approval, I will begin implementation.
