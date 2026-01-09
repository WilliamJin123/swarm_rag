# Key Usage & LLM Integration Plan

## Overview
This document outlines the plan to integrate real LLM providers into the evolutionary loop using the `keycycle` package. This will replace the `MockLLMOptimizer` with a concrete implementation that calls LLM APIs (Cerebras, Groq, OpenAI).

## Key Management (`keycycle`)

The `keycycle` package provides a `MultiProviderWrapper` to manage multiple API keys and providers efficiently.

### Requirements
- **Package**: `keycycle` (Available in the environment).
- **Environment File**: `.env` containing API keys. figure out how keycycle deals with api keys (if necessary), 



## `RealLLMOptimizer` Implementation

We will implement `RealLLMOptimizer` inheriting from `LLMOptimizer`.

### Class Structure

```python
class RealLLMOptimizer(LLMOptimizer):
    def __init__(self, model_name: str = "something"):
        self.client = get_llm_client()
        self.model = model_name # Default to a fast/strong model available on Groq/Cerebras

    def refine_genome(
        self, 
        genome: Genome, 
        evolution_context: EvolutionContext,
        history: List[str] = None
    ) -> Dict[str, Any]:
        
        # 1. Construct Prompt
        # Convert genome/context to JSON
        # Build system and user prompts (as defined in LLM_INTEGRATION_PLAN.md)
        
        # 2. Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                response_format={"type": "json_object"} # If supported by provider
            )
            
            content = response.choices[0].message.content
            
            # 3. Parse Response
            data = json.loads(content)
            
            # 4. Validate & Return
            # Ensure "diagnosis" and "proposed_changes" keys exist
            return data
            
        except Exception as e:
            # Handle API errors (retries, logging)
            print(f"LLM Optimization failed: {e}")
            # Fallback: return no changes or raise
            return {"diagnosis": f"Error: {e}", "proposed_changes": {}}
```

## Provider Strategy

- **Primary**: **Cerebras** and **Groq** due to high inference speed, which is crucial for evolutionary loops where we might optimize many agents per generation.

## Next Steps

1.  **Verify `keycycle`**: Package is confirmed in `.venv/Lib/site-packages`.
2.  **Implement `RealLLMOptimizer`**: Create the class in `swarm_rag_module/swarm_rag/evolution/llm/optimizer.py` (or a new file `real_optimizer.py`).
3.  **Update `EvolutionEngine`**: Allow configuring the optimizer type (Mock vs Real) and passing model names.
4.  **Testing**: Run with a small population and limited generations to verify API usage and JSON parsing robustness.