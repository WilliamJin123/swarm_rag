from typing import List, Dict, Any, Optional
import json
import random
import logging

try:
    from keycycle import MultiProviderWrapper
except ImportError:
    MultiProviderWrapper = None

from ..types.genome import Genome
from ..types.config import EvolutionContext
from .utils import genome_to_json_context

logger = logging.getLogger(__name__)


class LLMOptimizer:
    def __init__(
        self, 
        env_path: str,
        provider: str = "cerebras",
        model: str = "llama-3.3-70b-versatile",
    ):
        if MultiProviderWrapper is None:
            raise ImportError("keycycle is required for LLMOptimizer")
        
        # TO FIX ENV LOADING ISSUES
        import os
        from dotenv import load_dotenv
        load_dotenv()

        self.model = model
        self.provider = provider

        self.wrapper = MultiProviderWrapper.from_env(
            provider=self.provider,
            default_model_id=self.model,
            env_file=env_path
        )

    def refine_genome(
        self, 
        genome: Genome, 
        evolution_context: EvolutionContext,
        history: List[str] = None
    ) -> Dict[str, Any]:
        
        # 1. Construct Prompt
        context_data = genome_to_json_context(genome)
        
        system_prompt = (
            "You are an expert AI Geneticist. Your job is to optimize individual Retrieval Agents.\n"
            "You will be given an agent's **Code** (parameters & logic) and its **Report Card** (metrics).\n"
            "- **High Cost**: Reduce `steps`, `n_agents`, or make movement more focused (less random/exploration).\n"
            "- **Low Recall**: Increase exploration (`pheromone_repulsion`), add `n_agents`, or relax thresholds.\n"
            "You must output a valid JSON object representing the **refined** agent.\n"
            "The JSON must have keys: 'diagnosis' (string) and 'proposed_changes' (nested object with 'params' and 'strategies')."
        )
        
        metrics_str = (
            f"- Quality: {context_data['performance']['quality_score']:.4f} (Target: 1.0)\n"
            f"- Cost: {context_data['performance']['cost_score']:.4f} (Target: 0.0 - Lower is better)\n"
            f"- Latency: {context_data['performance']['latency']:.4f}s"
        )
        
        user_prompt = (
            f"**Agent ID**: {context_data['id']}\n"
            f"**Metrics**:\n{metrics_str}\n\n"
            f"**Current Logic**:\n{json.dumps(context_data['current_config'], indent=2)}\n\n"
            "**Task**: This agent is underperforming. Analyze the metrics above.\n"
            "1. Diagnosis: Why is the score low?\n"
            "2. Action: Edit the `params` or `strategies` to fix the specific weakness identified in the diagnosis.\n"
            "Return JSON format."
        )
        
        # 2. Call LLM
        try:
            # Get client from wrapper (handles rotation if supported/implemented in wrapper)
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
            
            # 3. Parse Response
            data = json.loads(content)
            
            # 4. Validate & Return
            if "diagnosis" not in data:
                data["diagnosis"] = "No diagnosis provided."
            if "proposed_changes" not in data:
                data["proposed_changes"] = {}
                
            return data
            
        except Exception as e:
            logger.error(f"LLM Optimization failed for {genome.id}: {e}")
            return {"diagnosis": f"Error: {e}", "proposed_changes": {}}

class MockLLMOptimizer(LLMOptimizer):
    """
    A Mock implementation that simulates the LLM's behavior without making network calls.
    Useful for testing the pipeline and loop logic.
    """
    
    def refine_genome(
        self, 
        genome: Genome, 
        evolution_context: EvolutionContext,
        history: List[str] = None
    ) -> Dict[str, Any]:
        """
        Returns a dummy response that slightly tweaks parameters.
        """
        
        # Simulate diagnosis
        diagnosis = (
            f"Agent {genome.id} has quality {genome.fitness.quality_score:.2f}. "
            "Simulated diagnosis: modifying parameters to improve exploration."
        )
        
        # Simulate small random changes to params
        current_params = genome.params.copy()
        new_params = {}
        
        # Randomly tweak 'n_agents'
        if random.random() < 0.5:
            delta = random.choice([-2, 2])
            val = current_params.get("n_agents", 20) + delta
            new_params["n_agents"] = max(5, min(50, val))
            
        # Randomly tweak 'decay'
        if random.random() < 0.5:
            delta = random.uniform(-0.1, 0.1)
            val = current_params.get("decay", 0.5) + delta
            new_params["decay"] = max(0.1, min(0.99, val))

        # Simulate strategy change (returning a string string)
        new_strategies = {}
        # For mock, we won't try to generate complex valid expressions yet 
        # unless we have the parser ready. 
        # Let's assume the Loop handles the parsing, so here we return strings.
        # We will just return None for strategies in the mock for now to avoid parsing errors
        # until the parser is implemented. Or we can return a simple valid string.
        
        if random.random() < 0.3:
            new_strategies["ranking"] = "semantic_similarity * 0.9 + pagerank * 0.1"

        return {
            "diagnosis": diagnosis,
            "proposed_changes": {
                "params": new_params,
                "strategies": new_strategies
            }
        }
