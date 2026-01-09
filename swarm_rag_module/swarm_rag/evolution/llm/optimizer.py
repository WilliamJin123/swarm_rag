from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
import random

from ..types.genome import Genome
from ..types.config import EvolutionContext

class LLMOptimizer(ABC):
    """
    Abstract Base Class for LLM-based Genome Optimization.
    The 'Black Box' interface.
    """
    
    @abstractmethod
    def refine_genome(
        self, 
        genome: Genome, 
        evolution_context: EvolutionContext,
        history: List[str] = None
    ) -> Dict[str, Any]: 
        """
        Analyzes a SINGLE genome's performance and returns a modified version
        representation (not the Genome object itself, but the data to update it).
        
        Args:
            genome: The agent to optimize.
            evolution_context: Global context (generation, config).
            history: Optional logs of previous edits.
            
        Returns:
            A Dictionary containing the *edits* (diff) or full specification.
            Format:
            {
                "diagnosis": "Analysis string...",
                "proposed_changes": {
                    "params": { ... },
                    "strategies": { "strategy_name": "expression_string" }
                }
            }
        """
        pass

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
