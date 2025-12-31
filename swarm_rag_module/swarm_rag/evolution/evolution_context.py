from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .genome import Genome

@dataclass
class EvolutionContext:
    """
    Shared context passed to all genetic operators (Selection, Crossover, Mutation).
    Replaces the messy **kwargs passing.
    """
    # Current State
    population: List[Genome]
    generation: int
    
    # Global Config (Hyperparams like mutation_rate, tournament_k, etc.)
    config: Dict[str, Any]
    
    # Registry Data (What features can we mutate into?)
    available_features: List[str]
    
    # Optional: Shared memory for island migration or history
    global_memory: Dict[str, Any] = field(default_factory=dict)

    def get_param(self, key: str, default: Any = None) -> Any:
        """Helper to safely get a config param."""
        return self.config.get(key, default)