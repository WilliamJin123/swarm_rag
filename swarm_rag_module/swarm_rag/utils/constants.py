"""
Shared Constants for SwarmRAG

This module provides shared constants that are used across multiple modules.
By centralizing these definitions, we avoid circular imports and ensure
consistency between evolution, core retriever, and other components.
"""
from typing import TypedDict


class SwarmParams(TypedDict):
    """
    Defines the contract for Swarm Hyperparameters.
    Acts as a Dictionary at runtime, but provides IDE autocompletion.
    """
    n_agents: int
    steps: int
    drop_zone_inc: float
    decay: float
    initial_pool_size: int
    start_subset: int


# Single source of truth for default swarm hyperparameters
# Used by both SwarmRetriever and evolution Genome system
DEFAULT_SWARM_PARAMS: SwarmParams = {
    "n_agents": 20,
    "steps": 4,
    "decay": 0.5,
    "drop_zone_inc": 0.05,
    "initial_pool_size": 30,
    "start_subset": 10,
}


__all__ = [
    "SwarmParams",
    "DEFAULT_SWARM_PARAMS",
]
