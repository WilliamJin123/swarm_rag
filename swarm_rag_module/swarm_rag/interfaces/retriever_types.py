"""
Type definitions for the simplified SwarmRetriever API.

This module defines:
- SingleResult / BatchResult: Return types for retrieval
- RetrievalConfig / RunConfig: Configuration TypedDicts
- TraversalState: Internal state dataclass for traversal
- QueryBuilder: Builder pattern for constructing queries
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..core.swarm_retriever import SwarmRetriever


# =============================================================================
# Result Types
# =============================================================================


class SingleResult(NamedTuple):
    """Result for a single query."""

    node_ids: torch.Tensor  # (top_k,)
    scores: torch.Tensor  # (top_k,)


class BatchResult(NamedTuple):
    """Result for a batch of queries."""

    node_ids: torch.Tensor  # (n_queries, top_k)
    scores: torch.Tensor  # (n_queries, top_k)


# =============================================================================
# Config Types
# =============================================================================


class RetrievalConfig(TypedDict, total=False):
    """
    Parameters that affect retrieval behavior (evolvable by genome).

    All parameters are optional - unspecified values use retriever defaults.
    """

    n_agents: int  # Number of agents to deploy
    steps: int  # Number of traversal steps
    decay: float  # Pheromone decay rate per step
    initial_pool_size: int  # Size of initial similarity search
    start_subset: int  # Number of top nodes for agent spawning
    top_k: int  # Number of results to return
    drop_zone_inc: float  # Weight increment for earlier nodes in spawn selection
    movement_strategies: Dict[str, Tuple[Union[str, Callable], float]]
    deposit_strategies: Dict[str, Tuple[Union[str, Callable], float]]
    ranking_strategies: Dict[str, Tuple[Union[str, Callable], float]]


class RunConfig(TypedDict, total=False):
    """
    Execution parameters (not evolvable).

    Controls how retrieval is executed, not what it computes.
    Benchmark results show batch_size=64 provides best throughput:
      - Sequential: ~7 q/s
      - Batch=32: ~26 q/s
      - Batch=64: ~47 q/s (16x speedup over sequential)
    """

    mode: Literal["sequential", "batched"]  # default: "batched"
    batch_size: int  # default: 64 (queries per batch)


# =============================================================================
# Internal State
# =============================================================================


@dataclass
class TraversalState:
    """
    Batched state for all queries in a processing chunk.

    All tensors are on the same device and have consistent batch dimension.

    Memory-optimized design:
    - Compact pheromone hash table (10K) instead of dense buffer (150K)
    - Embedding cache reused across steps to avoid redundant fetches
    """

    query_embeddings: torch.Tensor  # (batch, embed_dim)
    agent_positions: torch.Tensor  # (batch, n_agents) - current node IDs
    visit_history: torch.Tensor  # (batch, n_agents, steps+1) - all visited nodes
    step: int  # current step number
    device: str

    # Compact pheromone hash table (replaces dense pheromones tensor)
    pheromone_keys: torch.Tensor  # (batch, buffer_size) - node IDs, -1 if empty
    pheromone_values: torch.Tensor  # (batch, buffer_size) - pheromone values

    # Embedding cache (reused across steps)
    emb_id_to_idx: torch.Tensor  # (max_node_id+1,) - node_id -> cache_idx, -1 if not cached
    emb_cache: torch.Tensor  # (cache_size, embed_dim) - cached embeddings
    emb_next_idx: int  # Next free cache slot

    # Config
    pheromone_buffer_size: int = 10000
    emb_cache_size: int = 2000

    @property
    def batch_size(self) -> int:
        return self.query_embeddings.shape[0]

    @property
    def n_agents(self) -> int:
        return self.agent_positions.shape[1]


# =============================================================================
# Query Builder
# =============================================================================


class QueryBuilder:
    """
    Builder pattern for constructing retrieval queries.

    Usage:
        # Single query
        result = retriever.query("what is X?").run()

        # Batch queries
        result = retriever.query(["q1", "q2"]).run()

        # With overrides
        result = retriever.query(queries).on("cpu").run(
            config=RetrievalConfig(n_agents=50),
            run=RunConfig(mode="batched", batch_size=32)
        )
    """

    def __init__(
        self,
        retriever: "SwarmRetriever",
        input_data: Union[str, int, torch.Tensor, List],
        pool: Optional[torch.Tensor] = None,
    ):
        """
        Initialize query builder.

        Args:
            retriever: The SwarmRetriever instance
            input_data: Query input - str, int (node ID), Tensor, or List
            pool: Optional precomputed initial pool (skips similarity search)
        """
        self._retriever = retriever
        self._input = input_data
        self._pool = pool
        self._device_override: Optional[str] = None
        self._is_batch = self._detect_batch(input_data)

    def _detect_batch(self, input_data: Any) -> bool:
        """Determine if input represents a batch of queries."""
        if isinstance(input_data, list):
            return True
        if isinstance(input_data, torch.Tensor) and input_data.dim() == 2:
            return True
        return False

    def on(self, device: str) -> "QueryBuilder":
        """
        Override the execution device.

        Args:
            device: Device string ("cuda", "cpu", "mps")

        Returns:
            Self for method chaining
        """
        self._device_override = device
        return self

    def run(
        self,
        config: Optional[RetrievalConfig] = None,
        run: Optional[RunConfig] = None,
    ) -> Union[SingleResult, BatchResult]:
        """
        Execute the query and return results.

        Args:
            config: Retrieval parameters (merged with retriever defaults)
            run: Execution parameters (mode, batch_size)

        Returns:
            SingleResult for single queries, BatchResult for batches
        """
        # Merge configs with defaults
        merged_config = self._retriever._merge_config(config)
        merged_run = self._merge_run_config(run)

        # Resolve device
        device = self._device_override or self._retriever._device

        # Resolve input to embeddings
        embeddings, pools, is_batch = self._retriever._resolve_input(
            self._input, self._pool, device
        )

        # Execute traversal
        node_ids, scores = self._retriever._traverse(
            query_embeddings=embeddings,
            initial_pools=pools,
            config=merged_config,
            run_config=merged_run,
            device=device,
        )

        # Return appropriate result type
        if is_batch:
            return BatchResult(node_ids=node_ids, scores=scores)
        else:
            # Squeeze batch dimension for single queries
            return SingleResult(
                node_ids=node_ids.squeeze(0), scores=scores.squeeze(0)
            )

    def _merge_run_config(self, run: Optional[RunConfig]) -> RunConfig:
        """Merge run config with defaults."""
        defaults: RunConfig = {"mode": "batched", "batch_size": 64}
        if run is None:
            return defaults
        return {**defaults, **run}
