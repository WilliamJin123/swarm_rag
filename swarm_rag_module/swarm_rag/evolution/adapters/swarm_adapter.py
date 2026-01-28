"""
SwarmRetriever Adapter - Translates genome output to SwarmRetriever format.

This adapter provides the translation layer between the evolution system
(which produces generic genome configurations) and the SwarmRetriever
(which has specific parameter requirements).

Benefits:
1. Evolution code doesn't need to know SwarmRetriever internals
2. SwarmRetriever signature changes are isolated to this adapter
3. Testing evolution with mock retrievers becomes easier
4. Validation happens in one place

Migration Note:
    The adapter now supports both old and new SwarmRetriever APIs.
    Set use_new_api=True to use the builder pattern:
        retriever.query(...).run(config=...)
"""
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Strategy naming convention: gN_type (e.g., g0_movement, g1_deposit)
STRATEGY_PREFIX = "g"
VALID_STRATEGY_SUFFIXES = {"movement", "deposit", "ranking"}


class SwarmRetrieverAdapter:
    """
    Translates genome output to SwarmRetriever-specific format.

    This adapter:
    1. Validates compiled genome kwargs
    2. Translates strategy names to SwarmRetriever conventions
    3. Ensures parameter bounds are respected
    4. Provides error messages for invalid configurations

    Usage (old API):
        adapter = SwarmRetrieverAdapter(retriever)
        kwargs = adapter.from_genome(compiled_genome)
        results = adapter.retrieve(query, **kwargs)

    Usage (new API):
        adapter = SwarmRetrieverAdapter(retriever, use_new_api=True)
        results = adapter.retrieve(query, compiled_genome)
        # Internally uses: retriever.query(query).run(config=...)
    """

    def __init__(
        self,
        retriever: Any = None,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        use_new_api: bool = False,
    ):
        """
        Initialize adapter.

        Args:
            retriever: Optional SwarmRetriever instance
            param_bounds: Optional parameter bounds for validation
            use_new_api: If True, use the new builder pattern API
        """
        self.retriever = retriever
        self.param_bounds = param_bounds or self._default_bounds()
        self.use_new_api = use_new_api

    def _default_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Default parameter bounds for SwarmRetriever."""
        return {
            "n_agents": (5, 30),
            "steps": (4, 12),
            "decay": (0.85, 0.99),
            "initial_pool_size": (10, 50),
            "start_subset": (5, 15),
            "drop_zone_inc": (0.05, 0.2),
        }

    def validate_genome_kwargs(self, compiled: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate compiled genome kwargs.

        Args:
            compiled: Output from GenomeCompiler.compile()

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check required parameters
        required_params = ["n_agents", "steps", "decay", "initial_pool_size"]
        for param in required_params:
            if param not in compiled:
                errors.append(f"Missing required parameter: {param}")

        # Validate parameter bounds
        for param, (low, high) in self.param_bounds.items():
            if param in compiled:
                value = compiled[param]
                if value < low or value > high:
                    errors.append(
                        f"Parameter '{param}' value {value} outside bounds [{low}, {high}]"
                    )

        # Validate strategy structure
        if "agent_groups" in compiled:
            for i, group in enumerate(compiled["agent_groups"]):
                if "count" not in group:
                    errors.append(f"Agent group {i} missing 'count' field")
                if group.get("count", 0) <= 0:
                    errors.append(f"Agent group {i} has invalid count: {group.get('count')}")

        return len(errors) == 0, errors

    def from_genome(
        self,
        compiled: Dict[str, Any],
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Translate compiled genome to SwarmRetriever kwargs.

        Args:
            compiled: Output from GenomeCompiler.compile()
            validate: Whether to validate before translation

        Returns:
            Dict ready to be unpacked into SwarmRetriever methods

        Raises:
            ValueError: If validation fails and validate=True
        """
        if validate:
            is_valid, errors = self.validate_genome_kwargs(compiled)
            if not is_valid:
                raise ValueError(f"Invalid genome configuration: {errors}")

        # The compiled genome is already in the right format for SwarmRetriever
        # This method exists for:
        # 1. Future-proofing against SwarmRetriever changes
        # 2. Providing a clear integration point
        # 3. Optional transformations

        return compiled.copy()

    def retrieve(
        self,
        query: str,
        compiled: Dict[str, Any],
        decision_tracker: Optional[Any] = None
    ) -> List[Any]:
        """
        Run retrieval with translated genome kwargs.

        Args:
            query: Query string
            compiled: Compiled genome kwargs
            decision_tracker: Optional decision tracker

        Returns:
            List of retrieved results (old API) or SingleResult (new API)
        """
        if self.retriever is None:
            raise RuntimeError("No retriever configured for adapter")

        kwargs = self.from_genome(compiled)

        if self.use_new_api:
            # New builder pattern API
            config = self._kwargs_to_config(kwargs)
            result = self.retriever.query(query).run(config=config)
            # Convert to list format for backward compatibility
            return self._result_to_list(result)

        # Old API
        if decision_tracker is not None:
            return self.retriever.retrieve(
                query=query,
                decision_tracker=decision_tracker,
                **kwargs
            )
        else:
            return self.retriever.retrieve(query=query, **kwargs)

    def retrieve_batch(
        self,
        queries: List[str],
        compiled: Dict[str, Any],
        max_workers: int = 1,
        genome_id: Optional[str] = None
    ) -> List[List[Any]]:
        """
        Run batch retrieval with translated genome kwargs.

        Args:
            queries: List of query strings
            compiled: Compiled genome kwargs
            max_workers: Max concurrent workers
            genome_id: Optional identifier for logging

        Returns:
            List of result lists (one per query)
        """
        if self.retriever is None:
            raise RuntimeError("No retriever configured for adapter")

        kwargs = self.from_genome(compiled)

        if self.use_new_api:
            # New builder pattern API
            config = self._kwargs_to_config(kwargs)
            # Determine batch_size based on max_workers hint
            batch_size = 64 if max_workers > 1 else 1
            run_config = {
                "mode": "batched" if max_workers > 1 else "sequential",
                "batch_size": batch_size,
            }
            result = self.retriever.query(queries).run(config=config, run=run_config)
            # Convert to list of lists format for backward compatibility
            return self._batch_result_to_lists(result)

        # Old API
        return self.retriever.retrieve_batch(
            queries=queries,
            max_workers=max_workers,
            genome_id=genome_id,
            **kwargs
        )

    def _kwargs_to_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert legacy kwargs to RetrievalConfig format."""
        # Filter to only config-relevant keys
        config_keys = {
            "n_agents", "steps", "decay", "initial_pool_size",
            "start_subset", "top_k", "drop_zone_inc",
            "movement_strategies", "deposit_strategies", "ranking_strategies",
        }
        return {k: v for k, v in kwargs.items() if k in config_keys}

    def _result_to_list(self, result) -> List[Dict]:
        """Convert SingleResult to list of dicts for backward compatibility."""
        results = []
        for i in range(result.node_ids.shape[0]):
            results.append({
                "id": int(result.node_ids[i].item()),
                "score": float(result.scores[i].item()),
            })
        return results

    def _batch_result_to_lists(self, result) -> List[List[Dict]]:
        """Convert BatchResult to list of list of dicts for backward compatibility."""
        all_results = []
        n_queries = result.node_ids.shape[0]
        for q_idx in range(n_queries):
            query_results = []
            for i in range(result.node_ids.shape[1]):
                query_results.append({
                    "id": int(result.node_ids[q_idx, i].item()),
                    "score": float(result.scores[q_idx, i].item()),
                })
            all_results.append(query_results)
        return all_results


def parse_strategy_key(key: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a strategy key into group prefix and type suffix.

    Strategy naming convention: gN_type
    - g0_movement -> ("g0", "movement")
    - g1_deposit -> ("g1", "deposit")
    - ranking -> (None, "ranking")

    Args:
        key: Strategy key string

    Returns:
        Tuple of (group_prefix, strategy_type) or (None, key) for global strategies
    """
    if "_" not in key:
        # Global strategy like "ranking"
        if key in VALID_STRATEGY_SUFFIXES:
            return None, key
        return None, None

    parts = key.rsplit("_", 1)
    if len(parts) != 2:
        return None, None

    prefix, suffix = parts

    # Validate prefix format (gN where N is a number)
    if not prefix.startswith(STRATEGY_PREFIX):
        return None, None

    try:
        int(prefix[len(STRATEGY_PREFIX):])
    except ValueError:
        return None, None

    # Validate suffix
    if suffix not in VALID_STRATEGY_SUFFIXES:
        return None, None

    return prefix, suffix


def validate_strategy_keys(strategies: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate strategy key naming conventions.

    Args:
        strategies: Dict mapping strategy keys to values

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    for key in strategies:
        prefix, suffix = parse_strategy_key(key)

        if prefix is None and suffix is None:
            errors.append(
                f"Invalid strategy key '{key}'. Expected format 'gN_type' "
                f"where type is one of {VALID_STRATEGY_SUFFIXES}"
            )

    return len(errors) == 0, errors
