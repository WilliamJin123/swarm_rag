import random
import time
import torch
from typing import Any, List, Dict, Optional, Sequence, Tuple, TypedDict, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from contextlib import contextmanager
import logging
from ..utils import LRUCache, get_device, move_to_device, tensor_like


class StepProfiler:
    """Lightweight profiler for retrieval hot paths. Enable with SWARM_PROFILE=1."""

    __slots__ = ('enabled', 'timings', '_cuda_sync')

    def __init__(self, enabled: bool = False, cuda_sync: bool = True):
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = {}
        self._cuda_sync = cuda_sync and torch.cuda.is_available()

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        if self._cuda_sync:
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if self._cuda_sync:
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)

    def reset(self):
        self.timings.clear()

    def summary(self) -> str:
        if not self.timings:
            return "No profiling data"
        lines = ["Profiling Summary (ms):"]
        for name, times in sorted(self.timings.items()):
            total = sum(times)
            avg = total / len(times) if times else 0
            lines.append(f"  {name}: total={total:.2f}, avg={avg:.3f}, calls={len(times)}")
        return "\n".join(lines)

from .heuristics import HeuristicRegistry, Heuristics, HeuristicContext
from ..interfaces.abstract_classes import VectorStore, GraphStore, EmbeddingProvider, Matrix

class AgentGroupConfig(TypedDict):
    """
    Configuration for a specific sub-group of agents.
    """
    count: int  # How many agents of this type?
    movement_strategies: Dict[str, Any]
    deposit_strategies: Dict[str, Any]

logger = logging.getLogger(__name__)


class SwarmRetriever:
    _DEFAULT_PARAMS = dict(
        # Global Defaults
        steps=4,
        n_agents=20,
        decay=0.5,
        drop_zone_inc=0.05,
        initial_pool_size=30,
        start_subset=10,
        top_k=20,
        # Default "Homogeneous" Strategy (Fallback)
        movement_strategies={
            "semantic": ("semantic_similarity", 0.3),
            "centrality": ("node_centrality", 0.4),
            "diversity": ("pheromone_repulsion", 0.3),
        },
        ranking_strategies={
            "visited": ("percentage_visited", 0.2),
            "semantic": ("semantic_rank", 0.8),
        },
        deposit_strategies={
            "flat_mark": ("flat", 1.0),
        },
    )
    
    PHEROMONE_EPSILON = 1e-6

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        embedding_provider: EmbeddingProvider,
        seed: int = None,
        cache_neighbors: bool = False,
        neighbor_cache_size: int = 10000,
        degree_cache_size: int = 10000,
        cache_vectors: bool = True,
        doc_cache_size: int = 50000,
        query_cache_size: int = 1000,
        device: str = None,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embed_fn = embedding_provider

        self.py_rng = random.Random(seed) if seed else random
        # Use torch Generator for random operations
        self._torch_gen = torch.Generator()
        if seed is not None:
            self._torch_gen.manual_seed(seed)

        self._neighbor_lock = Lock()
        self._doc_lock = Lock()
        self._query_lock = Lock()

        self.avg_degree = self.graph_store.get_avg_degree()
        # Pre-compute log(1 + avg_degree) to avoid per-step computation
        self._avg_log_degree = float(torch.log(torch.tensor(1.0 + self.avg_degree)).item())
        # Cache max_node_id for dense pheromone tensor building
        self._max_node_id = self.graph_store.n_nodes
        # Pheromone buffer size with headroom for out-of-bounds IDs
        self._pheromone_buffer_size = max(self._max_node_id + 1024, 150000)

        self.cache_neighbors = cache_neighbors
        if self.cache_neighbors:
            self.neighbor_cache = LRUCache(neighbor_cache_size)
            self.degree_cache = LRUCache(degree_cache_size)

        self.cache_vectors = cache_vectors
        if self.cache_vectors:
            self.doc_cache = LRUCache(doc_cache_size)
            self.query_cache = LRUCache(query_cache_size)

        # Device configuration - flows down from caller
        if device is None:
            device = get_device()
        self._device = device
        self._use_gpu = (device != "cpu")  # True for cuda and mps

        if self._use_gpu:
            logger.info(f"SwarmRetriever: GPU acceleration enabled on {self._device}")
        else:
            logger.debug("SwarmRetriever: Using CPU mode")

        # Profiler for performance analysis (enabled via SWARM_PROFILE=1)
        import os
        self._profiler = StepProfiler(
            enabled=os.environ.get('SWARM_PROFILE', '0') == '1',
            cuda_sync=self._use_gpu
        )

        # CUDA graph acceleration (enabled via enable_cuda_graphs())
        self._cuda_graph_enabled = False
        self._cuda_graph = None
        self._graph_buffers = None
        self._graph_n_agents = None
        self._graph_max_degree = None

    def enable_compiled_mode(self) -> bool:
        """
        Enable torch.compile() optimization for the step computation.

        torch.compile() fuses GPU kernels and reduces Python overhead,
        providing ~10-20% speedup on repeated operations.

        Returns:
            True if compilation successful, False otherwise
        """
        if not self._use_gpu:
            logger.warning("Compiled mode requires GPU")
            return False

        try:
            # Compile the score computation function
            self._compiled_compute_scores = torch.compile(
                self._compute_agent_scores,
                mode="reduce-overhead",
                fullgraph=False
            )
            self._cuda_graph_enabled = True
            logger.info("torch.compile() enabled for step computation")
            return True
        except Exception as e:
            logger.warning(f"torch.compile() failed: {e}")
            return False

    def disable_compiled_mode(self):
        """Disable torch.compile() optimization."""
        self._cuda_graph_enabled = False
        self._compiled_compute_scores = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _compute_agent_scores(
        self,
        neighbor_sims: torch.Tensor,
        all_neighbor_degrees: torch.Tensor,
        neighbor_pheromones: torch.Tensor,
        neighbor_mask: torch.Tensor,
        max_pheromone: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined agent scores - separable for torch.compile().

        This function is designed to be compilable by torch.compile().
        """
        # Semantic similarity: scale to [0, 1]
        semantic_scores = (neighbor_sims + 1.0) / 2.0

        # Centrality heuristic (log degree normalized)
        log_degrees = torch.log(1 + all_neighbor_degrees)
        centrality_scores = log_degrees / (log_degrees + self._avg_log_degree + 1e-8)

        # Pheromone repulsion
        normalized_pheromones = neighbor_pheromones / (max_pheromone + 1e-8)
        repulsion_scores = 1.0 - normalized_pheromones

        # Combine with default weights
        total_scores = (
            0.3 * semantic_scores +
            0.4 * centrality_scores +
            0.3 * repulsion_scores
        )

        # Apply mask and clamp
        total_scores = torch.where(neighbor_mask, total_scores, torch.zeros_like(total_scores))
        total_scores = torch.clamp(total_scores, min=0.001)

        return total_scores

    def _create_pheromone_tensor(self) -> torch.Tensor:
        """Create fresh pheromone tensor for a query."""
        return torch.zeros(self._pheromone_buffer_size, dtype=torch.float32, device=self._device)

    def _safe_scatter_add(self, pheromone_tensor: torch.Tensor,
                          deposit_ids: torch.Tensor, deposit_vals: torch.Tensor) -> torch.Tensor:
        """Scatter add with bounds checking - no CUDA sync required."""
        if deposit_ids.numel() == 0:
            return pheromone_tensor

        # Ensure correct dtype for scatter indexing
        deposit_ids = deposit_ids.to(dtype=torch.long)

        # Filter to valid range: 0 <= id < buffer_size (pure tensor ops, no sync)
        buffer_size = pheromone_tensor.size(0)
        valid_mask = (deposit_ids >= 0) & (deposit_ids < buffer_size)

        if not valid_mask.any():
            return pheromone_tensor

        # Apply mask if needed
        if not valid_mask.all():
            deposit_ids = deposit_ids[valid_mask]
            deposit_vals = deposit_vals[valid_mask]

        pheromone_tensor.scatter_add_(0, deposit_ids, deposit_vals)
        return pheromone_tensor

    def _resolve_params(self, **user_params) -> Dict:
        """Standard parameter resolution."""
        active_user_params = {k: v for k, v in user_params.items() if v is not None}
        resolved_params = self._DEFAULT_PARAMS.copy()
        resolved_params.update(active_user_params)
        return resolved_params
    
    def retrieve(
            self,
            query: Any,
            agent_groups: Optional[List[AgentGroupConfig]] = None,
            seed: Optional[int] = None,
            n_agents: Optional[int] = None,
            steps: Optional[int] = None,
            decay: Optional[float] = None,
            drop_zone_inc: Optional[float] = None,
            initial_pool_size: Optional[int] = None,
            start_subset: Optional[int] = None,
            top_k: Optional[int] = None,
            movement_strategies: Optional[Dict] = None,
            ranking_strategies: Optional[Dict] = None,
            deposit_strategies: Optional[Dict] = None,
            decision_tracker: Optional[Any] = None,  # DecisionTracker for LLM context
        ) -> List[Dict]:
            
        # Handle explicit seeding for this run
        if seed is not None:
            py_rng = random.Random(seed)
            torch_gen = torch.Generator()
            torch_gen.manual_seed(seed)
            # Also update instance RNGs for sequential consistency if needed later
            self.py_rng = py_rng
            self._torch_gen = torch_gen
        else:
            py_rng = self.py_rng
            torch_gen = self._torch_gen
        
        params = self._resolve_params(
            n_agents=n_agents,
            steps=steps,
            decay=decay,
            drop_zone_inc=drop_zone_inc,
            initial_pool_size=initial_pool_size,
            start_subset=start_subset,
            top_k=top_k,
            ranking_strategies=ranking_strategies,
            movement_strategies=movement_strategies,
            deposit_strategies=deposit_strategies,
        )

        resolved_agents = self._prepare_agents(
            agent_groups=agent_groups,
            n_agents=params['n_agents'], 
            movement_strategies=params['movement_strategies'],
            deposit_strategies=params['deposit_strategies'],
        )

        query_vec = self._get_cached_query_vector(query)

        return self._retrieve(
            query_vec=query_vec,
            resolved_agents=resolved_agents,
            py_rng=py_rng,
            torch_gen=torch_gen,
            decision_tracker=decision_tracker,
            **params)

    def retrieve_batch(
        self,
        queries: List[Any],
        agent_groups: Optional[List[AgentGroupConfig]] = None,
        seed: Optional[int] = None,
        n_agents: Optional[int] = None,
        steps: Optional[int] = None,
        decay: Optional[float] = None,
        drop_zone_inc: Optional[float] = None,
        initial_pool_size: Optional[int] = None,
        start_subset: Optional[int] = None,
        top_k: Optional[int] = None,
        movement_strategies: Optional[Dict] = None,
        ranking_strategies: Optional[Dict] = None,
        deposit_strategies: Optional[Dict] = None,
        max_workers: Optional[int] = 4,
        **kwargs
    ) -> List[List[Dict]]:
        """
        Hybrid batch retrieval that intelligently chooses between sequential and parallel processing.
        """

        if not queries: return []
        
        # We don't update self.*_rng here to avoid side effects during parallel exec.
        # Instead, we pass explicit RNGs to workers.
        base_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

        params = self._resolve_params(
            n_agents=n_agents,
            steps=steps,
            decay=decay,
            drop_zone_inc=drop_zone_inc,
            initial_pool_size=initial_pool_size,
            start_subset=start_subset,
            top_k=top_k,
            ranking_strategies=ranking_strategies,
            movement_strategies=movement_strategies,
            deposit_strategies=deposit_strategies,
            **kwargs
        )

        resolved_agents = self._prepare_agents(
            agent_groups=agent_groups,
            n_agents=params['n_agents'], 
            movement_strategies=params['movement_strategies'],
            deposit_strategies=params['deposit_strategies'],
        )

        # Batch embed all queries
        query_matrix = self._get_cached_query_embeddings_batch(queries)

        # Decide processing strategy
        if max_workers > 1 and len(queries) > 1:
            return self._retrieve_batch_parallel(
                query_matrix,
                resolved_agents,
                base_seed=base_seed,
                max_workers=max_workers,
                **params
            )
        else:
            return self._retrieve_batch_sequential(
                query_matrix,
                resolved_agents,
                base_seed=base_seed,
                **params
            )

    def _retrieve_batch_sequential(
        self,
        query_vectors: Sequence[torch.Tensor],
        resolved_agents: List[Tuple[Callable, Callable]],
        base_seed: int,
        **kwargs
    ) -> List[List[Dict]]:
        """Process queries sequentially."""
        results = []
        total = len(query_vectors)
        gid = kwargs.get('genome_id', '')
        if gid != '': gid = f"[{gid}]"

        for i, vec in enumerate(query_vectors):
            if (i + 1) % 10 == 0 or (i + 1) == total:
                logger.info(f"    [Retriever] {gid} Sequential Progress: {i+1}/{total} queries")

            # Deterministic seeding for each query
            q_seed = base_seed + i
            py_rng = random.Random(q_seed)
            torch_gen = torch.Generator()
            torch_gen.manual_seed(q_seed)

            result = self._retrieve(
                query_vec=vec,
                resolved_agents=resolved_agents,
                py_rng=py_rng,
                torch_gen=torch_gen,
                **kwargs
            )
            results.append(result)
        return results

    def _retrieve_batch_parallel(
        self,
        query_vectors: Sequence[torch.Tensor],
        resolved_agents: List[Tuple[Callable, Callable]],
        max_workers: int,
        base_seed: int,
        **kwargs
    ) -> List[List[Dict]]:
        """Process queries in parallel with controlled concurrency and determinism."""

        def process(idx: int, vec: torch.Tensor) -> tuple[int, List[Dict]]:
            # Isolated RNGs for thread safety and determinism
            task_seed = base_seed + idx
            task_py_rng = random.Random(task_seed)
            task_torch_gen = torch.Generator()
            task_torch_gen.manual_seed(task_seed)

            result = self._retrieve(
                query_vec=vec,
                resolved_agents=resolved_agents,
                py_rng=task_py_rng,
                torch_gen=task_torch_gen,
                **kwargs
                )
            return idx, result
        
        total = len(query_vectors) 
        completed = 0
        gid = kwargs.get('genome_id', '')
        if gid != '': gid = f"[{gid}]"

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with their indices
            futures_to_index = {
                executor.submit(process, i, vec): i
                for i, vec in enumerate(query_vectors)
            }
            
            # Collect results maintaining order
            results = [None] * len(query_vectors)
            for future in as_completed(futures_to_index):
                completed += 1
                if completed % 10 == 0 or completed == total:
                     logger.info(f"    [Retriever] {gid} Parallel Progress: {completed}/{total} queries")

                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    idx = futures_to_index[future]
                    logger.error(f"Query {idx} failed: {e}")
                    results[idx] = []
            
            return results

    def _retrieve(
        self,
        query_vec: torch.Tensor,
        resolved_agents: List[Tuple[Callable, Callable]],
        steps: int,
        decay: float,
        drop_zone_inc: float,
        initial_pool_size: int,
        start_subset: int,
        top_k: int,
        ranking_strategies: Dict,
        py_rng: random.Random,
        torch_gen: torch.Generator,
        decision_tracker: Optional[Any] = None,  # DecisionTracker for LLM context
        **kwargs  # Catch unused global params
    ) -> List[Dict]:
        """
        Core retrieval logic shared between retrieve() and retrieve_batch().
        """

        n_agents = len(resolved_agents)
        prof = self._profiler

        # Normalize and Flatten
        query_vec = torch.as_tensor(query_vec)
        query_vec = query_vec.flatten()
        query_vec = query_vec / (torch.linalg.norm(query_vec) + 1e-8)

        # Pre-compose ranking strategy
        ranking_func = self._compose_strategy(ranking_strategies, "ranking")

        with prof.section("initial_search"):
            # Initial search with caching - now returns tensors
            search_ids, search_scores = self.vector_store.search(query_vec, limit=initial_pool_size)

        with prof.section("pool_filter"):
            # Filter to nodes that exist in graph (vectorized, no sync)
            if isinstance(search_ids, torch.Tensor):
                search_ids_tensor = search_ids.to(device=self._device, dtype=torch.long)
            else:
                search_ids_tensor = torch.as_tensor(search_ids, device=self._device, dtype=torch.long)
            valid_mask = (search_ids_tensor >= 0) & (search_ids_tensor < self._max_node_id)
            valid_pool_tensor = search_ids_tensor[valid_mask]

        if valid_pool_tensor.numel() == 0:
            return []

        # Select drop zone (first start_subset valid nodes)
        dz_len = min(start_subset, valid_pool_tensor.numel())
        drop_zone_tensor = valid_pool_tensor[:dz_len]

        # Cache Warming (need list for ThreadPoolExecutor)
        if self.cache_neighbors:
            drop_zone_list = drop_zone_tensor.tolist()
            with ThreadPoolExecutor(max_workers=min(4, dz_len)) as ex:
                list(ex.map(self._get_cached_neighbors, drop_zone_list))

        # Spawn Agents using GPU-based weighted sampling
        # Weights: "better" (earlier) nodes get higher weight
        weights = torch.arange(dz_len, 0, -1, device=self._device, dtype=torch.float32)
        weights = weights * drop_zone_inc + 1.0
        weights = weights / weights.sum()  # Normalize to probabilities

        # Sample with replacement using multinomial
        # Note: torch generator only works on CPU, so we skip it for GPU tensors
        if self._use_gpu:
            sampled_indices = torch.multinomial(weights, n_agents, replacement=True)
        else:
            sampled_indices = torch.multinomial(weights, n_agents, replacement=True, generator=torch_gen)
        agent_locations = drop_zone_tensor[sampled_indices]

        # Position history tensor: (n_agents, steps + 1), -1 = unvisited
        # Eliminates per-agent .item() calls in the hot loop
        position_history = torch.full((n_agents, steps + 1), -1, device=self._device, dtype=torch.long)
        position_history[:, 0] = agent_locations

        # Tensor-based pheromones for GPU acceleration
        query_pheromones = self._create_pheromone_tensor()

        # Initialize decision tracking if provided
        if decision_tracker is not None:
            decision_tracker.start_query(
                query_id=id(query_vec),  # Use object id as query identifier
                n_agents=n_agents,
                n_steps=steps
            )

        # Check if we can use batched GPU processing
        use_batched = (
            self._use_gpu and
            hasattr(self.graph_store, 'get_neighbors_batch') and
            getattr(self.graph_store, 'is_gpu', False) and
            decision_tracker is None  # Batched mode doesn't support decision tracking
        )

        # --- TRAVERSAL LOOP ---
        for step in range(steps):
            pheromone_updates = None
            # GPU tensor operation - no sync needed, clamp ensures min of 1.0
            max_pheromone = torch.clamp(query_pheromones.max(), min=1.0)

            if use_batched:
                with prof.section("step_batched"):
                    # GPU-accelerated batched agent processing
                    new_locations, pheromone_updates = self._step_agents_batched(
                        agent_locations=agent_locations,
                        query_vec=query_vec,
                        query_pheromones=query_pheromones,
                        resolved_agents=resolved_agents,
                        step=step,
                        max_pheromone=max_pheromone,
                        torch_gen=torch_gen,
                    )

                # Pure tensor assignment - no .item() calls
                position_history[:, step + 1] = new_locations
                agent_locations = new_locations

            else:
                # Sequential processing with decision tracking support
                seq_updates = {}
                for agent_idx, (move_fn, deposit_fn) in enumerate(resolved_agents):
                    current_loc = agent_locations[agent_idx].item()

                    result = self._process_agent_step(
                        agent_id=agent_idx,
                        current_loc=current_loc,
                        query_vec=query_vec,
                        query_pheromones=query_pheromones,
                        move_func=move_fn,
                        deposit_func=deposit_fn,
                        step=step,
                        max_pheromone=max_pheromone,
                        torch_gen=torch_gen,
                        decision_tracker=decision_tracker,
                    )

                    if result:
                        next_node = result['new_location']
                        agent_locations[agent_idx] = next_node
                        position_history[agent_idx, step + 1] = next_node

                        deposit = result['deposit']
                        # Aggressive pruning: only track significant deposits
                        if deposit > self.PHEROMONE_EPSILON:
                            seq_updates[next_node] = seq_updates.get(next_node, 0.0) + deposit

                pheromone_updates = seq_updates if seq_updates else None

            # Decay: single GPU operation
            query_pheromones *= decay

            # Deposit: handle both batched (tensor tuple) and sequential (dict) returns
            if pheromone_updates is not None:
                if isinstance(pheromone_updates, tuple):
                    # Batched path: (deposit_ids, deposit_vals) tensors
                    deposit_ids, deposit_vals = pheromone_updates
                    query_pheromones = self._safe_scatter_add(query_pheromones, deposit_ids, deposit_vals)
                else:
                    # Sequential path: dict
                    for node_id, amount in pheromone_updates.items():
                        if node_id < query_pheromones.size(0):
                            query_pheromones[node_id] += amount

        with prof.section("ranking"):
            result = self._ranking_from_history(
                position_history,
                query_vec,
                ranking_func,
                top_k,
                n_agents
            )

        return result

    # === HELPERS ===

    def _prepare_agents(
        self,
        agent_groups: Optional[List[AgentGroupConfig]],
        n_agents: int,
        movement_strategies: Dict,
        deposit_strategies: Dict
    ) -> List[Tuple[Callable, Callable]]:
        """
        Flattens agent groups into a single list of (move_fn, dep_fn) tuples.
        Pre-composes strategies into single callables.
        """
        agents = []

        if agent_groups:
            for group in agent_groups:
                count = group.get('count', 1)
                if count <= 0: continue
                
                move_fn = self._compose_strategy(group.get('movement_strategies'), "movement")
                dep_fn = self._compose_strategy(group.get('deposit_strategies'), "deposit")
                
                # Replicate references (cheap)
                agents.extend([(move_fn, dep_fn)] * count)
        else:
            # Homogeneous fallback
            move_fn = self._compose_strategy(movement_strategies, "movement")
            dep_fn = self._compose_strategy(deposit_strategies, "deposit")
            agents.extend([(move_fn, dep_fn)] * n_agents)
            
        return agents

    def _compose_strategy(
        self, 
        strategy_dict: Dict, 
        strategy_type: str
    ) -> Callable[[HeuristicContext], Any]:
        """
        Pre-composes heuristics into a single optimized callable.
        """
        components = []
        for key, (fn_or_name, weight) in strategy_dict.items():
            if callable(fn_or_name):
                fn = fn_or_name
            elif isinstance(fn_or_name, str):
                if strategy_type == "movement":
                    fn = HeuristicRegistry.get_movement(fn_or_name)
                elif strategy_type == "ranking":
                    fn = HeuristicRegistry.get_ranking(fn_or_name)
                elif strategy_type == "deposit":
                    fn = HeuristicRegistry.get_deposit(fn_or_name)
                else:
                    raise ValueError(f"Unknown strategy type: {strategy_type}")
            else:
                raise TypeError(f"Invalid heuristic entry: {fn_or_name}")
            components.append((fn, float(weight)))

        # Optimization: Flatten execution loop inside the callable
        
        if strategy_type == "deposit":
            def combined_deposit(ctx: HeuristicContext) -> float:
                total = 0.0
                for func, w in components:
                    val = func(ctx)
                    # Optimize for common case: tensor of size 1 or scalar
                    if isinstance(val, torch.Tensor):
                        if val.numel() == 1:
                            total += val.item() * w
                        else:
                            # Fallback if heuristic returns full tensor (rare for deposit on single node)
                            total += torch.sum(val).item() * w
                    else:
                        total += val * w
                return total
            return combined_deposit

        elif strategy_type == "ranking":
            def combined_ranking(ctx: HeuristicContext) -> torch.Tensor:
                # Initialize accumulator based on target shape
                if ctx.target_vecs is not None:
                    device = ctx.target_vecs.device
                    total = torch.zeros(ctx.target_vecs.shape[0], device=device, dtype=torch.float32)
                else:
                    total = torch.tensor(0.0)

                for func, w in components:
                    val = func(ctx)
                    # Ensure tensor
                    if not isinstance(val, torch.Tensor):
                        val = torch.as_tensor(val, device=total.device, dtype=torch.float32)
                    total = total + val * w
                return total
            return combined_ranking

        elif strategy_type == "movement":
            def combined_movement(ctx: HeuristicContext) -> torch.Tensor:
                if not components:
                    return torch.empty(0)

                # Unroll first iteration to init accumulator with correct shape/type
                fn0, w0 = components[0]
                total_scores = fn0(ctx) * w0

                for i in range(1, len(components)):
                    func, w = components[i]
                    total_scores += func(ctx) * w

                return total_scores
            return combined_movement

        return lambda ctx: 0.0  # Fallback

    def _fetch_embeddings(
        self,
        node_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified embedding fetch. Keeps everything as tensors.

        Args:
            node_ids: Tensor of node IDs to fetch

        Returns:
            (embeddings, valid_ids) - both tensors on device
            Only returns rows for valid IDs (no NaN rows).
        """
        device = node_ids.device

        if not self.cache_vectors:
            # Direct fetch from store
            matrix, valid_mask = self.vector_store.fetch_batch(node_ids)
            return matrix[valid_mask], node_ids[valid_mask]

        # Cache-aware fetch - need list for dict key lookup
        node_ids_list = node_ids.tolist()

        raw_vecs = [None] * len(node_ids_list)
        missing_indices = []
        missing_ids = []

        # Phase 1: Read (Locked)
        with self._doc_lock:
            for i, node_id in enumerate(node_ids_list):
                cached_vec = self.doc_cache.get(node_id)
                if cached_vec is not None:
                    raw_vecs[i] = cached_vec
                else:
                    missing_indices.append(i)
                    missing_ids.append(node_id)

        # Phase 2: Fetch (Unlocked)
        if missing_ids:
            fetched_matrix, valid_fetched_mask = self.vector_store.fetch_batch(missing_ids)

            # Phase 3: Write-back (Locked)
            with self._doc_lock:
                for i, is_valid in enumerate(valid_fetched_mask):
                    if is_valid:
                        original_idx = missing_indices[i]
                        vec = fetched_matrix[i]
                        self.doc_cache.set(node_ids_list[original_idx], vec)
                        raw_vecs[original_idx] = vec

        # Build output tensors
        valid_data = [(nid, v) for nid, v in zip(node_ids_list, raw_vecs) if v is not None]

        if not valid_data:
            return torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)

        valid_ids_list, valid_vecs = zip(*valid_data)
        embeddings = torch.stack(list(valid_vecs))
        valid_ids_tensor = torch.as_tensor(valid_ids_list, dtype=torch.long, device=device)
        return embeddings, valid_ids_tensor

    def _process_agent_step(
        self,
        agent_id: int,
        current_loc: int,
        query_vec: torch.Tensor,
        query_pheromones: torch.Tensor,
        move_func: Callable,
        deposit_func: Callable,
        step: int,
        max_pheromone: float,
        torch_gen: torch.Generator,
        decision_tracker: Optional[Any] = None,  # DecisionTracker for LLM context
    ) -> Optional[Dict]:
        """Vectorized agent step processing using pre-composed heuristics."""
        neighbors = self._get_cached_neighbors(current_loc)
        if len(neighbors) == 0:
            return None
        if step % 2 == 0:
            logger.debug(f"Agent {agent_id} at {current_loc} (degree={len(neighbors)})")

        # Fetch Matrix & IDs (Two-phase fetch handled internally by _fetch_vectors_batch)
        candidate_matrix, valid_ids = self._fetch_vectors_batch(neighbors)
        if len(valid_ids) == 0:
            return None

        # Direct tensor indexing for pheromone lookup
        valid_ids_tensor = torch.as_tensor(valid_ids, device=candidate_matrix.device, dtype=torch.long)
        pheromone_size = query_pheromones.size(0)
        clamped_ids = valid_ids_tensor.clamp(0, pheromone_size - 1)
        p_vals = query_pheromones[clamped_ids]
        # Zero out-of-bounds
        p_vals = torch.where(valid_ids_tensor >= pheromone_size, torch.zeros_like(p_vals), p_vals)

        # Safe degree fetching using Two-Phase Fetch for degrees
        degrees = self._fetch_degrees_batch(valid_ids)

        ctx = HeuristicContext(
            query_vec=query_vec,
            target_vecs=candidate_matrix,
            target_ids=valid_ids,
            pheromone_values=p_vals,
            node_degrees=degrees,
            max_pheromone=max_pheromone,
            avg_degree=self.avg_degree,
            step_index=step,
            agent_index=agent_id,
            graph=self.graph_store
        )

        # Calculate weighted scores via single call
        total_scores = move_func(ctx)

        # Capture individual heuristic scores if tracking decisions
        heuristic_scores = {}
        if decision_tracker is not None and decision_tracker.enabled:
            heuristic_scores = self._capture_heuristic_scores(ctx)

        # Ensure tensor on correct device
        if not isinstance(total_scores, torch.Tensor):
            total_scores = torch.as_tensor(total_scores, dtype=torch.float32, device=candidate_matrix.device)
        total_scores = torch.atleast_1d(torch.clamp(total_scores, min=0.001))

        # Ensure total_scores matches valid_ids length (broadcast scalar if needed)
        if len(total_scores) == 1 and len(valid_ids) > 1:
            total_scores = total_scores.expand(len(valid_ids)).clone()
        elif len(total_scores) != len(valid_ids):
            logger.warning(f"Score mismatch: {len(total_scores)} scores vs {len(valid_ids)} candidates")
            return None

        if len(valid_ids) > 5:
            total_scores = torch.where(total_scores < 0.01, torch.zeros_like(total_scores), total_scores)

        if torch.sum(total_scores) == 0:
            return None

        # Selection
        probs = total_scores / torch.sum(total_scores)
        chosen_idx = int(torch.multinomial(probs, 1, generator=torch_gen).item())
        next_node = valid_ids[chosen_idx]

        # Calculate deposit via single call
        ctx.target_vecs = candidate_matrix[chosen_idx : chosen_idx+1]
        ctx.target_ids = [next_node]
        ctx.pheromone_values = p_vals[chosen_idx : chosen_idx+1]
        ctx.node_degrees = degrees[chosen_idx : chosen_idx+1]

        deposit_amount = deposit_func(ctx)

        # Record decision for LLM context
        if decision_tracker is not None:
            decision_tracker.record_decision(
                agent_id=agent_id,
                step=step,
                current_node=current_loc,
                candidates=valid_ids,
                heuristic_scores=heuristic_scores,
                final_scores=total_scores,
                chosen_node=next_node,
                chosen_index=chosen_idx,
                deposit=deposit_amount
            )

        return {
            'new_location': next_node,
            'node_id': next_node,
            'deposit': deposit_amount
        }

    def _step_agents_batched(
        self,
        agent_locations: torch.Tensor,
        query_vec: torch.Tensor,
        query_pheromones: torch.Tensor,
        resolved_agents: List[Tuple[Callable, Callable]],
        step: int,
        max_pheromone: float,
        torch_gen: torch.Generator,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Process all agents in one batched GPU operation.

        This method provides significant speedup by:
        1. Batch fetching all neighbors for all agents in single GPU call
        2. Batch fetching all embeddings in single GPU call
        3. Computing all similarities in one matrix operation
        4. Vectorized softmax selection

        Args:
            agent_locations: Tensor of current agent positions
            query_vec: Query embedding vector
            query_pheromones: Pheromone tensor (size: _pheromone_buffer_size)
            resolved_agents: List of (move_fn, deposit_fn) tuples
            step: Current step index
            max_pheromone: Maximum pheromone value
            torch_gen: PyTorch random generator

        Returns:
            Tuple of (new_locations tensor, (deposit_ids, deposit_vals) or None)
        """
        n_agents = len(agent_locations)
        device = self._device
        prof = self._profiler

        # Convert positions to tensor on target device
        positions = agent_locations.to(device=device, dtype=torch.long)

        with prof.section("step.neighbors"):
            # Batch fetch all neighbors using GPU graph store
            all_neighbors, neighbor_mask = self.graph_store.get_neighbors_batch(positions)
            # all_neighbors: (n_agents, max_degree), neighbor_mask: (n_agents, max_degree)

        if all_neighbors is None or neighbor_mask is None:
            # Fallback to sequential processing if batch not supported
            return self._step_agents_sequential_fallback(
                agent_locations, query_vec, query_pheromones,
                resolved_agents, step, max_pheromone, torch_gen
            )

        max_degree = all_neighbors.shape[1]

        # Handle agents with no neighbors
        agent_has_neighbors = neighbor_mask.any(dim=1)  # (n_agents,)

        # Try FAST PATH: fused neighbor similarity computation (skips unique→fetch→scatter)
        fused_sims = None
        if hasattr(self.vector_store, 'compute_neighbor_similarities'):
            with prof.section("step.fused_sim"):
                fused_sims = self.vector_store.compute_neighbor_similarities(
                    query_vec, all_neighbors, neighbor_mask
                )

        if fused_sims is not None:
            # FAST PATH: fused computation succeeded
            neighbor_sims = fused_sims
        else:
            # SLOW PATH: original unique→fetch→scatter pipeline
            with prof.section("step.unique"):
                flat_neighbors_gpu = all_neighbors[neighbor_mask]
                valid_flat = flat_neighbors_gpu[flat_neighbors_gpu >= 0]
                unique_neighbors_gpu = torch.unique(valid_flat)

            if unique_neighbors_gpu.numel() == 0:
                return agent_locations, None

            with prof.section("step.fetch_emb"):
                embs, valid_mask = self.vector_store.fetch_batch(unique_neighbors_gpu)
                unique_embs = embs[valid_mask].to(device=device, dtype=torch.float32)
                valid_unique_ids = unique_neighbors_gpu[valid_mask].to(device=device, dtype=torch.long)

            if valid_unique_ids.numel() == 0:
                return agent_locations, None

            with prof.section("step.similarity"):
                query_tensor = torch.as_tensor(query_vec, device=device, dtype=torch.float32).view(1, -1)
                query_tensor = torch.nn.functional.normalize(query_tensor, p=2, dim=1)
                all_similarities = torch.mm(query_tensor, unique_embs.t()).squeeze(0)

            neighbor_sims = torch.full(
                (n_agents, max_degree), -float('inf'),
                device=device, dtype=torch.float32
            )

            with prof.section("step.id_mapping"):
                if valid_unique_ids.numel() > 0:
                    mapping_size = self._max_node_id + 1
                    id_to_idx_tensor = torch.full((mapping_size,), -1, device=device, dtype=torch.long)
                    id_to_idx_tensor[valid_unique_ids] = torch.arange(valid_unique_ids.numel(), device=device)

                    valid_neighbor_ids = all_neighbors[neighbor_mask]
                    clamped_ids = valid_neighbor_ids.clamp(0, mapping_size - 1)
                    emb_indices = id_to_idx_tensor[clamped_ids]

                    valid_emb_mask = emb_indices >= 0
                    if valid_emb_mask.any():
                        valid_emb_indices = emb_indices[valid_emb_mask]
                        neighbor_sims_flat = neighbor_sims[neighbor_mask]
                        neighbor_sims_flat[valid_emb_mask] = all_similarities[valid_emb_indices]
                        neighbor_sims[neighbor_mask] = neighbor_sims_flat

        # Build pheromone tensor
        neighbor_pheromones = torch.zeros(
            (n_agents, max_degree), device=device, dtype=torch.float32
        )

        # Fetch degrees for all neighbors (for centrality heuristic)
        all_neighbor_degrees = torch.ones(
            (n_agents, max_degree), device=device, dtype=torch.float32
        )
        if hasattr(self.graph_store, 'get_degrees_batch'):
            valid_neighbor_ids = all_neighbors[neighbor_mask]
            if valid_neighbor_ids.numel() > 0:
                valid_degrees = self.graph_store.get_degrees_batch(valid_neighbor_ids)
                all_neighbor_degrees[neighbor_mask] = valid_degrees.float()

        # Direct tensor indexing for pheromone lookup - no conversion needed
        valid_neighbor_ids = all_neighbors[neighbor_mask]
        pheromone_size = query_pheromones.size(0)
        clamped_ids = valid_neighbor_ids.clamp(0, pheromone_size - 1)
        flat_pheromones = query_pheromones[clamped_ids]
        # Zero out-of-bounds (IDs >= tensor size)
        out_of_bounds = valid_neighbor_ids >= pheromone_size
        flat_pheromones = torch.where(out_of_bounds, torch.zeros_like(flat_pheromones), flat_pheromones)
        neighbor_pheromones[neighbor_mask] = flat_pheromones

        # Compute combined scores using vectorized heuristics
        # Semantic similarity (already computed)
        semantic_scores = neighbor_sims.clone()
        semantic_scores = (semantic_scores + 1.0) / 2.0  # Scale to [0, 1]

        # Centrality heuristic (log degree normalized)
        log_degrees = torch.log(1 + all_neighbor_degrees)
        # Use pre-computed avg_log_degree (computed once in __init__)
        centrality_scores = log_degrees / (log_degrees + self._avg_log_degree + 1e-8)

        # Pheromone repulsion
        normalized_pheromones = neighbor_pheromones / (max_pheromone + 1e-8)
        repulsion_scores = 1.0 - normalized_pheromones

        # Combine with default weights (matching DEFAULT_PARAMS)
        # semantic: 0.3, centrality: 0.4, diversity: 0.3
        total_scores = (
            0.3 * semantic_scores +
            0.4 * centrality_scores +
            0.3 * repulsion_scores
        )

        # Apply mask: set invalid neighbors to 0
        total_scores = torch.where(neighbor_mask, total_scores, torch.zeros_like(total_scores))

        # Clamp minimum scores
        total_scores = torch.clamp(total_scores, min=0.001)

        # Softmax selection for each agent
        # Add small epsilon to avoid division by zero
        score_sums = total_scores.sum(dim=1, keepdim=True)
        probs = total_scores / (score_sums + 1e-10)

        # Handle agents with no valid neighbors
        probs = torch.where(
            score_sums > 0,
            probs,
            torch.zeros_like(probs)
        )

        # Vectorized GPU sampling using torch.multinomial
        # For agents with no valid neighbors, we'll keep them in place

        # Prepare for vectorized sampling
        sampling_probs = probs.clone()

        # Handle agents with no valid probabilities
        prob_sums = sampling_probs.sum(dim=1)
        has_valid = prob_sums > 1e-10

        # Initialize chosen positions and new locations
        chosen_positions = torch.zeros(n_agents, dtype=torch.long, device=device)
        new_locations_tensor = torch.as_tensor(agent_locations, device=device, dtype=torch.long)

        # Vectorized sampling for agents with valid probabilities
        valid_agent_mask = has_valid & agent_has_neighbors
        if valid_agent_mask.any():
            # Get valid agents' probabilities
            valid_probs = sampling_probs[valid_agent_mask]
            # Normalize (probabilities should already be normalized but ensure it)
            valid_probs = valid_probs / (valid_probs.sum(dim=1, keepdim=True) + 1e-10)

            # torch.multinomial for vectorized sampling
            # This samples one index per row based on probabilities
            sampled_indices = torch.multinomial(valid_probs, num_samples=1).squeeze(-1)
            chosen_positions[valid_agent_mask] = sampled_indices

            # Gather chosen neighbors using the sampled positions
            valid_chosen = all_neighbors[valid_agent_mask].gather(
                1, chosen_positions[valid_agent_mask].unsqueeze(1)
            ).squeeze(1)
            new_locations_tensor[valid_agent_mask] = valid_chosen

        # Keep as tensor - no numpy conversion
        new_locations = new_locations_tensor

        # Return tensors directly - no .tolist() conversion
        deposit_amount = 1.0
        if deposit_amount > self.PHEROMONE_EPSILON and valid_agent_mask.any():
            moved_nodes = new_locations[valid_agent_mask]
            unique_nodes, counts = torch.unique(moved_nodes, return_counts=True)
            deposit_vals = deposit_amount * counts.float()
            return new_locations, (unique_nodes, deposit_vals)
        else:
            return new_locations, None

    def _step_agents_sequential_fallback(
        self,
        agent_locations: torch.Tensor,
        query_vec: torch.Tensor,
        query_pheromones: torch.Tensor,
        resolved_agents: List[Tuple[Callable, Callable]],
        step: int,
        max_pheromone: float,
        torch_gen: torch.Generator,
    ) -> Tuple[torch.Tensor, Optional[Dict[int, float]]]:
        """
        Fallback to sequential agent processing when batch mode unavailable.
        Returns dict for sequential path compatibility.
        """
        new_locations = agent_locations.clone()
        pheromone_updates = {}

        for agent_idx, (move_fn, deposit_fn) in enumerate(resolved_agents):
            current_loc = agent_locations[agent_idx].item()

            result = self._process_agent_step(
                agent_id=agent_idx,
                current_loc=current_loc,
                query_vec=query_vec,
                query_pheromones=query_pheromones,
                move_func=move_fn,
                deposit_func=deposit_fn,
                step=step,
                max_pheromone=max_pheromone,
                torch_gen=torch_gen,
            )

            if result:
                next_node = result['new_location']
                new_locations[agent_idx] = next_node

                deposit = result['deposit']
                if deposit > self.PHEROMONE_EPSILON:
                    pheromone_updates[next_node] = pheromone_updates.get(next_node, 0.0) + deposit

        return new_locations, pheromone_updates if pheromone_updates else None

    def _capture_heuristic_scores(self, ctx: HeuristicContext) -> Dict[str, torch.Tensor]:
        """
        Capture individual heuristic scores for decision analysis.

        This method computes individual heuristic contributions separately
        to provide insight into agent decision-making. Used only when
        decision tracking is enabled.
        """
        scores = {}
        try:
            scores["semantic_similarity"] = Heuristics.semantic_similarity(ctx)
        except Exception:
            pass
        try:
            scores["node_centrality"] = Heuristics.node_centrality(ctx)
        except Exception:
            pass
        try:
            scores["pheromone_repulsion"] = Heuristics.pheromone_repulsion(ctx)
        except Exception:
            pass
        return scores

    def _ranking_from_history(
        self,
        position_history: torch.Tensor,
        query_vec: torch.Tensor,
        ranking_func: Callable,
        top_k: int,
        n_agents: int
    ) -> List[Dict]:
        """
        Tensor-native ranking from position history.

        Uses torch.unique for visit counts and batched ranking context.
        Eliminates CPU-GPU transfers in the ranking hot path.

        Args:
            position_history: Shape (n_agents, max_steps+1), -1 for unvisited
            query_vec: Query embedding vector
            ranking_func: Composed ranking function
            top_k: Number of top results
            n_agents: Total agent count

        Returns:
            List of top-k results with scores
        """
        device = position_history.device

        # Flatten and get unique visited nodes with counts
        all_positions = position_history.flatten()
        valid_positions = all_positions[all_positions >= 0]

        if valid_positions.numel() == 0:
            return []

        unique_visited, visit_counts = torch.unique(valid_positions, return_counts=True)

        # Fetch embeddings - uses unified tensor-native method
        embeddings, valid_ids = self._fetch_embeddings(unique_visited)

        if valid_ids.numel() == 0:
            return []

        # Align visit_counts with valid_ids (some nodes may not have embeddings)
        # Build a mapping from unique_visited to visit_counts
        # Use pre-known max_node_id to avoid CUDA sync
        id_to_count = torch.zeros(self._max_node_id + 1, dtype=torch.long, device=device)
        id_to_count[unique_visited] = visit_counts
        aligned_counts = id_to_count[valid_ids]

        # Batched ranking context
        query_vec = query_vec.to(device=embeddings.device)
        ctx = HeuristicContext(
            query_vec=query_vec,
            target_vecs=embeddings,
            target_ids=valid_ids,
            votes=aligned_counts,
            total_agents=n_agents,
            graph=self.graph_store
        )

        # Get scores from ranking function (returns tensor)
        scores = ranking_func(ctx).flatten()

        # Top-k selection on GPU
        k = min(top_k, scores.numel())
        top_scores, top_indices = torch.topk(scores, k=k)
        top_ids = valid_ids[top_indices]

        # Convert at API boundary only
        return [
            {'id': int(nid), 'score': float(sc)}
            for nid, sc in zip(top_ids.tolist(), top_scores.tolist())
        ]

    def _get_cached_neighbors(self, node_id: int) -> torch.Tensor:
        """Gets or computes and caches the neighbor list, if enabled."""
        if not self.cache_neighbors:
            neighbors = self.graph_store.get_neighbors(node_id)
            if not isinstance(neighbors, torch.Tensor):
                neighbors = torch.as_tensor(neighbors, dtype=torch.long)
            return neighbors
        with self._neighbor_lock:
            cached = self.neighbor_cache.get(node_id)
        if cached is not None:
            return cached

        neighbors = self.graph_store.get_neighbors(node_id)
        if not isinstance(neighbors, torch.Tensor):
            neighbors = torch.as_tensor(neighbors, dtype=torch.long)
        with self._neighbor_lock:
            self.neighbor_cache.set(node_id, neighbors)
            self.degree_cache.set(node_id, len(neighbors))
        return neighbors
    
    def _fetch_vectors_batch(self, node_ids: Union[Sequence[int], torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
        """
        Fetches vectors efficiently using Two-Phase Fetch (Read Locked -> Fetch Unlocked -> Write Locked).

        Args:
            node_ids: Sequence or tensor of node IDs

        Returns:
            Tuple of (vectors tensor, valid_ids list)
        """
        # Convert to list for cache lookups (dict keys must be hashable)
        if isinstance(node_ids, torch.Tensor):
            node_ids_list = node_ids.tolist()
        else:
            node_ids_list = list(node_ids)

        if not self.cache_vectors:
            matrix, valid_mask = self.vector_store.fetch_batch(node_ids)

            if torch.all(valid_mask):
                return matrix, node_ids_list

            filtered_matrix = matrix[valid_mask]
            filtered_ids = [nid for i, nid in enumerate(node_ids_list) if valid_mask[i]]
            return filtered_matrix, filtered_ids

        raw_vecs = [None] * len(node_ids_list)
        missing_indices = []
        missing_ids = []

        # Phase 1: Read (Locked)
        with self._doc_lock:
            for i, node_id in enumerate(node_ids_list):
                cached_vec = self.doc_cache.get(node_id)
                if cached_vec is not None:
                    raw_vecs[i] = cached_vec
                else:
                    missing_indices.append(i)
                    missing_ids.append(node_id)

        # Phase 2: Fetch (Unlocked)
        if missing_ids:
            fetched_matrix, valid_fetched_mask = self.vector_store.fetch_batch(missing_ids)

            # Phase 3: Write-back (Locked)
            with self._doc_lock:
                for i, is_valid in enumerate(valid_fetched_mask):
                    if is_valid:
                        original_idx = missing_indices[i]
                        vec = fetched_matrix[i]
                        self.doc_cache.set(node_ids_list[original_idx], vec)
                        raw_vecs[original_idx] = vec

        valid_data = [(nid, v) for nid, v in zip(node_ids_list, raw_vecs) if v is not None]

        if not valid_data:
            return torch.empty(0), []

        valid_ids, valid_vecs = zip(*valid_data)
        return torch.stack(list(valid_vecs)), list(valid_ids)
        
    def _fetch_degrees_batch(self, node_ids: Sequence[int]) -> torch.Tensor:
        """
        Fetches degrees efficiently using Two-Phase Fetch.
        Returns tensor of degrees (int32).
        """
        if not self.cache_neighbors:
            # If caching is disabled, we must fetch neighbors to count them
            return torch.as_tensor([len(self._get_cached_neighbors(nid)) for nid in node_ids], dtype=torch.int32)

        degrees = torch.empty(len(node_ids), dtype=torch.int32)
        missing_indices = []
        missing_ids = []

        # Phase 1: Read (Locked)
        with self._neighbor_lock:
            for i, nid in enumerate(node_ids):
                d = self.degree_cache.get(nid)
                if d is not None:
                    degrees[i] = d
                else:
                    missing_indices.append(i)
                    missing_ids.append(nid)

        # Phase 2: Fetch (Unlocked)
        if missing_ids:
            # We must fetch neighbors to get the degree
            # We buffer the results to write back to cache
            fetched_data = []
            for nid in missing_ids:
                nb = self.graph_store.get_neighbors(nid)
                fetched_data.append((nid, nb))

            # Phase 3: Write-back (Locked)
            with self._neighbor_lock:
                for i, (nid, nb) in zip(missing_indices, fetched_data):
                    d = len(nb)
                    # Cache both the neighbors and the degree since we paid the cost
                    if not isinstance(nb, torch.Tensor):
                        nb = torch.as_tensor(nb, dtype=torch.long)
                    self.neighbor_cache.set(nid, nb)
                    self.degree_cache.set(nid, d)
                    degrees[i] = d

        return degrees

    def _get_cached_query_vector(self, query: Any) -> torch.Tensor:
        """Gets or computes and caches the query embedding, if enabled."""
        if not self.cache_vectors:
            emb = self.embed_fn.embed_query(query)
            if not isinstance(emb, torch.Tensor):
                emb = torch.as_tensor(emb, dtype=torch.float32, device=self._device)
            else:
                emb = emb.to(device=self._device)
            return emb
        with self._query_lock:
            cached = self.query_cache.get(query)
        if cached is not None:
            return cached.to(device=self._device)

        emb = self.embed_fn.embed_query(query)
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb, dtype=torch.float32, device=self._device)
        else:
            emb = emb.to(device=self._device)
        with self._query_lock:
            self.query_cache.set(query, emb)
        return emb
        
    def _get_cached_query_embeddings_batch(self, queries: list) -> Matrix:
        """
        Retrieves embeddings for a batch of queries, returning a single 2D tensor.
        """
        if not queries:
            return torch.empty(0)
        if not self.cache_vectors:
            embs = self.embed_fn.embed_query_batch(queries)
            if not isinstance(embs, torch.Tensor):
                embs = torch.as_tensor(embs, dtype=torch.float32)
            return embs

        results_by_index: Dict[Any, torch.Tensor] = {}
        missing_indices = []
        missing_queries = []

        with self._query_lock:
            for i, q in enumerate(queries):
                cached_vec = self.query_cache.get(q)
                if cached_vec is not None:
                    results_by_index[i] = cached_vec
                else:
                    missing_indices.append(i)
                    missing_queries.append(q)

        if missing_queries:
            batch_embeddings = self.embed_fn.embed_query_batch(missing_queries)
            if not isinstance(batch_embeddings, torch.Tensor):
                batch_embeddings = torch.as_tensor(batch_embeddings, dtype=torch.float32)
            with self._query_lock:
                for i, emb in zip(missing_indices, batch_embeddings):
                    q = queries[i]
                    self.query_cache.set(q, emb)
                    results_by_index[i] = emb

        if not results_by_index:
            return torch.empty(0)

        first_embedding = next(iter(results_by_index.values()))
        embedding_dim = first_embedding.shape[0]
        batch_size = len(queries)

        final_embeddings = torch.empty((batch_size, embedding_dim), dtype=first_embedding.dtype)

        for i in range(batch_size):
            final_embeddings[i, :] = results_by_index[i]

        return final_embeddings

    @property
    def device(self) -> str:
        """Return the device this retriever is using."""
        return self._device

    @property
    def is_gpu_enabled(self) -> bool:
        """Check if GPU acceleration is active."""
        return self._use_gpu

    def retrieve_with_precomputed(
        self,
        query_embedding: torch.Tensor,
        initial_pool: List[int],
        agent_groups: Optional[List[AgentGroupConfig]] = None,
        seed: Optional[int] = None,
        n_agents: Optional[int] = None,
        steps: Optional[int] = None,
        decay: Optional[float] = None,
        drop_zone_inc: Optional[float] = None,
        start_subset: Optional[int] = None,
        top_k: Optional[int] = None,
        movement_strategies: Optional[Dict] = None,
        ranking_strategies: Optional[Dict] = None,
        deposit_strategies: Optional[Dict] = None,
        decision_tracker: Optional[Any] = None,
    ) -> List[Dict]:
        """
        Retrieve using pre-computed query embedding and initial pool.

        This method skips the embedding lookup and initial search steps,
        using pre-computed data from SharedPrecomputeContext. This provides
        significant speedup when evaluating multiple genomes on the same queries.

        Args:
            query_embedding: Pre-computed query embedding tensor (1D or 2D with shape (1, dim))
            initial_pool: Pre-computed initial candidate pool (list of IDs)
            agent_groups: Optional agent group configurations
            seed: Random seed for reproducibility
            n_agents: Number of agents
            steps: Number of traversal steps
            decay: Pheromone decay rate
            drop_zone_inc: Drop zone increment
            start_subset: Size of starting subset
            top_k: Number of results to return
            movement_strategies: Movement strategy configuration
            ranking_strategies: Ranking strategy configuration
            deposit_strategies: Deposit strategy configuration
            decision_tracker: Optional decision tracker for LLM context

        Returns:
            List of top-k results with scores
        """
        # Handle explicit seeding for this run
        if seed is not None:
            py_rng = random.Random(seed)
            torch_gen = torch.Generator()
            torch_gen.manual_seed(seed)
        else:
            py_rng = self.py_rng
            torch_gen = self._torch_gen

        params = self._resolve_params(
            n_agents=n_agents,
            steps=steps,
            decay=decay,
            drop_zone_inc=drop_zone_inc,
            start_subset=start_subset,
            top_k=top_k,
            ranking_strategies=ranking_strategies,
            movement_strategies=movement_strategies,
            deposit_strategies=deposit_strategies,
        )

        resolved_agents = self._prepare_agents(
            agent_groups=agent_groups,
            n_agents=params['n_agents'],
            movement_strategies=params['movement_strategies'],
            deposit_strategies=params['deposit_strategies'],
        )

        # Use the pre-computed query embedding directly
        query_vec = torch.as_tensor(query_embedding, dtype=torch.float32)
        if query_vec.ndim == 2:
            query_vec = query_vec.squeeze(0)
        query_vec = query_vec.flatten()
        query_vec = query_vec / (torch.linalg.norm(query_vec) + 1e-8)

        # Use the pre-computed initial pool
        if not initial_pool:
            return []

        return self._retrieve_with_pool_internal(
            query_vec=query_vec,
            initial_pool=initial_pool,
            resolved_agents=resolved_agents,
            py_rng=py_rng,
            torch_gen=torch_gen,
            decision_tracker=decision_tracker,
            **params
        )

    def retrieve_batch_with_precomputed(
        self,
        query_embeddings: torch.Tensor,
        initial_pools: List[List[int]],
        agent_groups: Optional[List[AgentGroupConfig]] = None,
        seed: Optional[int] = None,
        n_agents: Optional[int] = None,
        steps: Optional[int] = None,
        decay: Optional[float] = None,
        drop_zone_inc: Optional[float] = None,
        start_subset: Optional[int] = None,
        top_k: Optional[int] = None,
        movement_strategies: Optional[Dict] = None,
        ranking_strategies: Optional[Dict] = None,
        deposit_strategies: Optional[Dict] = None,
        max_workers: Optional[int] = 4,
        **kwargs
    ) -> List[List[Dict]]:
        """
        Batch retrieve using pre-computed query embeddings and initial pools.

        This method provides significant speedup by using pre-computed data
        from SharedPrecomputeContext, eliminating redundant embedding lookups
        and initial searches across multiple genome evaluations.

        Args:
            query_embeddings: Pre-computed query embeddings tensor (n_queries, dim)
            initial_pools: Pre-computed initial pools, one per query
            agent_groups: Optional agent group configurations
            seed: Base random seed for reproducibility
            n_agents: Number of agents
            steps: Number of traversal steps
            decay: Pheromone decay rate
            drop_zone_inc: Drop zone increment
            start_subset: Size of starting subset
            top_k: Number of results to return
            movement_strategies: Movement strategy configuration
            ranking_strategies: Ranking strategy configuration
            deposit_strategies: Deposit strategy configuration
            max_workers: Maximum parallel workers (CPU mode)
            **kwargs: Additional arguments

        Returns:
            List of result lists, one per query
        """
        if len(query_embeddings) == 0:
            return []

        base_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

        params = self._resolve_params(
            n_agents=n_agents,
            steps=steps,
            decay=decay,
            drop_zone_inc=drop_zone_inc,
            start_subset=start_subset,
            top_k=top_k,
            ranking_strategies=ranking_strategies,
            movement_strategies=movement_strategies,
            deposit_strategies=deposit_strategies,
            **kwargs
        )

        resolved_agents = self._prepare_agents(
            agent_groups=agent_groups,
            n_agents=params['n_agents'],
            movement_strategies=params['movement_strategies'],
            deposit_strategies=params['deposit_strategies'],
        )

        # Process based on device mode
        if self._use_gpu or max_workers <= 1:
            # Sequential processing for GPU (CUDA thread-locality)
            return self._retrieve_batch_precomputed_sequential(
                query_embeddings,
                initial_pools,
                resolved_agents,
                base_seed=base_seed,
                **params
            )
        else:
            # Parallel processing for CPU
            return self._retrieve_batch_precomputed_parallel(
                query_embeddings,
                initial_pools,
                resolved_agents,
                base_seed=base_seed,
                max_workers=max_workers,
                **params
            )

    def _retrieve_batch_precomputed_sequential(
        self,
        query_embeddings: torch.Tensor,
        initial_pools: List[List[int]],
        resolved_agents: List[Tuple[Callable, Callable]],
        base_seed: int,
        **kwargs
    ) -> List[List[Dict]]:
        """Process pre-computed queries sequentially."""
        results = []
        total = len(query_embeddings)
        gid = kwargs.get('genome_id', '')
        if gid:
            gid = f"[{gid}]"

        for i, (vec, pool) in enumerate(zip(query_embeddings, initial_pools)):
            if (i + 1) % 10 == 0 or (i + 1) == total:
                logger.info(f"    [Retriever] {gid} Precomputed Sequential: {i+1}/{total}")

            q_seed = base_seed + i
            py_rng = random.Random(q_seed)
            torch_gen = torch.Generator()
            torch_gen.manual_seed(q_seed)

            # Normalize query vector
            query_vec = torch.as_tensor(vec, dtype=torch.float32).flatten()
            query_vec = query_vec / (torch.linalg.norm(query_vec) + 1e-8)

            if not pool:
                results.append([])
                continue

            result = self._retrieve_with_pool_internal(
                query_vec=query_vec,
                initial_pool=pool,
                resolved_agents=resolved_agents,
                py_rng=py_rng,
                torch_gen=torch_gen,
                **kwargs
            )
            results.append(result)

        return results

    def _retrieve_batch_precomputed_parallel(
        self,
        query_embeddings: torch.Tensor,
        initial_pools: List[List[int]],
        resolved_agents: List[Tuple[Callable, Callable]],
        max_workers: int,
        base_seed: int,
        **kwargs
    ) -> List[List[Dict]]:
        """Process pre-computed queries in parallel."""
        def process(idx: int, vec: torch.Tensor, pool: List[int]) -> tuple[int, List[Dict]]:
            task_seed = base_seed + idx
            task_py_rng = random.Random(task_seed)
            task_torch_gen = torch.Generator()
            task_torch_gen.manual_seed(task_seed)

            query_vec = torch.as_tensor(vec, dtype=torch.float32).flatten()
            query_vec = query_vec / (torch.linalg.norm(query_vec) + 1e-8)

            if not pool:
                return idx, []

            result = self._retrieve_with_pool_internal(
                query_vec=query_vec,
                initial_pool=pool,
                resolved_agents=resolved_agents,
                py_rng=task_py_rng,
                torch_gen=task_torch_gen,
                **kwargs
            )
            return idx, result

        total = len(query_embeddings)
        completed = 0
        gid = kwargs.get('genome_id', '')
        if gid:
            gid = f"[{gid}]"

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures_to_index = {
                executor.submit(process, i, vec, pool): i
                for i, (vec, pool) in enumerate(zip(query_embeddings, initial_pools))
            }

            results = [None] * total
            for future in as_completed(futures_to_index):
                completed += 1
                if completed % 10 == 0 or completed == total:
                    logger.info(f"    [Retriever] {gid} Precomputed Parallel: {completed}/{total}")

                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    idx = futures_to_index[future]
                    logger.error(f"Query {idx} failed: {e}")
                    results[idx] = []

            return results

    def _retrieve_batch_multi_query_gpu(
        self,
        query_embeddings: torch.Tensor,
        initial_pools: List[List[int]],
        resolved_agents: List[Tuple[Callable, Callable]],
        base_seed: int,
        batch_size: int = 32,
        **kwargs
    ) -> List[List[Dict]]:
        """
        Process multiple queries simultaneously on GPU.

        Batches queries in chunks of `batch_size` to manage memory.
        Each chunk runs the full swarm traversal with batched tensor ops.

        Args:
            query_embeddings: Pre-computed query embeddings (n_queries, dim)
            initial_pools: List of initial candidate pools per query
            resolved_agents: List of (move_fn, deposit_fn) tuples
            base_seed: Base random seed for reproducibility
            batch_size: Number of queries to process simultaneously
            **kwargs: Additional params (steps, decay, top_k, etc.)

        Returns:
            List of result lists, one per query
        """
        # Stub: fall back to sequential for now
        return self._retrieve_batch_precomputed_sequential(
            query_embeddings, initial_pools, resolved_agents, base_seed, **kwargs
        )

    def _init_multi_query_state(
        self,
        batch_size: int,
        n_agents: int,
        n_nodes: int,
        steps: int,
        initial_pools: List[List[int]],
        drop_zone_inc: float,
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize batched state tensors for multi-query processing.

        Returns:
            agent_locations: (batch_size, n_agents) - current positions
            query_pheromones: (batch_size, n_nodes) - pheromone values
            position_history: (batch_size, n_agents, steps+1) - trajectory history
        """
        device = self._device

        # Initialize pheromones: (batch_size, n_nodes)
        query_pheromones = torch.zeros(
            (batch_size, n_nodes),
            dtype=torch.float32, device=device
        )

        # Initialize position history: (batch_size, n_agents, steps+1)
        position_history = torch.full(
            (batch_size, n_agents, steps + 1), -1,
            dtype=torch.long, device=device
        )

        # Initialize agent locations from pools
        agent_locations = torch.zeros(
            (batch_size, n_agents), dtype=torch.long, device=device
        )

        torch.manual_seed(seed)
        for q in range(batch_size):
            pool = initial_pools[q]
            if not pool:
                continue
            pool_tensor = torch.tensor(pool, device=device, dtype=torch.long)
            pool_len = len(pool)

            # Weighted sampling favoring earlier pool entries
            weights = torch.tensor(
                [1.0 + drop_zone_inc * (pool_len - i - 1) for i in range(pool_len)],
                device=device
            )
            weights = weights / weights.sum()

            # Sample agent starting positions
            indices = torch.multinomial(weights, n_agents, replacement=True)
            agent_locations[q] = pool_tensor[indices]

        # Record initial positions
        position_history[:, :, 0] = agent_locations

        return agent_locations, query_pheromones, position_history

    def _get_neighbors_multi_query(
        self,
        agent_locations: torch.Tensor,  # (batch_size, n_agents)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch fetch neighbors for all agents across all queries.

        Args:
            agent_locations: (batch_size, n_agents) positions

        Returns:
            all_neighbors: (batch_size, n_agents, max_degree) neighbor IDs
            neighbor_mask: (batch_size, n_agents, max_degree) validity mask
        """
        batch_size, n_agents = agent_locations.shape
        device = agent_locations.device

        # Flatten all positions
        flat_positions = agent_locations.flatten()  # (batch_size * n_agents,)

        # Batch fetch from graph store
        flat_neighbors, flat_mask = self.graph_store.get_neighbors_batch(flat_positions)
        # flat_neighbors: (batch_size * n_agents, max_degree)
        # flat_mask: (batch_size * n_agents, max_degree)

        max_degree = flat_neighbors.shape[1]

        # Reshape back to (batch_size, n_agents, max_degree)
        all_neighbors = flat_neighbors.view(batch_size, n_agents, max_degree)
        neighbor_mask = flat_mask.view(batch_size, n_agents, max_degree)

        return all_neighbors, neighbor_mask

    def _compute_similarities_multi_query(
        self,
        query_vecs: torch.Tensor,       # (batch_size, dim)
        all_neighbors: torch.Tensor,    # (batch_size, n_agents, max_degree)
        neighbor_mask: torch.Tensor,    # (batch_size, n_agents, max_degree)
    ) -> torch.Tensor:
        """
        Compute query-neighbor similarities for all queries simultaneously.

        Returns:
            similarities: (batch_size, n_agents, max_degree) similarity scores
        """
        batch_size, n_agents, max_degree = all_neighbors.shape
        device = query_vecs.device

        # Initialize output
        similarities = torch.full(
            (batch_size, n_agents, max_degree), 0.0,
            device=device, dtype=torch.float32
        )

        # Get unique neighbor IDs across all queries
        valid_neighbors = all_neighbors[neighbor_mask]
        if valid_neighbors.numel() == 0:
            return similarities

        unique_ids = torch.unique(valid_neighbors[valid_neighbors >= 0])
        if unique_ids.numel() == 0:
            return similarities

        # Fetch embeddings for unique IDs
        valid_embs, valid_ids = self._fetch_embeddings(unique_ids)

        if valid_ids.numel() == 0:
            return similarities

        # Normalize query vectors and embeddings
        query_vecs_norm = torch.nn.functional.normalize(query_vecs, p=2, dim=1)
        valid_embs_norm = torch.nn.functional.normalize(valid_embs, p=2, dim=1)

        # Compute all query-embedding similarities: (batch_size, n_unique)
        all_sims = torch.mm(query_vecs_norm, valid_embs_norm.t())

        # Build ID-to-index mapping
        max_id = self._max_node_id + 1
        id_to_idx = torch.full((max_id,), -1, device=device, dtype=torch.long)
        id_to_idx[valid_ids] = torch.arange(valid_ids.numel(), device=device)

        # Map neighbor IDs to embedding indices
        clamped_neighbors = all_neighbors.clamp(0, max_id - 1)
        emb_indices = id_to_idx[clamped_neighbors]  # (batch, agents, degree)

        # Scatter similarities into output
        # For each (q, a, n), if emb_indices[q,a,n] >= 0:
        #   similarities[q,a,n] = all_sims[q, emb_indices[q,a,n]]
        valid_emb_mask = (emb_indices >= 0) & neighbor_mask

        # Use advanced indexing: need batch indices for all_sims
        batch_idx = torch.arange(batch_size, device=device)[:, None, None].expand_as(emb_indices)

        # Gather similarities
        # all_sims is (batch, n_unique), we need sims for each (batch, agent, degree)
        flat_emb_idx = emb_indices[valid_emb_mask]
        flat_batch_idx = batch_idx[valid_emb_mask]
        gathered_sims = all_sims[flat_batch_idx, flat_emb_idx]

        similarities[valid_emb_mask] = gathered_sims

        # Scale to [0, 1]
        similarities = (similarities + 1.0) / 2.0
        similarities = torch.where(neighbor_mask, similarities, torch.zeros_like(similarities))

        return similarities

    def _lookup_pheromones_multi_query(
        self,
        query_pheromones: torch.Tensor,  # (batch_size, n_nodes)
        all_neighbors: torch.Tensor,      # (batch_size, n_agents, max_degree)
    ) -> torch.Tensor:
        """
        Look up pheromone values for all neighbors across all queries.

        Returns:
            pheromone_vals: (batch_size, n_agents, max_degree)
        """
        batch_size = query_pheromones.shape[0]
        n_nodes = query_pheromones.shape[1]
        device = query_pheromones.device

        # Clamp neighbor IDs to valid range
        clamped = all_neighbors.clamp(0, n_nodes - 1)

        # Advanced indexing: pheromones[q, neighbors[q,a,n]]
        batch_idx = torch.arange(batch_size, device=device)[:, None, None]
        batch_idx = batch_idx.expand_as(clamped)

        pheromone_vals = query_pheromones[batch_idx, clamped]

        # Zero out-of-bounds lookups
        out_of_bounds = (all_neighbors < 0) | (all_neighbors >= n_nodes)
        pheromone_vals = torch.where(out_of_bounds, torch.zeros_like(pheromone_vals), pheromone_vals)

        return pheromone_vals

    def _deposit_pheromones_multi_query(
        self,
        query_pheromones: torch.Tensor,  # (batch_size, n_nodes)
        new_locations: torch.Tensor,      # (batch_size, n_agents)
        deposit_amount: float = 1.0,
    ) -> torch.Tensor:
        """
        Deposit pheromones at new agent locations for all queries.

        Returns:
            Updated pheromone tensor
        """
        batch_size, n_nodes = query_pheromones.shape
        device = query_pheromones.device

        # Process each query (scatter_add doesn't support batch dimension well)
        for q in range(batch_size):
            locs = new_locations[q]
            valid_locs = locs[(locs >= 0) & (locs < n_nodes)]
            if valid_locs.numel() > 0:
                unique_locs, counts = torch.unique(valid_locs, return_counts=True)
                deposits = deposit_amount * counts.float()
                query_pheromones[q].scatter_add_(0, unique_locs, deposits)

        return query_pheromones

    def _step_multi_query(
        self,
        agent_locations: torch.Tensor,   # (batch_size, n_agents)
        query_vecs: torch.Tensor,        # (batch_size, dim)
        query_pheromones: torch.Tensor,  # (batch_size, n_nodes)
        step: int,
        max_pheromone: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Execute one step for all agents across all queries.

        Returns:
            new_locations: (batch_size, n_agents)
            deposit_locations: (batch_size, n_agents) or None
        """
        batch_size, n_agents = agent_locations.shape
        device = agent_locations.device

        # 1. Get neighbors for all agents
        all_neighbors, neighbor_mask = self._get_neighbors_multi_query(agent_locations)
        # (batch_size, n_agents, max_degree)

        if all_neighbors is None:
            return agent_locations, None

        max_degree = all_neighbors.shape[2]

        # 2. Compute similarities
        similarities = self._compute_similarities_multi_query(
            query_vecs, all_neighbors, neighbor_mask
        )

        # 3. Lookup pheromones
        pheromone_vals = self._lookup_pheromones_multi_query(
            query_pheromones, all_neighbors
        )

        # 4. Get degrees for centrality heuristic
        flat_neighbors = all_neighbors.flatten()
        flat_degrees = self.graph_store.get_degrees_batch(flat_neighbors.clamp(0, self._max_node_id))
        all_degrees = flat_degrees.float().view(batch_size, n_agents, max_degree)

        # 5. Compute heuristic scores
        # Centrality: log(1 + degree) / (log(1 + degree) + avg_log_degree)
        log_degrees = torch.log(1 + all_degrees)
        centrality_scores = log_degrees / (log_degrees + self._avg_log_degree + 1e-8)

        # Pheromone repulsion: 1 - normalized_pheromone
        normalized_pheromones = pheromone_vals / (max_pheromone + 1e-8)
        repulsion_scores = 1.0 - normalized_pheromones

        # Combine with default weights
        total_scores = (
            0.3 * similarities +
            0.4 * centrality_scores +
            0.3 * repulsion_scores
        )

        # Apply mask
        total_scores = torch.where(neighbor_mask, total_scores, torch.zeros_like(total_scores))
        total_scores = torch.clamp(total_scores, min=0.001)

        # 6. Normalize to probabilities
        score_sums = total_scores.sum(dim=-1, keepdim=True)
        probs = total_scores / (score_sums + 1e-10)

        # 7. Sample next positions
        # Reshape for multinomial: (batch_size * n_agents, max_degree)
        flat_probs = probs.view(-1, max_degree)

        # Handle zero-sum rows
        row_sums = flat_probs.sum(dim=1, keepdim=True)
        flat_probs = torch.where(
            row_sums > 1e-10,
            flat_probs,
            torch.ones_like(flat_probs) / max_degree  # Uniform fallback
        )
        flat_probs = flat_probs / (flat_probs.sum(dim=1, keepdim=True) + 1e-10)

        chosen_idx = torch.multinomial(flat_probs, 1).view(batch_size, n_agents)

        # Gather chosen neighbors
        new_locations = all_neighbors.gather(2, chosen_idx.unsqueeze(-1)).squeeze(-1)

        return new_locations, new_locations.clone()

    def _retrieve_with_pool_internal(
        self,
        query_vec: torch.Tensor,
        initial_pool: List[int],
        resolved_agents: List[Tuple[Callable, Callable]],
        py_rng: random.Random,
        torch_gen: torch.Generator,
        steps: int,
        decay: float,
        drop_zone_inc: float,
        start_subset: int,
        top_k: int,
        ranking_strategies: Dict,
        decision_tracker: Optional[Any] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Internal retrieval using a pre-computed initial pool.

        Shared implementation used by both retrieve_with_precomputed
        and _retrieve_with_pool.
        """
        n_agents = len(resolved_agents)

        # Pre-compose ranking strategy
        ranking_func = self._compose_strategy(ranking_strategies, "ranking")

        if not initial_pool:
            return []

        drop_zone = initial_pool[:start_subset]
        dz_len = len(drop_zone)

        # Cache warming
        if self.cache_neighbors:
            with ThreadPoolExecutor(max_workers=min(4, dz_len)) as ex:
                list(ex.map(self._get_cached_neighbors, drop_zone))

        # Spawn agents
        weights = [1.0 + drop_zone_inc * (dz_len - i - 1) for i in range(dz_len)]
        agent_locations = torch.as_tensor(
            py_rng.choices(drop_zone, weights=weights, k=n_agents),
            dtype=torch.long,
            device=self._device
        )

        # Position history tensor: (n_agents, steps + 1), -1 = unvisited
        position_history = torch.full((n_agents, steps + 1), -1, device=self._device, dtype=torch.long)
        position_history[:, 0] = agent_locations

        # Tensor-based pheromones for GPU acceleration
        query_pheromones = self._create_pheromone_tensor()

        # Initialize decision tracking if provided
        if decision_tracker is not None:
            decision_tracker.start_query(
                query_id=id(query_vec),
                n_agents=n_agents,
                n_steps=steps
            )

        # Check if we can use batched GPU processing
        use_batched = (
            self._use_gpu and
            hasattr(self.graph_store, 'get_neighbors_batch') and
            getattr(self.graph_store, 'is_gpu', False) and
            decision_tracker is None
        )

        # Traversal loop
        for step in range(steps):
            pheromone_updates = None
            # GPU tensor operation - no sync needed, clamp ensures min of 1.0
            max_pheromone = torch.clamp(query_pheromones.max(), min=1.0)

            if use_batched:
                new_locations, pheromone_updates = self._step_agents_batched(
                    agent_locations=agent_locations,
                    query_vec=query_vec,
                    query_pheromones=query_pheromones,
                    resolved_agents=resolved_agents,
                    step=step,
                    max_pheromone=max_pheromone,
                    torch_gen=torch_gen,
                )

                # Pure tensor assignment - no .item() calls
                position_history[:, step + 1] = new_locations
                agent_locations = new_locations
            else:
                seq_updates = {}
                for agent_idx, (move_fn, deposit_fn) in enumerate(resolved_agents):
                    current_loc = agent_locations[agent_idx].item()

                    result = self._process_agent_step(
                        agent_id=agent_idx,
                        current_loc=current_loc,
                        query_vec=query_vec,
                        query_pheromones=query_pheromones,
                        move_func=move_fn,
                        deposit_func=deposit_fn,
                        step=step,
                        max_pheromone=max_pheromone,
                        torch_gen=torch_gen,
                        decision_tracker=decision_tracker,
                    )

                    if result:
                        next_node = result['new_location']
                        agent_locations[agent_idx] = next_node
                        position_history[agent_idx, step + 1] = next_node

                        deposit = result['deposit']
                        if deposit > self.PHEROMONE_EPSILON:
                            seq_updates[next_node] = seq_updates.get(next_node, 0.0) + deposit

                pheromone_updates = seq_updates if seq_updates else None

            # Decay: single GPU operation
            query_pheromones *= decay

            # Deposit: handle both batched (tensor tuple) and sequential (dict) returns
            if pheromone_updates is not None:
                if isinstance(pheromone_updates, tuple):
                    # Batched path: (deposit_ids, deposit_vals) tensors
                    deposit_ids, deposit_vals = pheromone_updates
                    query_pheromones = self._safe_scatter_add(query_pheromones, deposit_ids, deposit_vals)
                else:
                    # Sequential path: dict
                    for node_id, amount in pheromone_updates.items():
                        if node_id < query_pheromones.size(0):
                            query_pheromones[node_id] += amount

        return self._ranking_from_history(
            position_history, query_vec, ranking_func, top_k, n_agents
        )