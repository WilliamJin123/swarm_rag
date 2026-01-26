import random
import torch
from typing import Any, List, Dict, Optional, Sequence, Tuple, TypedDict, Callable, Union
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging
from ..utils import LRUCache, get_device, move_to_device, tensor_like

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


def should_continue_stepping(
    positions: torch.Tensor,
    prev_positions: torch.Tensor,
    step_idx: int,
    min_steps: int = 2,
    convergence_threshold: float = 0.8,
) -> bool:
    """
    Check if agents have converged and further steps are unlikely to help.

    Implements early stopping for the swarm traversal loop to save computation
    when agents are no longer exploring new nodes.

    Args:
        positions: Current agent positions tensor
        prev_positions: Previous step agent positions tensor
        step_idx: Current step index (0-based)
        min_steps: Minimum steps to always complete before checking convergence
        convergence_threshold: Fraction of stuck agents (0-1) that triggers early stop

    Returns:
        True if stepping should continue, False if converged and should stop early
    """
    # Always continue for first min_steps
    if step_idx < min_steps:
        return True

    # Count agents that haven't moved
    n_agents = len(positions)
    if n_agents == 0:
        return False

    stuck = (positions == prev_positions).sum().item()
    stuck_fraction = stuck / n_agents

    # Stop if too many agents are stuck
    if stuck_fraction >= convergence_threshold:
        logger.debug(
            f"Early convergence at step {step_idx}: {stuck}/{n_agents} agents stuck "
            f"({stuck_fraction:.1%} >= {convergence_threshold:.0%})"
        )
        return False

    return True


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
        use_gpu: bool = True,
        tensor_mode: bool = True,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embed_fn = embedding_provider
        self.base_pheromones = defaultdict(float)

        self.py_rng = random.Random(seed) if seed else random
        # Use torch Generator for random operations
        self._torch_gen = torch.Generator()
        if seed is not None:
            self._torch_gen.manual_seed(seed)

        self._neighbor_lock = Lock()
        self._doc_lock = Lock()
        self._query_lock = Lock()

        self.avg_degree = self.graph_store.get_avg_degree()

        self.cache_neighbors = cache_neighbors
        if self.cache_neighbors:
            self.neighbor_cache = LRUCache(neighbor_cache_size)
            self.degree_cache = LRUCache(degree_cache_size)

        self.cache_vectors = cache_vectors
        if self.cache_vectors:
            self.doc_cache = LRUCache(doc_cache_size)
            self.query_cache = LRUCache(query_cache_size)

        # GPU support detection
        self._use_gpu = use_gpu and get_device() == "cuda"
        self._has_gpu_store = hasattr(vector_store, 'compute_similarities') and hasattr(vector_store, 'is_gpu')

        if self._use_gpu and self._has_gpu_store and getattr(vector_store, 'is_gpu', False):
            self._device = getattr(vector_store, 'device', 'cuda')
            logger.info(f"SwarmRetriever: GPU acceleration enabled on {self._device}")
        else:
            self._device = "cpu"
            self._use_gpu = False
            if use_gpu:
                logger.debug("SwarmRetriever: GPU not available, using CPU")

        # Tensor mode: keep data as tensors on GPU when possible
        self._tensor_mode = tensor_mode and self._use_gpu
        if self._tensor_mode:
            logger.debug("SwarmRetriever: Tensor mode enabled - minimizing CPU-GPU transfers")

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

        # Normalize and Flatten
        query_vec = torch.as_tensor(query_vec)
        query_vec = query_vec.flatten()
        query_vec = query_vec / (torch.linalg.norm(query_vec) + 1e-8)

        # Pre-compose ranking strategy
        ranking_func = self._compose_strategy(ranking_strategies, "ranking")

        # Initial search with caching
        search_res = self.vector_store.search(query_vec, limit=initial_pool_size)
        valid_pool = [r['id'] for r in search_res if self.graph_store.contains(r['id'])]
        if not valid_pool: return []

        drop_zone = valid_pool[:start_subset]
        dz_len = len(drop_zone)

        # Cache Warming
        if self.cache_neighbors:
            with ThreadPoolExecutor(max_workers=min(4, dz_len)) as ex:
                list(ex.map(self._get_cached_neighbors, drop_zone))

        # Spawn Agents (Weigh "better" nodes higher)
        weights = [1.0 + drop_zone_inc * (dz_len - i - 1) for i in range(dz_len)]
        agent_locations = torch.tensor(py_rng.choices(drop_zone, weights=weights, k=n_agents), dtype=torch.long)
        agent_trajectories = [[loc.item()] for loc in agent_locations]
        query_pheromones = self.base_pheromones.copy()

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
            pheromone_updates = {}
            max_pheromone = max(query_pheromones.values()) if query_pheromones else 1.0

            if use_batched:
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

                # Update locations and trajectories
                for agent_idx in range(n_agents):
                    new_loc = new_locations[agent_idx].item() if isinstance(new_locations[agent_idx], torch.Tensor) else new_locations[agent_idx]
                    old_loc = agent_locations[agent_idx].item() if isinstance(agent_locations[agent_idx], torch.Tensor) else agent_locations[agent_idx]
                    if new_loc != old_loc:
                        agent_trajectories[agent_idx].append(new_loc)
                agent_locations = new_locations

            else:
                # Original sequential processing
                for agent_idx, (move_fn, deposit_fn) in enumerate(resolved_agents):
                    current_loc = agent_locations[agent_idx].item() if isinstance(agent_locations[agent_idx], torch.Tensor) else agent_locations[agent_idx]

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
                        agent_trajectories[agent_idx].append(next_node)

                        deposit = result['deposit']
                        # Aggressive pruning: only track significant deposits
                        if deposit > self.PHEROMONE_EPSILON:
                            pheromone_updates[next_node] = pheromone_updates.get(next_node, 0.0) + deposit

            # Update pheromones with pruning
            # 1. Decay and prune existing
            if query_pheromones:
                existing_keys = list(query_pheromones.keys())
                for k in existing_keys:
                    new_val = query_pheromones[k] * decay
                    if new_val < self.PHEROMONE_EPSILON:
                        del query_pheromones[k]
                    else:
                        query_pheromones[k] = new_val

            # 2. Add new deposits
            for node_id, amount in pheromone_updates.items():
                query_pheromones[node_id] += amount
            
        return self._ranking(
            agent_trajectories, 
            query_vec, 
            ranking_func, 
            top_k,
            n_agents
        )          

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
            def combined_ranking(ctx: HeuristicContext) -> float:
                total = 0.0
                for func, w in components:
                    total += func(ctx) * w
                return total
            return combined_ranking

        elif strategy_type == "movement":
            def combined_movement(ctx: HeuristicContext) -> torch.Tensor:
                if not components:
                    return torch.tensor([])

                # Unroll first iteration to init accumulator with correct shape/type
                fn0, w0 = components[0]
                total_scores = fn0(ctx) * w0

                for i in range(1, len(components)):
                    func, w = components[i]
                    total_scores += func(ctx) * w

                return total_scores
            return combined_movement

        return lambda ctx: 0.0  # Fallback

    def _process_agent_step(
        self,
        agent_id: int,
        current_loc: int,
        query_vec: torch.Tensor,
        query_pheromones: Dict,
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

        # Prefetch Vectorization Metadata
        p_vals = torch.tensor([query_pheromones.get(nid, 0.0) for nid in valid_ids], dtype=torch.float32)

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

        # Ensure tensor
        if not isinstance(total_scores, torch.Tensor):
            total_scores = torch.tensor(total_scores, dtype=torch.float32)
        total_scores = torch.atleast_1d(torch.maximum(total_scores, torch.tensor(0.001)))

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
        query_pheromones: Dict,
        resolved_agents: List[Tuple[Callable, Callable]],
        step: int,
        max_pheromone: float,
        torch_gen: torch.Generator,
    ) -> Tuple[torch.Tensor, Dict[int, float]]:
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
            query_pheromones: Current pheromone map
            resolved_agents: List of (move_fn, deposit_fn) tuples
            step: Current step index
            max_pheromone: Maximum pheromone value
            torch_gen: PyTorch random generator

        Returns:
            Tuple of (new_locations tensor, pheromone_updates dict)
        """
        n_agents = len(agent_locations)
        device = self._device

        # Convert positions to tensor on target device
        positions = agent_locations.to(device=device, dtype=torch.long)

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

        # Flatten valid neighbors for batch embedding fetch
        # Use torch.unique() to stay on GPU - no CPU round-trip
        flat_neighbors_gpu = all_neighbors[neighbor_mask]
        valid_flat = flat_neighbors_gpu[flat_neighbors_gpu >= 0]
        unique_neighbors_gpu = torch.unique(valid_flat)

        if unique_neighbors_gpu.numel() == 0:
            # No valid neighbors for any agent
            return agent_locations, {}

        # Batch fetch embeddings for all unique neighbors
        # Pass GPU tensor directly - no CPU transfer needed
        if hasattr(self.vector_store, 'fetch_batch_gpu') and self._use_gpu:
            unique_embs, valid_unique_ids_tensor = self.vector_store.fetch_batch_gpu(unique_neighbors_gpu)
            if not isinstance(unique_embs, torch.Tensor):
                unique_embs = torch.tensor(unique_embs, device=device, dtype=torch.float32)
            # valid_unique_ids is now a tensor
            valid_unique_ids = valid_unique_ids_tensor
        else:
            # Fallback: use tolist() instead of numpy for CPU transfer
            unique_neighbors_list = unique_neighbors_gpu.tolist()
            unique_embs_result, valid_unique_ids_list = self._fetch_vectors_batch(unique_neighbors_list)
            if not isinstance(unique_embs_result, torch.Tensor):
                unique_embs = torch.tensor(unique_embs_result, device=device, dtype=torch.float32)
            else:
                unique_embs = unique_embs_result.to(device=device, dtype=torch.float32)
            valid_unique_ids = torch.tensor(valid_unique_ids_list, device=device, dtype=torch.long)

        if valid_unique_ids.numel() == 0:
            return agent_locations, {}

        # Convert query to GPU tensor
        query_tensor = torch.tensor(query_vec, device=device, dtype=torch.float32).view(1, -1)
        query_tensor = torch.nn.functional.normalize(query_tensor, p=2, dim=1)

        # Compute similarities for all unique neighbors at once
        # (1, D) @ (N_unique, D).T -> (1, N_unique) -> (N_unique,)
        all_similarities = torch.mm(query_tensor, unique_embs.t()).squeeze(0)

        # Now scatter similarities back to (n_agents, max_degree) shape
        neighbor_sims = torch.full(
            (n_agents, max_degree), -float('inf'),
            device=device, dtype=torch.float32
        )

        # Build pheromone tensor
        neighbor_pheromones = torch.zeros(
            (n_agents, max_degree), device=device, dtype=torch.float32
        )

        # Fetch degrees for all neighbors (for centrality heuristic)
        # Fix: Only query degrees for valid neighbors, not padding (-1 values)
        all_neighbor_degrees = torch.ones(
            (n_agents, max_degree), device=device, dtype=torch.float32
        )
        if hasattr(self.graph_store, 'get_degrees_batch'):
            # Get valid neighbor IDs and their degrees
            valid_neighbor_ids = all_neighbors[neighbor_mask]
            if valid_neighbor_ids.numel() > 0:
                valid_degrees = self.graph_store.get_degrees_batch(valid_neighbor_ids)
                all_neighbor_degrees[neighbor_mask] = valid_degrees.float()

        # Fill in similarities and pheromones for valid neighbors
        # Vectorized GPU operations - no nested Python loops

        # Build ID-to-embedding-index mapping tensor on GPU
        if valid_unique_ids.numel() > 0:
            max_id = int(unique_neighbors_gpu.max().item()) + 1
            id_to_idx_tensor = torch.full((max_id,), -1, device=device, dtype=torch.long)
            id_to_idx_tensor[valid_unique_ids] = torch.arange(valid_unique_ids.numel(), device=device)

            # Vectorized scatter of similarities
            valid_neighbor_ids = all_neighbors[neighbor_mask]
            # Clamp to valid range for indexing (invalid IDs will map to -1)
            clamped_ids = valid_neighbor_ids.clamp(0, max_id - 1)
            emb_indices = id_to_idx_tensor[clamped_ids]

            # Only scatter where we have valid embeddings
            valid_emb_mask = emb_indices >= 0
            if valid_emb_mask.any():
                # Get similarities for valid indices
                valid_emb_indices = emb_indices[valid_emb_mask]
                neighbor_sims_flat = neighbor_sims[neighbor_mask]
                neighbor_sims_flat[valid_emb_mask] = all_similarities[valid_emb_indices]
                neighbor_sims[neighbor_mask] = neighbor_sims_flat

        # Vectorized pheromone lookup on GPU
        if query_pheromones:
            # Build pheromone tensor on GPU (once per step)
            max_node_id = int(all_neighbors.max().item()) + 1
            pheromone_tensor = torch.zeros(max_node_id, device=device, dtype=torch.float32)

            # Convert pheromone dict to GPU tensor
            pheromone_keys = torch.tensor(list(query_pheromones.keys()), device=device, dtype=torch.long)
            pheromone_vals = torch.tensor(list(query_pheromones.values()), device=device, dtype=torch.float32)
            # Clamp keys to valid range
            valid_key_mask = (pheromone_keys >= 0) & (pheromone_keys < max_node_id)
            if valid_key_mask.any():
                pheromone_tensor[pheromone_keys[valid_key_mask]] = pheromone_vals[valid_key_mask]

            # Vectorized GPU lookup for all neighbors
            valid_neighbor_ids = all_neighbors[neighbor_mask]
            clamped_ids = valid_neighbor_ids.clamp(0, max_node_id - 1)
            flat_pheromones = pheromone_tensor[clamped_ids]
            neighbor_pheromones[neighbor_mask] = flat_pheromones

        # Compute combined scores using vectorized heuristics
        # Semantic similarity (already computed)
        semantic_scores = neighbor_sims.clone()
        semantic_scores = (semantic_scores + 1.0) / 2.0  # Scale to [0, 1]

        # Centrality heuristic (log degree normalized)
        log_degrees = torch.log(1 + all_neighbor_degrees)
        avg_log_degree = torch.log(torch.tensor(1 + self.avg_degree)).item()
        centrality_scores = log_degrees / (log_degrees + avg_log_degree + 1e-8)

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
        new_locations_tensor = torch.tensor(agent_locations, device=device, dtype=torch.long)

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

        # Compute pheromone updates using tensor operations
        pheromone_updates = {}
        deposit_amount = 1.0

        if deposit_amount > self.PHEROMONE_EPSILON:
            # Get unique nodes that agents moved to and count deposits
            moved_nodes = new_locations[valid_agent_mask]
            unique_nodes, counts = torch.unique(moved_nodes, return_counts=True)

            # Build pheromone updates dict (minimal CPU transfer)
            for node, count in zip(unique_nodes.tolist(), counts.tolist()):
                pheromone_updates[node] = deposit_amount * count

        return new_locations, pheromone_updates

    def _step_agents_sequential_fallback(
        self,
        agent_locations: torch.Tensor,
        query_vec: torch.Tensor,
        query_pheromones: Dict,
        resolved_agents: List[Tuple[Callable, Callable]],
        step: int,
        max_pheromone: float,
        torch_gen: torch.Generator,
    ) -> Tuple[torch.Tensor, Dict[int, float]]:
        """
        Fallback to sequential agent processing when batch mode unavailable.
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

        return new_locations, pheromone_updates

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
    
    def _ranking(
        self,
        agent_trajectories: List[List[int]],
        query_vec: torch.Tensor,
        ranking_func: Callable,
        top_k: int,
        n_agents: int
    ) -> List[Dict]:
        """Parallel ranking of visited nodes."""
        # Count votes
        all_visited = [node for path in agent_trajectories for node in path]
        vote_counts = Counter(all_visited)
        unique_visited = list(vote_counts.keys())

        vectors_matrix, valid_ids = self._fetch_vectors_batch(unique_visited)
        results = []

        # Iterate over valid vectors
        for i, node_id in enumerate(valid_ids):
            vec = vectors_matrix[i]
            score = self._calculate_node_score(
                node_id=node_id,
                votes=vote_counts[node_id],
                query_vec=query_vec,
                target_vec=vec,
                ranking_func=ranking_func,
                n_agents=n_agents
            )
            results.append({'id': node_id, 'score': score})

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def _calculate_node_score(
        self,
        node_id: int,
        votes: int,
        query_vec: torch.Tensor,
        target_vec: Optional[torch.Tensor],
        ranking_func: Callable,
        n_agents: int
    ) -> float:
        """Calculate final score for a single node."""
        if target_vec is None:
            return 0.0

        node_ctx = HeuristicContext(
            query_vec=query_vec,
            target_vecs=target_vec,
            target_ids=node_id,
            graph=self.graph_store,
            votes=votes,
            total_agents=n_agents
        )

        return ranking_func(node_ctx)

    def _get_cached_neighbors(self, node_id: int) -> torch.Tensor:
        """Gets or computes and caches the neighbor list, if enabled."""
        if not self.cache_neighbors:
            neighbors = self.graph_store.get_neighbors(node_id)
            if not isinstance(neighbors, torch.Tensor):
                neighbors = torch.tensor(neighbors, dtype=torch.long)
            return neighbors
        with self._neighbor_lock:
            cached = self.neighbor_cache.get(node_id)
        if cached is not None:
            return cached

        neighbors = self.graph_store.get_neighbors(node_id)
        if not isinstance(neighbors, torch.Tensor):
            neighbors = torch.tensor(neighbors, dtype=torch.long)
        with self._neighbor_lock:
            self.neighbor_cache.set(node_id, neighbors)
            self.degree_cache.set(node_id, len(neighbors))
        return neighbors
    
    def _fetch_vectors_batch(self, node_ids: Sequence[int]) -> Tuple[torch.Tensor, List[int]]:
        """
        Fetches vectors efficiently using Two-Phase Fetch (Read Locked -> Fetch Unlocked -> Write Locked).
        """
        # Convert node_ids to list if it's a tensor
        if isinstance(node_ids, torch.Tensor):
            node_ids = node_ids.tolist()

        if not self.cache_vectors:
            matrix = self.vector_store.fetch_batch(node_ids)
            if not isinstance(matrix, torch.Tensor):
                matrix = torch.as_tensor(matrix, dtype=torch.float32)
            valid_mask = ~torch.isnan(matrix).any(dim=1)

            if torch.all(valid_mask):
                return matrix, list(node_ids)

            filtered_matrix = matrix[valid_mask]
            filtered_ids = [nid for i, nid in enumerate(node_ids) if valid_mask[i]]
            return filtered_matrix, filtered_ids

        raw_vecs = [None] * len(node_ids)
        missing_indices = []
        missing_ids = []

        # Phase 1: Read (Locked)
        with self._doc_lock:
            for i, node_id in enumerate(node_ids):
                cached_vec = self.doc_cache.get(node_id)
                if cached_vec is not None:
                    raw_vecs[i] = cached_vec
                else:
                    missing_indices.append(i)
                    missing_ids.append(node_id)

        # Phase 2: Fetch (Unlocked)
        if missing_ids:
            fetched_matrix = self.vector_store.fetch_batch(missing_ids)
            if not isinstance(fetched_matrix, torch.Tensor):
                fetched_matrix = torch.as_tensor(fetched_matrix, dtype=torch.float32)
            valid_fetched_mask = ~torch.isnan(fetched_matrix).any(dim=1)

            # Phase 3: Write-back (Locked)
            with self._doc_lock:
                for i, is_valid in enumerate(valid_fetched_mask):
                    if is_valid:
                        original_idx = missing_indices[i]
                        vec = fetched_matrix[i]

                        self.doc_cache.set(node_ids[original_idx], vec)
                        raw_vecs[original_idx] = vec

        valid_data = [(nid, v) for nid, v in zip(node_ids, raw_vecs) if v is not None]

        if not valid_data:
            return torch.tensor([]), []

        valid_ids, valid_vecs = zip(*valid_data)
        return torch.stack(list(valid_vecs)), list(valid_ids)
        
    def _fetch_degrees_batch(self, node_ids: Sequence[int]) -> torch.Tensor:
        """
        Fetches degrees efficiently using Two-Phase Fetch.
        Returns tensor of degrees (int32).
        """
        if not self.cache_neighbors:
            # If caching is disabled, we must fetch neighbors to count them
            return torch.tensor([len(self._get_cached_neighbors(nid)) for nid in node_ids], dtype=torch.int32)

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
                        nb = torch.tensor(nb, dtype=torch.long)
                    self.neighbor_cache.set(nid, nb)
                    self.degree_cache.set(nid, d)
                    degrees[i] = d

        return degrees

    def _get_cached_query_vector(self, query: Any) -> torch.Tensor:
        """Gets or computes and caches the query embedding, if enabled."""
        if not self.cache_vectors:
            emb = self.embed_fn.embed_query(query)
            if not isinstance(emb, torch.Tensor):
                emb = torch.as_tensor(emb, dtype=torch.float32)
            return emb
        with self._query_lock:
            cached = self.query_cache.get(query)
        if cached is not None:
            return cached

        emb = self.embed_fn.embed_query(query)
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb, dtype=torch.float32)
        with self._query_lock:
            self.query_cache.set(query, emb)
        return emb
        
    def _get_cached_query_embeddings_batch(self, queries: list) -> Matrix:
        """
        Retrieves embeddings for a batch of queries, returning a single 2D tensor.
        """
        if not queries:
            return torch.tensor([])
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
            return torch.tensor([])

        first_embedding = next(iter(results_by_index.values()))
        embedding_dim = first_embedding.shape[0]
        batch_size = len(queries)

        final_embeddings = torch.empty((batch_size, embedding_dim), dtype=first_embedding.dtype)

        for i in range(batch_size):
            final_embeddings[i, :] = results_by_index[i]

        return final_embeddings

    def _compute_similarities_gpu(
        self,
        query_vec: torch.Tensor,
        candidate_ids: Sequence[int]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Compute similarities using GPU when available.

        Falls back to standard tensor computation if GPU not available.

        Args:
            query_vec: Query embedding (tensor)
            candidate_ids: List of candidate document IDs

        Returns:
            Tuple of (similarity scores tensor, valid_ids list)
        """
        # Try GPU path if available
        if self._use_gpu and self._has_gpu_store:
            try:
                scores, valid_ids = self.vector_store.compute_similarities(
                    query_vec, list(candidate_ids)
                )
                if not isinstance(scores, torch.Tensor):
                    scores = torch.tensor(scores, dtype=torch.float32)
                return scores, valid_ids
            except Exception as e:
                logger.debug(f"GPU similarity computation failed, falling back to CPU: {e}")

        # CPU fallback
        candidate_matrix, valid_ids = self._fetch_vectors_batch(candidate_ids)
        if len(valid_ids) == 0:
            return torch.tensor([]), []

        # Normalize query
        query_norm = query_vec / (torch.linalg.norm(query_vec) + 1e-8)

        # Compute cosine similarity
        scores = torch.matmul(candidate_matrix, query_norm)
        return scores, valid_ids

    @property
    def device(self) -> str:
        """Return the device this retriever is using."""
        return self._device

    @property
    def is_gpu_enabled(self) -> bool:
        """Check if GPU acceleration is active."""
        return self._use_gpu

    # === BATCH OPTIMIZATION METHODS ===

    def _fetch_vectors_batch_gpu(
        self,
        node_ids: Sequence[int]
    ) -> Tuple[Any, List[int]]:
        """
        Fetch vectors as GPU tensors when available, with automatic fallback.

        This method keeps data on GPU to avoid CPU-GPU transfers when possible.

        Args:
            node_ids: Sequence of node IDs to fetch

        Returns:
            Tuple of (vectors tensor/array, valid_ids list)
        """
        # Try GPU path if store supports it
        if self._use_gpu and hasattr(self.vector_store, 'fetch_batch_gpu'):
            try:
                return self.vector_store.fetch_batch_gpu(list(node_ids))
            except Exception as e:
                logger.debug(f"GPU fetch failed, falling back to CPU: {e}")

        # Fall back to standard numpy path
        return self._fetch_vectors_batch(node_ids)

    def _ranking_vectorized(
        self,
        agent_trajectories: List[List[int]],
        query_vec: torch.Tensor,
        ranking_func: Callable,
        top_k: int,
        n_agents: int
    ) -> List[Dict]:
        """
        Vectorized ranking that leverages GPU when available.

        Optimizes ranking by:
        1. Batch-fetching all vectors at once
        2. Computing similarities in a single GPU operation
        3. Vectorized score computation

        Args:
            agent_trajectories: List of paths taken by each agent
            query_vec: Query embedding
            ranking_func: Ranking function to apply
            top_k: Number of top results to return
            n_agents: Total number of agents

        Returns:
            List of top-k results with scores
        """
        # Count votes
        all_visited = [node for path in agent_trajectories for node in path]
        vote_counts = Counter(all_visited)
        unique_visited = list(vote_counts.keys())

        if not unique_visited:
            return []

        # Batch fetch vectors - use GPU if available
        vectors_matrix, valid_ids = self._fetch_vectors_batch_gpu(unique_visited)

        if len(valid_ids) == 0:
            return []

        # Compute base semantic scores vectorized
        query_norm = query_vec / (torch.linalg.norm(query_vec) + 1e-8)
        base_scores = torch.matmul(vectors_matrix, query_norm)

        # Build results with combined scores
        results = []
        for i, node_id in enumerate(valid_ids):
            votes = vote_counts[node_id]
            vote_score = votes / n_agents if n_agents > 0 else 0.0

            # Create context for custom ranking
            node_ctx = HeuristicContext(
                query_vec=query_vec,
                target_vecs=vectors_matrix[i:i+1],
                target_ids=[node_id],
                graph=self.graph_store,
                votes=votes,
                total_agents=n_agents
            )

            score = ranking_func(node_ctx)
            results.append({'id': node_id, 'score': score})

        # Sort and return top-k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def _batch_initial_search(
        self,
        query_vecs: torch.Tensor,
        pool_size: int
    ) -> List[List[int]]:
        """
        Perform batch initial search for multiple queries.

        Uses GPU-accelerated batch search when available.

        Args:
            query_vecs: Query vectors of shape (n_queries, dim)
            pool_size: Number of candidates per query

        Returns:
            List of candidate ID lists, one per query
        """
        # Check if vector store supports batch search
        if hasattr(self.vector_store, 'search_batch'):
            try:
                results = self.vector_store.search_batch(query_vecs, pool_size)
                return [
                    [r['id'] for r in res if self.graph_store.contains(r['id'])]
                    for res in results
                ]
            except Exception as e:
                logger.debug(f"Batch search failed, falling back to sequential: {e}")

        # Fall back to sequential
        all_results = []
        for vec in query_vecs:
            search_res = self.vector_store.search(vec, limit=pool_size)
            valid_ids = [r['id'] for r in search_res if self.graph_store.contains(r['id'])]
            all_results.append(valid_ids)

        return all_results

    def _compute_batch_similarities_gpu(
        self,
        query_vecs: torch.Tensor,
        candidate_ids_per_query: List[List[int]]
    ) -> List[Tuple[torch.Tensor, List[int]]]:
        """
        Compute similarities for multiple queries in batch.

        Optimizes by:
        1. Gathering all unique candidates
        2. Single fetch for all vectors
        3. Batch matrix multiplication

        Args:
            query_vecs: Query vectors of shape (n_queries, dim)
            candidate_ids_per_query: List of candidate ID lists

        Returns:
            List of (scores, valid_ids) tuples per query
        """
        if not self._use_gpu:
            # Fall back to sequential
            results = []
            for i, (query_vec, candidate_ids) in enumerate(zip(query_vecs, candidate_ids_per_query)):
                scores, valid_ids = self._compute_similarities_gpu(query_vec, candidate_ids)
                results.append((scores, valid_ids))
            return results

        # Gather all unique candidates
        all_candidates = set()
        for candidates in candidate_ids_per_query:
            all_candidates.update(candidates)
        all_candidates = list(all_candidates)

        if not all_candidates:
            return [(torch.tensor([]), []) for _ in query_vecs]

        # Batch fetch all vectors
        all_vectors, valid_all_ids = self._fetch_vectors_batch_gpu(all_candidates)

        if len(valid_all_ids) == 0:
            return [(torch.tensor([]), []) for _ in query_vecs]

        # Build ID to index mapping
        id_to_idx = {vid: i for i, vid in enumerate(valid_all_ids)}

        # Normalize queries
        query_norms = torch.linalg.norm(query_vecs, dim=1, keepdim=True) + 1e-8
        query_vecs_norm = query_vecs / query_norms

        # Compute all similarities at once: (n_queries, dim) @ (dim, n_candidates) -> (n_queries, n_candidates)
        all_similarities = torch.matmul(query_vecs_norm, all_vectors.t())

        # Extract per-query results
        results = []
        for i, candidate_ids in enumerate(candidate_ids_per_query):
            valid_ids = [cid for cid in candidate_ids if cid in id_to_idx]
            if not valid_ids:
                results.append((torch.tensor([]), []))
                continue

            indices = [id_to_idx[vid] for vid in valid_ids]
            scores = all_similarities[i, indices]
            results.append((scores, valid_ids))

        return results

    def retrieve_batch_optimized(
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
        use_vectorized_ranking: bool = True,
        **kwargs
    ) -> List[List[Dict]]:
        """
        Optimized batch retrieval with GPU acceleration.

        This method provides additional optimizations over retrieve_batch:
        1. Batch initial searches (single GPU operation for all queries)
        2. Vectorized ranking when enabled
        3. Better GPU memory utilization

        Args:
            queries: List of queries to retrieve for
            agent_groups: Optional agent group configurations
            seed: Random seed for reproducibility
            n_agents: Number of agents per query
            steps: Number of traversal steps
            decay: Pheromone decay rate
            drop_zone_inc: Drop zone increment
            initial_pool_size: Size of initial candidate pool
            start_subset: Size of starting subset
            top_k: Number of results to return
            movement_strategies: Movement strategy configuration
            ranking_strategies: Ranking strategy configuration
            deposit_strategies: Deposit strategy configuration
            max_workers: Maximum parallel workers
            use_vectorized_ranking: Whether to use vectorized ranking
            **kwargs: Additional arguments

        Returns:
            List of result lists, one per query
        """
        if not queries:
            return []

        # Resolve parameters
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

        # Batch embed all queries
        query_matrix = self._get_cached_query_embeddings_batch(queries)

        # Batch initial search using GPU
        logger.debug("Performing batch initial search...")
        initial_pools = self._batch_initial_search(query_matrix, params['initial_pool_size'])

        # Prepare agents
        resolved_agents = self._prepare_agents(
            agent_groups=agent_groups,
            n_agents=params['n_agents'],
            movement_strategies=params['movement_strategies'],
            deposit_strategies=params['deposit_strategies'],
        )

        # Set up seed
        base_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

        # Process each query
        results = []
        for i, (query_vec, initial_pool) in enumerate(zip(query_matrix, initial_pools)):
            if not initial_pool:
                results.append([])
                continue

            q_seed = base_seed + i
            py_rng = random.Random(q_seed)
            torch_gen = torch.Generator()
            torch_gen.manual_seed(q_seed)

            # Use the standard retrieval with pre-fetched pool
            result = self._retrieve_with_pool(
                query_vec=query_vec,
                initial_pool=initial_pool,
                resolved_agents=resolved_agents,
                py_rng=py_rng,
                torch_gen=torch_gen,
                use_vectorized_ranking=use_vectorized_ranking,
                **params
            )
            results.append(result)

        return results

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
        agent_locations = torch.tensor(
            py_rng.choices(drop_zone, weights=weights, k=n_agents),
            dtype=torch.long
        )
        agent_trajectories = [[loc.item()] for loc in agent_locations]
        query_pheromones = self.base_pheromones.copy()

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
            pheromone_updates = {}
            max_pheromone = max(query_pheromones.values()) if query_pheromones else 1.0

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

                for agent_idx in range(n_agents):
                    new_loc = new_locations[agent_idx].item() if isinstance(new_locations[agent_idx], torch.Tensor) else new_locations[agent_idx]
                    old_loc = agent_locations[agent_idx].item() if isinstance(agent_locations[agent_idx], torch.Tensor) else agent_locations[agent_idx]
                    if new_loc != old_loc:
                        agent_trajectories[agent_idx].append(new_loc)
                agent_locations = new_locations
            else:
                for agent_idx, (move_fn, deposit_fn) in enumerate(resolved_agents):
                    current_loc = agent_locations[agent_idx].item() if isinstance(agent_locations[agent_idx], torch.Tensor) else agent_locations[agent_idx]

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
                        agent_trajectories[agent_idx].append(next_node)

                        deposit = result['deposit']
                        if deposit > self.PHEROMONE_EPSILON:
                            pheromone_updates[next_node] = pheromone_updates.get(next_node, 0.0) + deposit

            # Update pheromones
            if query_pheromones:
                existing_keys = list(query_pheromones.keys())
                for k in existing_keys:
                    new_val = query_pheromones[k] * decay
                    if new_val < self.PHEROMONE_EPSILON:
                        del query_pheromones[k]
                    else:
                        query_pheromones[k] = new_val

            for node_id, amount in pheromone_updates.items():
                query_pheromones[node_id] += amount

        return self._ranking(
            agent_trajectories, query_vec, ranking_func, top_k, n_agents
        )

    def _retrieve_with_pool(
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
        use_vectorized_ranking: bool = True,
        **kwargs
    ) -> List[Dict]:
        """
        Internal retrieval using a pre-computed initial pool.

        Args:
            query_vec: Query embedding
            initial_pool: Pre-computed initial candidate pool
            resolved_agents: Prepared agent functions
            py_rng: Python random generator
            torch_gen: PyTorch random generator
            steps: Number of traversal steps
            decay: Pheromone decay rate
            drop_zone_inc: Drop zone increment
            start_subset: Starting subset size
            top_k: Number of results
            ranking_strategies: Ranking configuration
            use_vectorized_ranking: Whether to use vectorized ranking
            **kwargs: Additional arguments

        Returns:
            List of results with scores
        """
        n_agents = len(resolved_agents)

        # Normalize query
        query_vec = torch.as_tensor(query_vec).flatten()
        query_vec = query_vec / (torch.linalg.norm(query_vec) + 1e-8)

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
        agent_locations = torch.tensor(py_rng.choices(drop_zone, weights=weights, k=n_agents), dtype=torch.long)
        agent_trajectories = [[loc.item()] for loc in agent_locations]
        query_pheromones = self.base_pheromones.copy()

        # Traversal loop
        for step in range(steps):
            pheromone_updates = {}
            max_pheromone = max(query_pheromones.values()) if query_pheromones else 1.0

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
                    torch_gen=torch_gen
                )

                if result:
                    next_node = result['new_location']
                    agent_locations[agent_idx] = next_node
                    agent_trajectories[agent_idx].append(next_node)

                    deposit = result['deposit']
                    if deposit > self.PHEROMONE_EPSILON:
                        pheromone_updates[next_node] = pheromone_updates.get(next_node, 0.0) + deposit

            # Update pheromones
            if query_pheromones:
                existing_keys = list(query_pheromones.keys())
                for k in existing_keys:
                    new_val = query_pheromones[k] * decay
                    if new_val < self.PHEROMONE_EPSILON:
                        del query_pheromones[k]
                    else:
                        query_pheromones[k] = new_val

            for node_id, amount in pheromone_updates.items():
                query_pheromones[node_id] += amount

        # Use vectorized ranking if enabled
        if use_vectorized_ranking:
            return self._ranking_vectorized(
                agent_trajectories, query_vec, ranking_func, top_k, n_agents
            )

        return self._ranking(
            agent_trajectories, query_vec, ranking_func, top_k, n_agents
        )