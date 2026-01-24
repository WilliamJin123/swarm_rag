import random
import numpy as np
from typing import Any, List, Dict, Optional, Sequence, Tuple, TypedDict, Callable, Union
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging
from ..utils import LRUCache, get_device, to_numpy

from .heuristics import HeuristicRegistry, Heuristics, HeuristicContext
from ..interfaces.abstract_classes import VectorStore, GraphStore, EmbeddingProvider, Matrix

# Optional torch import for GPU operations
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

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
        use_gpu: bool = True,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embed_fn = embedding_provider
        self.base_pheromones = defaultdict(float)

        self.py_rng = random.Random(seed) if seed else random
        self.np_rng = np.random.default_rng(seed) if seed else np.random

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
        self._use_gpu = use_gpu and _TORCH_AVAILABLE and get_device() == "cuda"
        self._has_gpu_store = hasattr(vector_store, 'compute_similarities') and hasattr(vector_store, 'is_gpu')

        if self._use_gpu and self._has_gpu_store and getattr(vector_store, 'is_gpu', False):
            self._device = getattr(vector_store, 'device', 'cuda')
            logger.info(f"SwarmRetriever: GPU acceleration enabled on {self._device}")
        else:
            self._device = "cpu"
            self._use_gpu = False
            if use_gpu:
                logger.debug("SwarmRetriever: GPU not available, using CPU")

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
            np_rng = np.random.default_rng(seed)
            # Also update instance RNGs for sequential consistency if needed later
            self.py_rng = py_rng
            self.np_rng = np_rng
        else:
            py_rng = self.py_rng
            np_rng = self.np_rng
        
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
            np_rng=np_rng,
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
        query_vectors: Sequence[np.ndarray],
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
            np_rng = np.random.default_rng(q_seed)

            result = self._retrieve(
                query_vec=vec, 
                resolved_agents=resolved_agents,
                py_rng=py_rng,
                np_rng=np_rng,
                **kwargs
            )
            results.append(result)
        return results

    def _retrieve_batch_parallel(
        self,
        query_vectors: Sequence[np.ndarray],
        resolved_agents: List[Tuple[Callable, Callable]],
        max_workers: int,
        base_seed: int,
        **kwargs
    ) -> List[List[Dict]]:
        """Process queries in parallel with controlled concurrency and determinism."""
        
        def process(idx: int, vec: np.ndarray) -> tuple[int, List[Dict]]:
            # Isolated RNGs for thread safety and determinism
            task_seed = base_seed + idx
            task_py_rng = random.Random(task_seed)
            task_np_rng = np.random.default_rng(task_seed)

            result = self._retrieve(
                query_vec=vec, 
                resolved_agents=resolved_agents,
                py_rng=task_py_rng,
                np_rng=task_np_rng,
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
        query_vec: np.ndarray,
        resolved_agents: List[Tuple[Callable, Callable]],
        steps: int,
        decay: float,
        drop_zone_inc: float,
        initial_pool_size: int,
        start_subset: int,
        top_k: int,
        ranking_strategies: Dict,
        py_rng: random.Random,
        np_rng: np.random.Generator,
        decision_tracker: Optional[Any] = None,  # DecisionTracker for LLM context
        **kwargs  # Catch unused global params
    ) -> List[Dict]:
        """
        Core retrieval logic shared between retrieve() and retrieve_batch().
        """  

        n_agents = len(resolved_agents)

        # Normalize and Flatten
        query_vec = np.asarray(query_vec)
        query_vec = query_vec.flatten()
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

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
        weights = [1.0 + drop_zone_inc * (dz_len - i - 1)  for i in range(dz_len)]
        agent_locations = np.array(py_rng.choices(drop_zone, weights=weights, k=n_agents))
        agent_trajectories = [[loc] for loc in agent_locations]
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
                    np_rng=np_rng,
                )

                # Update locations and trajectories
                for agent_idx in range(n_agents):
                    if new_locations[agent_idx] != agent_locations[agent_idx]:
                        agent_trajectories[agent_idx].append(new_locations[agent_idx])
                agent_locations = new_locations

            else:
                # Original sequential processing
                for agent_idx, (move_fn, deposit_fn) in enumerate(resolved_agents):
                    current_loc = agent_locations[agent_idx]

                    result = self._process_agent_step(
                        agent_id=agent_idx,
                        current_loc=current_loc,
                        query_vec=query_vec,
                        query_pheromones=query_pheromones,
                        move_func=move_fn,
                        deposit_func=deposit_fn,
                        step=step,
                        max_pheromone=max_pheromone,
                        np_rng=np_rng,
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
                    # Optimize for common case: numpy array of size 1 or scalar
                    if isinstance(val, np.ndarray):
                        if val.size == 1:
                            total += val.item() * w
                        else:
                            # Fallback if heuristic returns full array (rare for deposit on single node)
                            total += np.sum(val) * w
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
            def combined_movement(ctx: HeuristicContext) -> np.ndarray:
                if not components:
                    return np.array([]) 
                
                # Unroll first iteration to init accumulator with correct shape/type
                fn0, w0 = components[0]
                total_scores = fn0(ctx) * w0
                
                for i in range(1, len(components)):
                    func, w = components[i]
                    total_scores += func(ctx) * w
                
                return total_scores
            return combined_movement

        return lambda ctx: 0.0 # Fallback

    def _process_agent_step(
        self,
        agent_id: int,
        current_loc: int,
        query_vec: np.ndarray,
        query_pheromones: Dict,
        move_func: Callable,
        deposit_func: Callable,
        step: int,
        max_pheromone: float,
        np_rng: np.random.Generator,
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
        p_vals = np.array([query_pheromones.get(nid, 0.0) for nid in valid_ids])

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

        total_scores = np.atleast_1d(np.maximum(total_scores, 0.001))

        # Ensure total_scores matches valid_ids length (broadcast scalar if needed)
        if len(total_scores) == 1 and len(valid_ids) > 1:
            total_scores = np.full(len(valid_ids), total_scores[0])
        elif len(total_scores) != len(valid_ids):
            logger.warning(f"Score mismatch: {len(total_scores)} scores vs {len(valid_ids)} candidates")
            return None

        if len(valid_ids) > 5:
            total_scores[total_scores < 0.01] = 0.0

        if np.sum(total_scores) == 0:
            return None

        # Selection
        probs = total_scores / np.sum(total_scores)
        chosen_idx = int(np_rng.choice(len(valid_ids), p=probs))
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
        agent_locations: np.ndarray,
        query_vec: np.ndarray,
        query_pheromones: Dict,
        resolved_agents: List[Tuple[Callable, Callable]],
        step: int,
        max_pheromone: float,
        np_rng: np.random.Generator,
    ) -> Tuple[np.ndarray, Dict[int, float]]:
        """
        Process all agents in one batched GPU operation.

        This method provides significant speedup by:
        1. Batch fetching all neighbors for all agents in single GPU call
        2. Batch fetching all embeddings in single GPU call
        3. Computing all similarities in one matrix operation
        4. Vectorized softmax selection

        Args:
            agent_locations: Array of current agent positions
            query_vec: Query embedding vector
            query_pheromones: Current pheromone map
            resolved_agents: List of (move_fn, deposit_fn) tuples
            step: Current step index
            max_pheromone: Maximum pheromone value
            np_rng: NumPy random generator

        Returns:
            Tuple of (new_locations array, pheromone_updates dict)
        """
        import torch

        n_agents = len(agent_locations)
        device = self._device

        # Convert positions to tensor
        positions = torch.tensor(agent_locations, device=device, dtype=torch.long)

        # Batch fetch all neighbors using GPU graph store
        all_neighbors, neighbor_mask = self.graph_store.get_neighbors_batch(positions)
        # all_neighbors: (n_agents, max_degree), neighbor_mask: (n_agents, max_degree)

        if all_neighbors is None or neighbor_mask is None:
            # Fallback to sequential processing if batch not supported
            return self._step_agents_sequential_fallback(
                agent_locations, query_vec, query_pheromones,
                resolved_agents, step, max_pheromone, np_rng
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
            unique_neighbors_cpu = unique_neighbors_gpu.cpu().numpy()
            unique_embs_np, valid_unique_ids_list = self._fetch_vectors_batch(unique_neighbors_cpu)
            unique_embs = torch.tensor(unique_embs_np, device=device, dtype=torch.float32)
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
        avg_log_degree = np.log(1 + self.avg_degree)
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

        # Single CPU transfer at the end for the final locations
        new_locations = new_locations_tensor.cpu().numpy()

        # Compute pheromone updates
        pheromone_updates = {}
        deposit_amount = 1.0

        # Only process agents that actually moved
        moved_mask = valid_agent_mask.cpu().numpy()
        new_locs_list = new_locations.tolist()

        for agent_idx in range(n_agents):
            if moved_mask[agent_idx]:
                next_node = new_locs_list[agent_idx]
                if deposit_amount > self.PHEROMONE_EPSILON:
                    pheromone_updates[next_node] = pheromone_updates.get(next_node, 0.0) + deposit_amount

        return new_locations, pheromone_updates

    def _step_agents_sequential_fallback(
        self,
        agent_locations: np.ndarray,
        query_vec: np.ndarray,
        query_pheromones: Dict,
        resolved_agents: List[Tuple[Callable, Callable]],
        step: int,
        max_pheromone: float,
        np_rng: np.random.Generator,
    ) -> Tuple[np.ndarray, Dict[int, float]]:
        """
        Fallback to sequential agent processing when batch mode unavailable.
        """
        new_locations = agent_locations.copy()
        pheromone_updates = {}

        for agent_idx, (move_fn, deposit_fn) in enumerate(resolved_agents):
            current_loc = agent_locations[agent_idx]

            result = self._process_agent_step(
                agent_id=agent_idx,
                current_loc=current_loc,
                query_vec=query_vec,
                query_pheromones=query_pheromones,
                move_func=move_fn,
                deposit_func=deposit_fn,
                step=step,
                max_pheromone=max_pheromone,
                np_rng=np_rng,
            )

            if result:
                next_node = result['new_location']
                new_locations[agent_idx] = next_node

                deposit = result['deposit']
                if deposit > self.PHEROMONE_EPSILON:
                    pheromone_updates[next_node] = pheromone_updates.get(next_node, 0.0) + deposit

        return new_locations, pheromone_updates

    def _capture_heuristic_scores(self, ctx: HeuristicContext) -> Dict[str, np.ndarray]:
        """
        Capture individual heuristic scores for decision analysis.

        This method computes individual heuristic contributions separately
        to provide insight into agent decision-making. Used only when
        decision tracking is enabled.
        """
        scores = {}
        try:
            scores["semantic_similarity"] = np.array(Heuristics.semantic_similarity(ctx))
        except Exception:
            pass
        try:
            scores["node_centrality"] = np.array(Heuristics.node_centrality(ctx))
        except Exception:
            pass
        try:
            scores["pheromone_repulsion"] = np.array(Heuristics.pheromone_repulsion(ctx))
        except Exception:
            pass
        return scores
    
    def _ranking(
        self, 
        agent_trajectories: List[List[int]], 
        query_vec: np.ndarray,
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
        query_vec: np.ndarray,
        target_vec: Optional[np.ndarray],
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

    def _get_cached_neighbors(self, node_id: int) -> np.ndarray:
        """Gets or computes and caches the neighbor list, if enabled."""
        if not self.cache_neighbors:
            return self.graph_store.get_neighbors(node_id)
        with self._neighbor_lock:
            cached = self.neighbor_cache.get(node_id)
        if cached is not None:
            return cached

        neighbors = self.graph_store.get_neighbors(node_id)
        with self._neighbor_lock:
            self.neighbor_cache.set(node_id, np.array(neighbors))
            self.degree_cache.set(node_id, len(neighbors))
        return neighbors
    
    def _fetch_vectors_batch(self, node_ids: Sequence[int]) -> Tuple[np.ndarray, List[int]]:
        """
        Fetches vectors efficiently using Two-Phase Fetch (Read Locked -> Fetch Unlocked -> Write Locked).
        """
        if not self.cache_vectors:
            matrix = self.vector_store.fetch_batch(node_ids)
            matrix = np.asarray(matrix)
            valid_mask = ~np.isnan(matrix).any(axis=1)
            
            if np.all(valid_mask):
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
            fetched_matrix = np.asarray(fetched_matrix)
            valid_fetched_mask = ~np.isnan(fetched_matrix).any(axis=1)
            
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
            return np.array([]), []
            
        valid_ids, valid_vecs = zip(*valid_data)
        return np.stack(valid_vecs), list(valid_ids)
        
    def _fetch_degrees_batch(self, node_ids: Sequence[int]) -> np.ndarray:
        """
        Fetches degrees efficiently using Two-Phase Fetch.
        Returns array of degrees (int32).
        """
        if not self.cache_neighbors:
            # If caching is disabled, we must fetch neighbors to count them
            # We assume get_neighbors is relatively fast or necessary
            return np.array([len(self._get_cached_neighbors(nid)) for nid in node_ids], dtype=np.int32)
        
        degrees = np.empty(len(node_ids), dtype=np.int32)
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
                    # nid check matches order because we append consistently
                    d = len(nb)
                    # Cache both the neighbors and the degree since we paid the cost
                    self.neighbor_cache.set(nid, np.array(nb))
                    self.degree_cache.set(nid, d)
                    degrees[i] = d
                    
        return degrees

    def _get_cached_query_vector(self, query: Any) -> np.ndarray:
        """Gets or computes and caches the query embedding, if enabled."""
        if not self.cache_vectors:
            return np.asarray(self.embed_fn.embed_query(query))
        with self._query_lock:
            cached = self.query_cache.get(query)
        if cached is not None:
            return cached

        emb = np.asarray(self.embed_fn.embed_query(query))
        with self._query_lock:
            self.query_cache.set(query, emb)
        return emb
        
    def _get_cached_query_embeddings_batch(self, queries: list) -> Matrix:
        """
        Retrieves embeddings for a batch of queries, returning a single 2D array.
        """
        if not queries:
            return np.array([]) 
        if not self.cache_vectors:
            return np.asarray(self.embed_fn.embed_query_batch(queries))

        results_by_index: Dict[Any, np.ndarray] = {}
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
            batch_embeddings = np.asarray(batch_embeddings)
            with self._query_lock:
                for i, emb in zip(missing_indices, batch_embeddings):
                    q = queries[i]
                    self.query_cache.set(q, emb)
                    results_by_index[i] = emb

        if not results_by_index:
            return np.array([])
        
        first_embedding = next(iter(results_by_index.values()))
        embedding_dim = first_embedding.shape[0]
        batch_size = len(queries)

        final_embeddings = np.empty((batch_size, embedding_dim), dtype=first_embedding.dtype)

        for i in range(batch_size):
            final_embeddings[i, :] = results_by_index[i]

        return final_embeddings

    def _compute_similarities_gpu(
        self,
        query_vec: np.ndarray,
        candidate_ids: Sequence[int]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Compute similarities using GPU when available.

        Falls back to standard numpy computation if GPU not available.

        Args:
            query_vec: Query embedding (numpy array)
            candidate_ids: List of candidate document IDs

        Returns:
            Tuple of (similarity scores array, valid_ids list)
        """
        # Try GPU path if available
        if self._use_gpu and self._has_gpu_store:
            try:
                scores, valid_ids = self.vector_store.compute_similarities(
                    query_vec, list(candidate_ids)
                )
                # Convert to numpy if tensor
                if _TORCH_AVAILABLE and isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                return scores, valid_ids
            except Exception as e:
                logger.debug(f"GPU similarity computation failed, falling back to CPU: {e}")

        # CPU fallback
        candidate_matrix, valid_ids = self._fetch_vectors_batch(candidate_ids)
        if len(valid_ids) == 0:
            return np.array([]), []

        # Normalize query
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        # Compute cosine similarity
        scores = np.dot(candidate_matrix, query_norm)
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
        query_vec: np.ndarray,
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

        # Convert to numpy for ranking if it's a tensor
        if _TORCH_AVAILABLE and isinstance(vectors_matrix, torch.Tensor):
            vectors_np = to_numpy(vectors_matrix)
        else:
            vectors_np = np.asarray(vectors_matrix)

        # Compute base semantic scores vectorized
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        base_scores = np.dot(vectors_np, query_norm)

        # Build results with combined scores
        results = []
        for i, node_id in enumerate(valid_ids):
            votes = vote_counts[node_id]
            vote_score = votes / n_agents if n_agents > 0 else 0.0

            # Create context for custom ranking
            node_ctx = HeuristicContext(
                query_vec=query_vec,
                target_vecs=vectors_np[i:i+1],
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
        query_vecs: np.ndarray,
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
        query_vecs: np.ndarray,
        candidate_ids_per_query: List[List[int]]
    ) -> List[Tuple[np.ndarray, List[int]]]:
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
            return [(np.array([]), []) for _ in query_vecs]

        # Batch fetch all vectors
        all_vectors, valid_all_ids = self._fetch_vectors_batch_gpu(all_candidates)

        if len(valid_all_ids) == 0:
            return [(np.array([]), []) for _ in query_vecs]

        # Build ID to index mapping
        id_to_idx = {vid: i for i, vid in enumerate(valid_all_ids)}

        # Convert vectors for computation
        if _TORCH_AVAILABLE and isinstance(all_vectors, torch.Tensor):
            all_vectors_np = to_numpy(all_vectors)
        else:
            all_vectors_np = np.asarray(all_vectors)

        # Normalize queries
        query_norms = np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-8
        query_vecs_norm = query_vecs / query_norms

        # Compute all similarities at once: (n_queries, dim) @ (dim, n_candidates) -> (n_queries, n_candidates)
        all_similarities = np.dot(query_vecs_norm, all_vectors_np.T)

        # Extract per-query results
        results = []
        for i, candidate_ids in enumerate(candidate_ids_per_query):
            valid_ids = [cid for cid in candidate_ids if cid in id_to_idx]
            if not valid_ids:
                results.append((np.array([]), []))
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
            np_rng = np.random.default_rng(q_seed)

            # Use the standard retrieval with pre-fetched pool
            result = self._retrieve_with_pool(
                query_vec=query_vec,
                initial_pool=initial_pool,
                resolved_agents=resolved_agents,
                py_rng=py_rng,
                np_rng=np_rng,
                use_vectorized_ranking=use_vectorized_ranking,
                **params
            )
            results.append(result)

        return results

    def _retrieve_with_pool(
        self,
        query_vec: np.ndarray,
        initial_pool: List[int],
        resolved_agents: List[Tuple[Callable, Callable]],
        py_rng: random.Random,
        np_rng: np.random.Generator,
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
            np_rng: NumPy random generator
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
        query_vec = np.asarray(query_vec).flatten()
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

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
        agent_locations = np.array(py_rng.choices(drop_zone, weights=weights, k=n_agents))
        agent_trajectories = [[loc] for loc in agent_locations]
        query_pheromones = self.base_pheromones.copy()

        # Traversal loop
        for step in range(steps):
            pheromone_updates = {}
            max_pheromone = max(query_pheromones.values()) if query_pheromones else 1.0

            for agent_idx, (move_fn, deposit_fn) in enumerate(resolved_agents):
                current_loc = agent_locations[agent_idx]

                result = self._process_agent_step(
                    agent_id=agent_idx,
                    current_loc=current_loc,
                    query_vec=query_vec,
                    query_pheromones=query_pheromones,
                    move_func=move_fn,
                    deposit_func=deposit_fn,
                    step=step,
                    max_pheromone=max_pheromone,
                    np_rng=np_rng
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