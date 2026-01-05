import random
import numpy as np
from typing import Any, List, Dict, Optional, Tuple, TypedDict
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging
from ..utils import LRUCache

from .heuristics import HeuristicRegistry, Heuristics, HeuristicContext
from ..interfaces.abstract_classes import VectorStore, GraphStore, EmbeddingProvider

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
            "visited": ("percentage_visited", 0.6),
            "semantic": ("semantic_rank", 0.4),
        },
        deposit_strategies={
            "flat_mark": ("flat", 1.0),
        },
    )
    def __init__(
        self, 
        vector_store: VectorStore, 
        graph_store: GraphStore, 
        embedding_provider: EmbeddingProvider,
        deterministic=False, seed=0,
        cache_neighbors: bool = False,
        neighbor_cache_size: int = 5000,
        cache_vectors: bool = True,
        doc_cache_size: int = 50000,
        query_cache_size: int = 1000,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embed_fn = embedding_provider
        self.base_pheromones = defaultdict(float)
        
        self.deterministic = deterministic
        self.seed = seed

        self._cache_lock = Lock()

        # Performance optimizations
        self.cache_neighbors = cache_neighbors
        if self.cache_neighbors:
            self.neighbor_cache = LRUCache(neighbor_cache_size)

        self.cache_vectors = cache_vectors
        if self.cache_vectors:
            self.doc_cache = LRUCache(doc_cache_size)
            self.query_cache = LRUCache(query_cache_size)

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
            deterministic: Optional[bool] = None,
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
            deposit_strategies: Optional[Dict] = None
        ) -> List[Dict]:
            
            current_deterministic = deterministic if deterministic is not None else self.deterministic
            current_seed = seed if seed is not None else self.seed

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

            resolved_groups = self._prepare_groups(
                agent_groups=agent_groups,
                n_agents=params['n_agents'], 
                movement_strategies=params['movement_strategies'],
                deposit_strategies=params['deposit_strategies'],
            )

            query_vec = self._get_cached_query_vector(query)

            return self._retrieve(
                query_vec=query_vec, 
                resolved_groups=resolved_groups,
                deterministic=current_deterministic,
                seed=current_seed,
                **params)

    def retrieve_batch(
        self,
        queries: List[Any],
        agent_groups: Optional[List[AgentGroupConfig]] = None,
        deterministic: Optional[bool] = None,
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
        max_workers: Optional[int] = 4
    ) -> List[List[Dict]]:
        """
        Hybrid batch retrieval that intelligently chooses between sequential and parallel processing.
        
        Args:
            queries: List of queries to process
            parallel_queries: Whether to enable parallel query processing
            max_workers: Max concurrent queries (auto-calculated if None)
            Other args: Same as retrieve()
        
        Returns:
            List of result lists (one per query)
        """

        if not queries: return []
        
        current_deterministic = deterministic if deterministic is not None else self.deterministic
        current_seed = seed if seed is not None else self.seed

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

        resolved_groups = self._prepare_groups(
            agent_groups=agent_groups,
            n_agents=params['n_agents'], 
            movement_strategies=params['movement_strategies'],
            deposit_strategies=params['deposit_strategies'],
        )

        # Batch embed all queries
        query_vectors = self._get_cached_query_embeddings_batch(queries)

        # Decide processing strategy
        if max_workers > 1 and len(queries) > 1:
            return self._retrieve_batch_parallel(
                query_vectors,
                resolved_groups,
                max_workers=max_workers,
                deterministic=current_deterministic,
                seed=current_seed,
                **params
            )
        else:
            return self._retrieve_batch_sequential(
                query_vectors,
                resolved_groups,
                deterministic=current_deterministic,
                seed=current_seed,
                **params
            )

    def _retrieve_batch_sequential(
        self,
        query_vectors: List[np.ndarray],
        resolved_groups: List[Dict],
        deterministic: bool,
        seed: int,
        **kwargs
    ) -> List[List[Dict]]:
        """Process queries sequentially."""
        results = []
        for vec in query_vectors:
            result = self._retrieve(
                query_vec=vec, 
                resolved_groups=resolved_groups,
                deterministic=deterministic, 
                seed=seed, 
                **kwargs
            )
            results.append(result)
        return results

    def _retrieve_batch_parallel(
        self,
        query_vectors: List[np.ndarray],
        resolved_groups: List[Dict],
        deterministic: bool,
        seed: int,
        max_workers: int,
        **kwargs
    ) -> List[List[Dict]]:
        """Process queries in parallel with controlled concurrency."""
        
        def process(idx: int, vec: np.ndarray) -> tuple[int, List[Dict]]:
            result = self._retrieve(
                query_vec=vec, 
                resolved_groups=resolved_groups,
                deterministic=deterministic, 
                seed=seed + idx, 
                **kwargs
                )
            return idx, result
        
        # Process queries in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with their indices
            futures_to_index = {
                executor.submit(process, i, vec): i
                for i, vec in enumerate(query_vectors)
            }
            
            # Collect results maintaining order
            results = [None] * len(query_vectors)
            for future in as_completed(futures_to_index):
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
        resolved_groups: List[Dict],
        deterministic: bool,
        seed: int,
        steps: int,
        decay: float,      
        drop_zone_inc: float,   
        initial_pool_size: int,
        start_subset: int,
        top_k: int,
        ranking_strategies: Dict,
        **kwargs # Catch unused global params
    ) -> List[Dict]:
        """
        Core retrieval logic shared between retrieve() and retrieve_batch().
        """  

        n_agents = sum(g['count'] for g in resolved_groups)

        if deterministic:
            py_rng = random.Random(seed)
            np_rng = np.random.default_rng(seed)
        else:
            py_rng = random
            np_rng = np.random

        # Normalize and Flatten
        query_vec = query_vec.flatten()
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        ranking_funcs = self._resolve_strategy_funcs(ranking_strategies, "ranking")

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

        # --- TRAVERSAL LOOP ---
        for step in range(steps):
            pheromone_updates = {}
            max_pheromone = max(query_pheromones.values()) if query_pheromones else 1.0

            current_agent_idx = 0
            for group in resolved_groups:
                count = group['count']
                move_funcs = group['movement_funcs']
                dep_funcs = group['deposit_funcs']
                end_agent_idx = current_agent_idx + count

                for i in range(current_agent_idx, end_agent_idx):
                    
                    current_loc = agent_locations[i] 
                    
                    result = self._process_agent_step(
                        agent_id=i,
                        current_loc=current_loc,
                        query_vec=query_vec,
                        query_pheromones=query_pheromones,
                        movement_funcs=move_funcs, 
                        deposit_funcs=dep_funcs, 
                        step=step,
                        max_pheromone=max_pheromone,
                        np_rng=np_rng
                    )

                    if result:
                        next_node = result['new_location']
                        agent_locations[i] = next_node 
                        agent_trajectories[i].append(next_node)
                        
                        deposit = result['deposit']
                        if deposit > 0.001:
                            pheromone_updates[next_node] = pheromone_updates.get(next_node, 0.0) + deposit

                current_agent_idx = end_agent_idx

            # Batch update pheromones
            # Apply decay
            for k in query_pheromones:
                query_pheromones[k] *= decay

            # Then add new deposits
            for node_id, amount in pheromone_updates.items():
                query_pheromones[node_id] += amount
            
        return self._ranking(
            agent_trajectories, 
            query_vec, 
            ranking_funcs, 
            top_k,
            n_agents
        )          

    # === HELPERS ===

    def _prepare_groups(
        self,
        agent_groups: Optional[List[AgentGroupConfig]],
        n_agents: int,
        movement_strategies: Dict,
        deposit_strategies: Dict
    ) -> List[Dict]:
        """
        Resolves strategies for groups ONCE.
        Returns a minimal list of dicts: [{'count': 5, 'move': func, 'dep': func}, ...]
        """
        resolved_groups = []

        if agent_groups:
            for group in agent_groups:
                count = group.get('count', 1)
                if count <= 0: continue
                resolved_groups.append({
                    'count': count,
                    'movement_funcs': self._resolve_strategy_funcs(group.get('movement_strategies'), "movement"),
                    'deposit_funcs': self._resolve_strategy_funcs(group.get('deposit_strategies'), "deposit")
                })
        else:
            # Homogeneous fallback
            resolved_groups.append({
                'count': n_agents,
                'movement_funcs': self._resolve_strategy_funcs(movement_strategies, "movement"),
                'deposit_funcs': self._resolve_strategy_funcs(deposit_strategies, "deposit")
            })
            
        return resolved_groups

    def _resolve_strategy_funcs(
        self, 
        strategy_dict: Dict, 
        strategy_type: str
    ) -> list[tuple]:
        """
        Resolves a dict of strategies to actual callable functions.
        Supports:
        - Function references (old way)
        - Names registered in HeuristicRegistry (new way)
        """
        
        resolved = []
        for key, (fn_or_name, weight) in strategy_dict.items():
            if callable(fn_or_name):
                # Still support direct function references for flexibility
                resolved.append((fn_or_name, weight))
            elif isinstance(fn_or_name, str):
                # Use the appropriate registry based on strategy type
                if strategy_type == "movement":
                    resolved.append((HeuristicRegistry.get_movement(fn_or_name), weight))
                elif strategy_type == "ranking":
                    resolved.append((HeuristicRegistry.get_ranking(fn_or_name), weight))
                elif strategy_type == "deposit":
                    resolved.append((HeuristicRegistry.get_deposit(fn_or_name), weight))
                else:
                    raise ValueError(f"Unknown strategy type: {strategy_type}")
            else:
                raise TypeError(f"Invalid heuristic entry: {fn_or_name}")
        return resolved

    def _process_agent_step(
        self, 
        agent_id: int,
        current_loc: int,
        query_vec: np.ndarray,
        query_pheromones: Dict,
        movement_funcs: List[tuple],
        deposit_funcs: List[tuple],
        step: int,
        max_pheromone: float,
        np_rng: np.random.Generator
    ) -> Optional[Dict]:
        """Vectorized agent step processing."""
        neighbors = self._get_cached_neighbors(current_loc)
        if not neighbors:
            return None
        if step % 2 == 0:
            logger.debug(f"Agent {agent_id} at {current_loc} (degree={len(neighbors)})")
        
        # Fetch Matrix & IDs
        candidate_matrix, valid_ids = self._fetch_vectors_batch(neighbors)
        if len(valid_ids) == 0:
            return None
        # Prefetch Vectorization Metadata
        p_vals = np.array([query_pheromones.get(nid, 0.0) for nid in valid_ids])
        degrees = np.array([len(self._get_cached_neighbors(nid)) for nid in valid_ids])
        
        avg_degree = self.graph_store.get_avg_degree()

        ctx = HeuristicContext(
            query_vec=query_vec,
            target_vecs=candidate_matrix,
            target_ids=valid_ids,
            pheromone_values=p_vals,
            node_degrees=degrees,
            max_pheromone=max_pheromone,
            avg_degree=avg_degree,
            step_index=step,
            agent_index=agent_id,
            graph=self.graph_store
        )

        # Calculate weighted scores
        total_scores = np.zeros(len(valid_ids), dtype=np.float32)
        for func, weight in movement_funcs:
            scores = func(ctx)
            total_scores += scores * weight
        total_scores = np.maximum(total_scores, 0.001)
        
        if len(valid_ids) > 5:
             total_scores[total_scores < 0.01] = 0.0
             
        if np.sum(total_scores) == 0:
            return None

        # Selection
        probs = total_scores / np.sum(total_scores)
        chosen_idx = np_rng.choice(len(valid_ids), p=probs)
        next_node = valid_ids[chosen_idx]
        
        # Calculate deposit
        ctx.target_vecs = candidate_matrix[chosen_idx : chosen_idx+1]
        ctx.target_ids = [next_node]
        ctx.pheromone_values = p_vals[chosen_idx : chosen_idx+1]
        ctx.node_degrees = degrees[chosen_idx : chosen_idx+1]
        
        # Summing arrays of shape (1,)
        deposit_array = np.zeros(1)
        for func, weight in deposit_funcs:
             deposit_array += func(ctx) * weight

        # Explicitly extract the float for our python dict
        deposit_amount = deposit_array.item()
        
        return {
            'new_location': next_node,
            'node_id': next_node,
            'deposit': deposit_amount
        }
    
    def _ranking(
        self, 
        agent_trajectories: List[List[int]], 
        query_vec: np.ndarray,
        ranking_funcs: List[tuple],
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
                ranking_funcs=ranking_funcs,
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
        ranking_funcs: List[tuple],
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
        
        return sum(
            func(node_ctx) * weight 
            for func, weight in ranking_funcs
        )

    def _get_cached_neighbors(self, node_id: int) -> List[int]:
        """Gets or computes and caches the neighbor list, if enabled."""
        if not self.cache_neighbors:
            return self.graph_store.get_neighbors(node_id)
        with self._cache_lock:
            cached = self.neighbor_cache.get(node_id)
        if cached is not None:
            return cached

        neighbors = self.graph_store.get_neighbors(node_id)
        with self._cache_lock:
            self.neighbor_cache.set(node_id, neighbors)
        return neighbors
    
    def _fetch_vectors_batch(self, node_ids: List[int]) -> Tuple[np.ndarray, List[int]]:
        """
        Fetches vectors efficiently and returns a dense matrix of foundn vectors.
        
        Returns:
            matrix: np.ndarray of shape (N_found, Dimension)
            valid_ids: List[int] of length N_found, mapping rows to node IDs.
        """
        if not self.cache_vectors:
            raw_vecs = self.vector_store.fetch_batch(node_ids)
        else:
            raw_vecs = [None] * len(node_ids)
            missing_indices = []
            missing_ids = []

            with self._cache_lock:
                for i, node_id in enumerate(node_ids):   
                    cached_vec = self.doc_cache.get(node_id)
                    if cached_vec is not None:
                        raw_vecs[i] = cached_vec
                    else:
                        missing_indices.append(i)
                        missing_ids.append(node_id)
            
            if missing_ids:
                fetched_vecs = self.vector_store.fetch_batch(missing_ids)
                with self._cache_lock:
                    for i, vec in zip(missing_indices, fetched_vecs):
                        if vec is not None:
                            self.doc_cache.set(node_ids[i], vec)
                        raw_vecs[i] = vec
            
        valid_data = [(nid, v) for nid, v in zip(node_ids, raw_vecs) if v is not None]
        
        if not valid_data:
            return np.array([]), []
            
        valid_ids, valid_vecs = zip(*valid_data)
        
        # Stack into (N, D) matrix
        matrix = np.stack(valid_vecs)
        return matrix, list(valid_ids)

    def _get_cached_query_vector(self, query: Any) -> np.ndarray:
        """Gets or computes and caches the query embedding, if enabled."""
        if not self.cache_vectors:
            return self.embed_fn.embed_query(query)
        with self._cache_lock:
            cached = self.query_cache.get(query)
        if cached is not None:
            return cached

        emb = self.embed_fn.embed_query(query)
        with self._cache_lock:
            self.query_cache.set(query, emb)
        return emb
        
    def _get_cached_query_embeddings_batch(self, queries: list) -> List[np.ndarray]:
        """
        Retrieves embeddings for a batch of queries, using the unified
        single-item cache to avoid redundant computations.
        """
        if not self.cache_vectors or not queries:
            return self.embed_fn.embed_query_batch(queries)

        results_by_index = {}
        missing_indices = []
        missing_queries = []
        with self._cache_lock:
            for i, q in enumerate(queries):
                cached_vec = self.query_cache.get(q)
                if cached_vec is not None:
                    results_by_index[i] = cached_vec
                else:
                    missing_indices.append(i)
                    missing_queries.append(q)

        if missing_queries:
            batch_embeddings = self.embed_fn.embed_query_batch(missing_queries)
            with self._cache_lock:
                for i, emb in zip(missing_indices, batch_embeddings):
                    q = queries[i]
                    self.query_cache.set(q, emb)
                    results_by_index[i] = emb

        return [results_by_index[i] for i in range(len(queries))]