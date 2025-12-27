import random
import numpy as np
from typing import Any, List, Dict, Optional
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import os
import psutil
from ..utils import LRUCache

from .heuristics import Heuristics, HeuristicContext
from ..interfaces.base import VectorStore, GraphStore, EmbeddingProvider

class SwarmRetriever:
    _DEFAULT_PARAMS = dict(
        n_agents=20,
        steps=4,
        decay=0.5,
        initial_pool_size=30,
        start_subset=10,
        top_k=20,
        movement_strategies={
            "semantic": (Heuristics.semantic_similarity, 0.3),
            "centrality": (Heuristics.node_centrality, 0.4),
            "diversity": (Heuristics.pheromone_repulsion, 0.3),
        },
        ranking_strategies={
            "visited": (Heuristics.percentage_visited, 0.6),
            "semantic": (Heuristics.semantic_rank, 0.4),
        },
        deposit_strategies={
            "flat_mark": (Heuristics.deposit_flat, 1.0),
        },
    )
    def __init__(
        self, 
        vector_store: VectorStore, 
        graph_store: GraphStore, 
        embedding_provider: EmbeddingProvider,
        max_workers: int = 16,
        cache_neighbors: bool = True,
        neighbor_cache_size: int = 5000,
        cache_vectors: bool = True,
        doc_cache_size: int = 50000,
        query_cache_size: int = 1000
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embed_fn = embedding_provider
        self.max_workers = max_workers
        self.base_pheromones = defaultdict(lambda: 1.0)

        # Thread safety
        self.pheromone_lock = Lock()
        
        # Performance optimizations
        self.cache_neighbors = cache_neighbors
        if self.cache_neighbors:
            self.neighbor_cache = LRUCache(neighbor_cache_size)

        self.cache_vectors = cache_vectors
        if self.cache_vectors:
            self.doc_cache = LRUCache(doc_cache_size)
            self.query_cache = LRUCache(query_cache_size)
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Shuts down the internal thread pool executor."""
        self.executor.shutdown(wait=True)

    def _resolve_params(self, **user_params) -> Dict:
        """
        Merges user-provided parameters with class defaults.
        User parameters override defaults only if they are explicitly provided (not None).
        """
        # Filter out parameters that were not provided (i.e., are None)
        # This ensures that a user passing `n_agents=None` doesn't override the default.
        active_user_params = {k: v for k, v in user_params.items() if v is not None}
        
        resolved_params = self._DEFAULT_PARAMS.copy()
        resolved_params.update(active_user_params)
        return resolved_params

    def retrieve(
            self, 
            query: Any, # Can be string or ID depending on provider
            n_agents: Optional[int] = None, 
            steps: Optional[int] = None,
            decay: Optional[float] = None,
            initial_pool_size: Optional[int] = None,
            start_subset: Optional[int] = None,
            top_k: Optional[int] = None,
            movement_strategies: Optional[Dict] = None,
            ranking_strategies: Optional[Dict] = None,
            deposit_strategies: Optional[Dict] = None
        ) -> List[Dict]:
            params = self._resolve_params(
                n_agents=n_agents,
                steps=steps,
                decay=decay,
                initial_pool_size=initial_pool_size,
                start_subset=start_subset,
                top_k=top_k,
                movement_strategies=movement_strategies,
                ranking_strategies=ranking_strategies,
                deposit_strategies=deposit_strategies
            )
            query_vec = self._get_cached_query_vector(query)
            return self._retrieve_internal(query_vec=query_vec, **params)

    def retrieve_batch(
        self,
        queries: List[Any],
        n_agents: Optional[int] = None,
        steps: Optional[int] = None,
        decay: Optional[float] = None,
        initial_pool_size: Optional[int] = None,
        start_subset: Optional[int] = None,
        top_k: Optional[int] = None,
        movement_strategies: Optional[Dict] = None,
        ranking_strategies: Optional[Dict] = None,
        deposit_strategies: Optional[Dict] = None,
        parallel_queries: bool = True,
        max_concurrent_queries: Optional[int] = None
    ) -> List[List[Dict]]:
        """
        Hybrid batch retrieval that intelligently chooses between sequential and parallel processing.
        
        Args:
            queries: List of queries to process
            parallel_queries: Whether to enable parallel query processing
            max_concurrent_queries: Max concurrent queries (auto-calculated if None)
            Other args: Same as retrieve()
        
        Returns:
            List of result lists (one per query)
        """

        if not queries:
            return []
        
        params = self._resolve_params(
            n_agents=n_agents,
            steps=steps,
            decay=decay,
            initial_pool_size=initial_pool_size,
            start_subset=start_subset,
            top_k=top_k,
            movement_strategies=movement_strategies,
            ranking_strategies=ranking_strategies,
            deposit_strategies=deposit_strategies
        )

        # Batch embed all queries
        query_vectors = self._get_cached_query_embeddings_batch(queries)

        # Decide processing strategy
        if (
            parallel_queries
            and len(queries) > 2
            and self._has_resources_for_parallel()
        ):
            max_concurrent = max_concurrent_queries or self._calculate_optimal_concurrency()
            return self._retrieve_batch_parallel(
                query_vectors,
                max_concurrent_queries=max_concurrent,
                **params
            )
        else:
            return self._retrieve_batch_sequential(
                query_vectors,
                **params
            )

    def _retrieve_batch_sequential(
        self,
        query_vectors: List[np.ndarray],
        **kwargs
    ) -> List[List[Dict]]:
        """Process queries sequentially."""
        results = []
        for vec in query_vectors:
            result = self._retrieve_internal(query_vec=vec, **kwargs)
            results.append(result)
        return results

    def _retrieve_batch_parallel(
        self,
        query_vectors: List[np.ndarray],
        max_concurrent_queries: int,
        **kwargs
    ) -> List[List[Dict]]:
        """Process queries in parallel with controlled concurrency."""
        
        def process_single_query(idx: int, vec: np.ndarray) -> tuple[int, List[Dict]]:
            result = self._retrieve_internal(query_vec=vec, **kwargs)
            return idx, result
        
        # Process queries in parallel
        with ThreadPoolExecutor(max_workers=max_concurrent_queries) as executor:
            # Submit all tasks with their indices
            future_to_index = {
                executor.submit(process_single_query, i, vec): i
                for i, vec in enumerate(query_vectors)
            }
            
            # Collect results maintaining order
            results = [None] * len(query_vectors)
            for future in as_completed(future_to_index):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    idx = future_to_index[future]
                    print(f"Query {idx} failed: {e}")
                    results[idx] = []
            
            return results

    def _calculate_optimal_concurrency(self) -> int:
        """Calculate optimal number of concurrent queries based on system resources."""
        cpu_count = os.cpu_count() or 4
        # Reserve half the cores for agent processing
        optimal = max(1, min(4, cpu_count // 2))
        return optimal

    def _has_resources_for_parallel(self) -> bool:
        """Check if system has enough resources for parallel processing."""
        # Check available memory (need at least 2GB free)
        if psutil.virtual_memory().available < 2 * 1024**3:
            return False
        
        # Check CPU load (avoid parallelizing if CPU is busy)
        if psutil.cpu_percent(interval=0.1) > 80:
            return False
        
        return True

    def _retrieve_internal(
        self,
        query_vec: np.ndarray,
        n_agents: int,
        steps: int,
        decay: float,
        initial_pool_size: int,
        start_subset: int,
        top_k: int,
        movement_strategies: Dict,
        ranking_strategies: Dict,
        deposit_strategies: Dict
    ) -> List[Dict]:
        """
        Core retrieval logic shared between retrieve() and retrieve_batch().
        """        
        # Initial search with caching
        search_res = self.vector_store.search(query_vec, limit=initial_pool_size)
        valid_pool = [r['id'] for r in search_res if self.graph_store.contains(r['id'])]
        if not valid_pool: 
            return []

        drop_zone = valid_pool[:start_subset]
        
        # Spawn Agents
        weights = [1.0 + 0.05 * (start_subset - i - 1)  for i in range(start_subset)]
        # Slightly higher weight on the most relevant for drops (0.05 inc)
        agent_locations = random.choices(drop_zone, weights=weights, k=n_agents)
        agent_trajectories = [[loc] for loc in agent_locations]
        query_pheromones = self.base_pheromones.copy()

        # --- TRAVERSAL LOOP ---
        for step in range(steps):
            # Submit all agent tasks
            futures = {
                self.executor.submit(
                    self._process_agent_step,
                    agent_id=i,
                    current_loc=agent_locations[i],
                    query_vec=query_vec,
                    query_pheromones=query_pheromones,
                    movement_strategies=movement_strategies,
                    deposit_strategies=deposit_strategies,
                    step=step
                ): i for i in range(n_agents)
            }

            new_locations = agent_locations.copy()
            pheromone_updates = defaultdict(float)
            
            for future in as_completed(futures):
                agent_id = futures[future]
                # try:
                #     result = future.result()
                #     if result:
                #         new_locations[agent_id] = result['new_location']
                #         agent_trajectories[agent_id].append(result['new_location'])
                #         if result['deposit'] > 0:
                #             pheromone_updates[result['node_id']] += result['deposit']
                # except Exception as e:
                #     print(f"Agent {agent_id} failed: {e}")
                result = future.result()
                if result:
                    new_locations[agent_id] = result['new_location']
                    agent_trajectories[agent_id].append(result['new_location'])
                    if result['deposit'] > 0:
                        pheromone_updates[result['node_id']] += result['deposit']
            
            # Batch update all agents
            agent_locations = new_locations
        
        # Batch update pheromones (thread-safe)
        with self.pheromone_lock:
            # Apply decay
            for k in query_pheromones:
                query_pheromones[k] *= decay
            # Then add new deposits
            for node_id, amount in pheromone_updates.items():
                query_pheromones[node_id] += amount
            
    
        # Parallel ranking
        return self._parallel_ranking(
            agent_trajectories, 
            query_vec, 
            ranking_strategies, 
            top_k,
            n_agents
        )          

    # === HELPERS ===
       
    def _process_agent_step(
        self, 
        agent_id: int,
        current_loc: int,
        query_vec: np.ndarray,
        query_pheromones: Dict,
        movement_strategies: Dict,
        deposit_strategies: Dict,
        step: int
    ) -> Optional[Dict]:
        """Process a single agent's movement in one step."""
        neighbors = self._get_cached_neighbors(current_loc)
        if not neighbors:
            return None
        if step % 2 == 0:
            print(f"Agent {agent_id} at {current_loc} (degree={len(neighbors)})")
        
        # Batch fetch vectors for all neighbors
        neighbor_vectors = self._fetch_vectors_batch(neighbors)
        if not neighbor_vectors:
            return None
        
        # Calculate scores for all neighbors
        scores = []
        valid_neighbors: List[int] = []
        
        max_pheromone = max(query_pheromones.values()) if query_pheromones else 1.0
        
        for i, neighbor in enumerate(neighbors):
            target_vec = neighbor_vectors[i]
            if target_vec is None: # Check for missing vector
                continue
            
            ctx = HeuristicContext(
                query_vec=query_vec,
                target_vec=target_vec,
                target_id=neighbor,
                current_id=current_loc,
                graph=self.graph_store,
                pheromones=query_pheromones,
                max_pheromone=max_pheromone,
                step_index=step,
                agent_index=agent_id
            )
            
            total_score = sum(
                func(ctx) * weight 
                for name, (func, weight) in movement_strategies.items()
            )
            scores.append(max(total_score, 0.001))
            valid_neighbors.append(i)
        
        if not valid_neighbors:
            return None
        
        # Stochastic selection
        chosen_index: int = random.choices(valid_neighbors, weights=scores, k=1)[0]
        next_node = neighbors[chosen_index]
        deposit_target_vec = neighbor_vectors[chosen_index]
        
        # Calculate deposit
        deposit_ctx = HeuristicContext(
            query_vec=query_vec,
            target_vec=deposit_target_vec,
            target_id=next_node,
            graph=self.graph_store,
            pheromones=query_pheromones
        )
        
        deposit_amount = sum(
            func(deposit_ctx) * weight 
            for name, (func, weight) in deposit_strategies.items()
        )
        
        return {
            'new_location': next_node,
            'node_id': next_node,
            'deposit': deposit_amount
        }
    
    def _parallel_ranking(
        self, 
        agent_trajectories: List[List[int]], 
        query_vec: np.ndarray,
        ranking_strategies: Dict,
        top_k: int,
        n_agents: int
    ) -> List[Dict]:
        """Parallel ranking of visited nodes."""
        # Count votes
        all_visited = [node for path in agent_trajectories for node in path]
        vote_counts = Counter(all_visited)
        unique_visited = list(vote_counts.keys())
        
        # Batch fetch all vectors
        final_vectors = self._fetch_vectors_batch(unique_visited)
        results = []
        # Parallel score calculation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for i, node_id in enumerate(unique_visited):
                vec = final_vectors[i]
                future = executor.submit(
                    self._calculate_node_score,
                    node_id=node_id,
                    votes=vote_counts[node_id],
                    query_vec=query_vec,
                    target_vec=vec,
                    ranking_strategies=ranking_strategies,
                    n_agents=n_agents
                )
                futures[future] = node_id
            
            for future in as_completed(futures):
                node_id = futures[future]
                try:
                    score = future.result()
                    results.append({'id': node_id, 'score': score})
                except Exception as e:
                    print(f"Failed to score node {node_id}: {e}")
        
        # Sort and return top-k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def _calculate_node_score(
        self,
        node_id: int,
        votes: int,
        query_vec: np.ndarray,
        target_vec: Optional[np.ndarray],
        ranking_strategies: Dict,
        n_agents: int
    ) -> float:
        """Calculate final score for a single node."""
        if target_vec is None:
            return 0.0
        
        node_ctx = HeuristicContext(
            query_vec=query_vec,
            target_vec=target_vec,
            target_id=node_id,
            graph=self.graph_store,
            votes=votes,
            total_agents=n_agents
        )
        
        return sum(
            func(node_ctx) * weight 
            for name, (func, weight) in ranking_strategies.items()
        )

    def _get_cached_neighbors(self, node_id: int) -> List[int]:
        """Gets or computes and caches the neighbor list, if enabled."""
        if not self.cache_neighbors:
            return self.graph_store.get_neighbors(node_id)

        cached = self.neighbor_cache.get(node_id)
        if cached is not None:
            return cached

        neighbors = self.graph_store.get_neighbors(node_id)
        self.neighbor_cache.set(node_id, neighbors)
        return neighbors
    
    def _fetch_vectors_batch(self, node_ids: List[int]) -> List[Optional[np.ndarray]]:
        """Fetches vectors efficiently using the cache, if enabled."""
        if not self.cache_vectors:  # Check if caching is enabled
            return self.vector_store.fetch_batch(node_ids)

        result: List[Optional[np.ndarray]] = [None] * len(node_ids)
        missing_indices = []
        missing_ids = []

        for i, node_id in enumerate(node_ids):
            cached_vec = self.doc_cache.get(node_id)
            if cached_vec is not None:
                # Cache hit: place the vector at its correct index.
                result[i] = cached_vec
            else:
                # Cache miss: record the index and ID for a batch fetch.
                missing_indices.append(i)
                missing_ids.append(node_id)
        
        if missing_ids:
            fetched_vecs = self.vector_store.fetch_batch(missing_ids)
            for i, vec in zip(missing_indices, fetched_vecs):
                if vec is not None:
                    self.doc_cache.set(node_ids[i], vec)
                result[i] = vec

        return result
            
    def _get_cached_query_vector(self, query: Any) -> np.ndarray:
        """Gets or computes and caches the query embedding, if enabled."""
        if not self.cache_vectors:
            return self.embed_fn.embed_query(query)

        cached = self.query_cache.get(query)
        if cached is not None:
            return cached

        emb = self.embed_fn.embed_query(query)
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

        for i, q in enumerate(queries):
            cached_vec = self.query_cache.get(q)
            if cached_vec is not None:
                results_by_index[i] = cached_vec
            else:
                missing_indices.append(i)
                missing_queries.append(q)

        if missing_queries:
            batch_embeddings = self.embed_fn.embed_query_batch(missing_queries)
            for i, emb in zip(missing_indices, batch_embeddings):
                q = queries[i]
                self.query_cache.set(q, emb)
                results_by_index[i] = emb


        return [results_by_index[i] for i in range(len(queries))]