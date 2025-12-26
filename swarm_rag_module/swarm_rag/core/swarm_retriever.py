import random
import numpy as np
from typing import Any, List, Dict, Optional
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
import os
import psutil
from functools import lru_cache

from .heuristics import Heuristics
from ..interfaces.base import VectorStore, GraphStore, EmbeddingProvider

def _create_neighbor_cacher(graph_store: GraphStore, maxsize: int):
    """
    Factory function that creates and returns a cached neighbor-getting function.
    This allows the cache size to be configured at runtime.
    """
    @lru_cache(maxsize=maxsize)
    def _cacher(node_id: int) -> List[int]:
        """
        The actual cached function. It closes over the `graph_store` instance.
        The cache key is simply the `node_id`.
        """
        return graph_store.get_neighbors(node_id)
    return _cacher

def _create_doc_cacher(vector_store: VectorStore, maxsize: int):
    """
    Factory to create a cached function for fetching single vectors.
    """
    @lru_cache(maxsize=maxsize)
    def _cacher(node_id: int) -> Optional[np.ndarray]:
        """
        The cached function. Closes over the `vector_store` instance.
        The cache key is the `node_id`.
        """
        batch = vector_store.fetch_batch([node_id])
        return batch.get(node_id)
    return _cacher

def _create_query_cacher(embedding_provider: EmbeddingProvider, maxsize: int):
    """
    Factory to create a cached function for query embeddings.
    """
    @lru_cache(maxsize=maxsize)
    def _cacher(query: Any) -> np.ndarray:
        """
        The cached function. Closes over the `embedding_provider`.
        The cache key is the `query` object itself (must be hashable).
        """
        return embedding_provider.embed_query(query)
    return _cacher

class SwarmRetriever:
    def __init__(
        self, 
        vector_store: VectorStore, 
        graph_store: GraphStore, 
        embedding_provider: EmbeddingProvider,
        max_workers: int = 16,
        cache_neighbors: bool = True,
        neighbor_cache_size: int = 5000,
        cache_vectors: bool = True,
        vector_cache_size: int = 50000,
        query_cache_size: int = 1000
    ):
        self.vector_store = vector_store
        self.graph = graph_store
        self.embed_fn = embedding_provider
        self.max_workers = max_workers
        self.base_pheromones = defaultdict(lambda: 1.0)

        # Thread safety
        self.pheromone_lock = Lock()
        
        # Performance optimizations
        self.cache_neighbors = cache_neighbors
        if self.cache_neighbors:
            self.neighbor_cache_size = neighbor_cache_size
            self._cached_get_neighbors = _create_neighbor_cacher(self.graph, self.neighbor_cache_size)

        self.cache_vectors = cache_vectors
        if self.cache_vectors:
            self.vector_cache_size = vector_cache_size
            self.query_cache_size = query_cache_size
            self._cached_get_single_vector = _create_doc_cacher(
                self.vector_store, self.vector_cache_size
            )
            self._cached_embed_query = _create_query_cacher(
                self.embed_fn, self.query_cache_size
            )
        

        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def retrieve_batch(
        self,
        queries: List[Any],
        n_agents: int = 20,
        steps: int = 4,
        decay: float = 0.5,
        initial_pool_size: int = 30,
        start_subset: int = 10,
        top_k: int = 20,
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
        
        # 1. Batch embed all queries
        query_vectors = self.embed_fn.embed_query_batch(queries)
        
        # 2. Update cache for all queries
        for query, vec in zip(queries, query_vectors):
            query_key = str(query) if isinstance(query, (str, int)) else id(query)
            self.query_vec_cache[query_key] = vec
        
        # 3. Decide processing strategy
        should_parallel = (
            parallel_queries 
            and len(queries) > 2
            and self._has_resources_for_parallel()
        )
        
        if should_parallel:
            max_concurrent = max_concurrent_queries or self._calculate_optimal_concurrency()
            return self._retrieve_batch_parallel(
                query_vectors,
                n_agents=n_agents,
                steps=steps,
                decay=decay,
                initial_pool_size=initial_pool_size,
                start_subset=start_subset,
                top_k=top_k,
                movement_strategies=movement_strategies,
                ranking_strategies=ranking_strategies,
                deposit_strategies=deposit_strategies,
                max_concurrent_queries=max_concurrent
            )
        else:
            return self._retrieve_batch_sequential(
                query_vectors,
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
        # Create semaphore to limit concurrent queries
        semaphore = Semaphore(max_concurrent_queries)
        
        def process_single_query(idx: int, vec: np.ndarray) -> tuple[int, List[Dict]]:
            with semaphore:
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

    def retrieve(
        self, 
        query: Any, # Can be string or ID depending on provider
        n_agents: int = 20, 
        steps: int = 4,
        decay: float = 0.5,
        initial_pool_size: int = 30,
        start_subset: int = 10,
        top_k : int = 20,
        movement_strategies = None,
        ranking_strategies = None,
        deposit_strategies = None
    ) -> List[Dict]:
        query_vec = self._get_cached_query_vector(query)
        return self._retrieve_internal(
            query_vec=query_vec,
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

    def _retrieve_internal(
        self,
        query_vec: np.ndarray,
        n_agents: int,
        steps: int,
        decay: float,
        initial_pool_size: int,
        start_subset: int,
        top_k: int,
        movement_strategies: Optional[Dict],
        ranking_strategies: Optional[Dict],
        deposit_strategies: Optional[Dict]
    ) -> List[Dict]:
        """
        Core retrieval logic shared between retrieve() and retrieve_batch().
        """
        # Setup Default Strategies
        if movement_strategies is None:
            movement_strategies = {
                "semantic": (Heuristics.semantic_similarity, 0.3),
                "centrality": (Heuristics.node_centrality, 0.4),
                "diversity": (Heuristics.pheromone_repulsion, 0.3)
            }
        
        if ranking_strategies is None:
            ranking_strategies = {
                "consensus": (Heuristics.consensus_vote, 0.6),
                "semantic": (Heuristics.semantic_rank, 0.4)
            }
        if deposit_strategies is None:
            deposit_strategies = {
                "flat_mark": (Heuristics.deposit_flat, 1.0) 
            }
        
        # Initial search with caching
        search_res = self.vector_store.search(query_vec, limit=initial_pool_size)
        valid_pool = [r['id'] for r in search_res if self.graph.contains(r['id'])]
        if not valid_pool: 
            return []

        drop_zone = valid_pool[:start_subset]
        
        # Spawn Agents
        agent_locations = np.array([random.choice(drop_zone) for _ in range(n_agents)])
        agent_trajectories = [[loc] for loc in agent_locations]
        query_pheromones = self.base_pheromones.copy()

        self._prefetch_vectors(agent_locations)

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
                try:
                    result = future.result()
                    if result:
                        new_locations[agent_id] = result['new_location']
                        agent_trajectories[agent_id].append(result['new_location'])
                        if result['deposit'] > 0:
                            pheromone_updates[result['node_id']] += result['deposit']
                except Exception as e:
                    print(f"Agent {agent_id} failed: {e}")
            
            # Batch update all agents
            agent_locations = new_locations
        
        # Batch update pheromones (thread-safe)
        with self.pheromone_lock:
            for node_id, amount in pheromone_updates.items():
                query_pheromones[node_id] += amount
            # Apply decay to all
            for k in query_pheromones:
                query_pheromones[k] *= decay
    
        # Parallel consensus ranking
        return self._parallel_consensus_ranking(
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
        if step % 2 == 0:
            print(f"Agent {agent_id} at {current_loc} (degree={len(neighbors)})")
        neighbors = self._get_cached_neighbors(current_loc)
        if not neighbors:
            return None
        
        # Batch fetch vectors for all neighbors
        neighbor_vectors = self._fetch_vectors_batch(neighbors)
        if not neighbor_vectors:
            return None
        
        # Calculate scores for all neighbors
        scores = []
        valid_neighbors = []
        
        max_pheromone = max(query_pheromones.values()) if query_pheromones else 1.0
        
        for neighbor in neighbors:
            if neighbor not in neighbor_vectors:
                continue
            
            ctx = {
                'query_vec': query_vec,
                'target_vec': neighbor_vectors[neighbor],
                'target_id': neighbor,
                'current_id': current_loc,
                'graph': self.graph,
                'pheromones': query_pheromones,
                'max_pheromone': max_pheromone,
                'step_index': step,
                'agent_index': agent_id
            }
            
            total_score = sum(
                func(ctx) * weight 
                for name, (func, weight) in movement_strategies.items()
            )
            scores.append(max(total_score, 0.001))
            valid_neighbors.append(neighbor)
        
        if not valid_neighbors:
            return None
        
        # Stochastic selection
        next_node = random.choices(valid_neighbors, weights=scores, k=1)[0]
        
        # Calculate deposit
        deposit_ctx = {
            'query_vec': query_vec,
            'target_vec': neighbor_vectors[next_node],
            'target_id': next_node,
            'graph': self.graph,
            'pheromones': query_pheromones
        }
        
        deposit_amount = sum(
            func(deposit_ctx) * weight 
            for name, (func, weight) in deposit_strategies.items()
        )
        
        return {
            'new_location': next_node,
            'node_id': next_node,
            'deposit': deposit_amount
        }
    
    def _parallel_consensus_ranking(
        self, 
        agent_trajectories: List[List[int]], 
        query_vec: np.ndarray,
        ranking_strategies: Dict,
        top_k: int,
        n_agents: int
    ) -> List[Dict]:
        """Parallel consensus ranking of visited nodes."""
        # Count votes
        all_visited = [node for path in agent_trajectories for node in path]
        vote_counts = Counter(all_visited)
        unique_visited = list(vote_counts.keys())
        
        # Batch fetch all vectors
        final_vectors = self._fetch_vectors_batch(unique_visited)
        
        # Parallel score calculation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._calculate_node_score,
                    node_id,
                    vote_counts[node_id],
                    query_vec,
                    final_vectors.get(node_id),
                    ranking_strategies,
                    n_agents=n_agents
                ): node_id for node_id in unique_visited
            }
            
            results = []
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
        
        ctx = {
            'query_vec': query_vec,
            'target_vec': target_vec,
            'target_id': node_id,
            'votes': votes,
            'total_agents': n_agents,
            'graph': self.graph
        }
        
        return sum(
            func(ctx) * weight 
            for name, (func, weight) in ranking_strategies.items()
        )

    def _get_cached_neighbors(self, node_id: int) -> List[int]:
        """Gets or computes and caches the neighbor list, if enabled."""
        if self.cache_neighbors:
            return self._cached_get_neighbors(node_id)
        else:
            return self.graph.get_neighbors(node_id)
    
    def _prefetch_vectors(self, node_ids: List[int]):
        """Prefetch vectors for given nodes."""
        missing = [nid for nid in node_ids if nid not in self.vector_cache]
        if missing:
            batch = self.vector_store.fetch_batch(missing)
            self.vector_cache.update(batch)
    
    def _fetch_vectors_batch(self, node_ids: List[int]) -> Dict[int, np.ndarray]:
        """Fetches vectors efficiently using the cache, if enabled."""
        if not self.cache_vectors:  # Check if caching is enabled
            return self.vector_store.fetch_batch(node_ids)

        result = {}
        missing_ids = []

        for nid in node_ids:
            cached_vec = self._cached_get_single_vector(nid)
            if cached_vec is not None:
                result[nid] = cached_vec
            else:
                missing_ids.append(nid)
        
        if missing_ids:
            fetched_batch = self.vector_store.fetch_batch(missing_ids)
            result.update(fetched_batch)
        
        return result
            
    def _get_cached_query_vector(self, query: Any) -> np.ndarray:
        """Gets or computes and caches the query embedding, if enabled."""
        if self.cache_vectors:
            # Call the cached helper function
            self._cached_embed_query(query)
        else:
            # Fallback: no caching, direct call
            return self.embed_fn.embed_query(query)
