import math
import random
import networkx as nx
import numpy as np
import lancedb
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter, defaultdict
import pandas as pd

class SwarmRetriever:
    def __init__(self, db_path: str, table_name: str, graph_path: str, embedding_fn):
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table(table_name)
        self.embed_fn = embedding_fn
    
        # Load from Parquet instead of Pickle for efficiency
        print(f"Loading graph structure from {graph_path}...")
        df_edges = pd.read_parquet(graph_path)
        self.graph = nx.from_pandas_edgelist(df_edges, source='source', target='target')
        print(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")
        
        self.base_pheromones = defaultdict(lambda: 1.0)

    def retrieve(
        self, 
        query: str, 
        n_agents: int = 20, 
        steps: int = 4,
        decay: float = 0.5,
        initial_pool_size: int = 30,
        start_subset: int = 10,
        top_k : int = 20,
        movement_weights: Optional[Dict[str, float]] = None,
        ranking_weights: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        The Swarm Traversal Loop.
        
        Args:
            query: The user's question.
            n_agents: Number of ants to deploy.
            steps: How many hops each ant takes.
            decay: Pheromone evaporation rate per step.
            initial_pool_size: Number of top vector matches to consider for the playground.
            start_subset: Agents are dropped onto the top N of that pool.
            top_k: Final top-k to return from consensus list.
            movement_weights: Dict with keys 'local_relevance', 'semantic', 'diversity'.
            ranking_weights: Dict with keys 'consensus', 'quality'.
        """
        if movement_weights is None:
            movement_weights = {"local_relevance": 0.4, "semantic": 0.3, "diversity": 0.3}
        if ranking_weights is None:
            ranking_weights = {"consensus": 0.6, "quality": 0.4}

        query_vec = np.array(self.embed_fn([query])[0])

        # 1. Broad Search (The Pool)
        search_res = self.table.search(query_vec).limit(initial_pool_size).to_list()
        # 2. Filter valid graph nodes
        valid_pool = [r['id'] for r in search_res if r['id'] in self.graph]
        if not valid_pool:
            return []
        # 3. Drop Zone (The Subset)
        drop_zone = valid_pool[:start_subset]
        if not drop_zone: drop_zone = valid_pool
        # Spawn Agents
        agent_locations = [random.choice(drop_zone) for _ in range(n_agents)]
        agent_trajectories = [[loc] for loc in agent_locations]

        # Dynamic Pheromone Map for this specific query
        # Tracks where agents have been to force diversity
        query_pheromones = self.base_pheromones.copy()

        for step in range(steps):
            # 1. Batch fetch vectors for all current locations & their neighbors
            current_neighbors = self._get_all_neighbors(agent_locations)
            all_needed_ids = set(agent_locations) | set(current_neighbors)
            vector_cache = self._fetch_vectors_batch(list(all_needed_ids))
        
            for i in range(n_agents):
                curr_id = agent_locations[i]

                # Get valid neighbors
                neighbors = list(self.graph.neighbors(curr_id))
                if not neighbors:
                    continue # Dead end, agent stays put
                
                # Calculate Probabilities for Next Step
                scores = []
                for neighbor in neighbors:
                    if neighbor not in vector_cache:
                        # Skip if we don't have data
                        continue
                    
                    # --- MOVEMENT LOGIC ---
                    
                    # 1. Semantic Similarity (Heuristic)
                    sim = self._cosine_similarity(query_vec, vector_cache[neighbor])

                    # 2. Local Relevance (Static Authority) (Heuristic #2)
                    # Using Degree for now
                    # TO CHANGE
                    local_rel = math.log(1 + self.graph.degree[neighbor])

                    # 3. Diversity Pheromone (Inverse Frequency)
                    # "If everyone went here, I should go somewhere else"
                    max_pheromone = max(query_pheromones.values()) if query_pheromones else 1.0
                    # Prevent division by zero if decay makes everything vanishingly small
                    max_pheromone = max(max_pheromone, 0.0001)

                    current_p = query_pheromones[neighbor]
                    norm_p = current_p / max_pheromone
                    diversity_score = 1.0 - norm_p

                    # Combined Weighted Score
                    score = (
                        (local_rel * movement_weights.get('local_relevance', 0.4)) + 
                        (sim * movement_weights.get('semantic', 0.3)) + 
                        (diversity_score * movement_weights.get('diversity', 0.3))
                    )
                    scores.append(max(score, 0.001)) # Avoid zero weights
                
                if not scores:
                    continue

                # Probabilistic Movement (Stochastic, not Greedy)
                valid_neighbors = [n for n in neighbors if n in vector_cache]
                if not valid_neighbors:
                    continue
                next_node = random.choices(valid_neighbors, weights=scores, k=1)[0]

                # Move Agent
                agent_locations[i] = next_node
                agent_trajectories[i].append(next_node)

                # Deposit Pheromone (Mark trail)
                # We add to the map so future agents avoid this path (Diversity)
                query_pheromones[next_node] += local_rel
    
            for k in query_pheromones:
                query_pheromones[k] *= decay

        # Consensus (Voting)
        all_visited = [node for path in agent_trajectories for node in path]
        vote_counts = Counter(all_visited)
        unique_visited = list(vote_counts.keys())
        final_vectors = self._fetch_vectors_batch(unique_visited)

        consensus_results = []
        for node_id, votes in vote_counts.items():
            if node_id not in final_vectors:
                continue

            # --- RANKING LOGIC ---
            # Final Score = (Consensus Agreement) + (Semantic Quality)

            semantic_score = self._cosine_similarity(query_vec, final_vectors[node_id])
            vote_score = votes / n_agents # Normalize to 0-1
            
            final_score = (
                (vote_score * ranking_weights.get('consensus', 0.6)) + 
                (semantic_score * ranking_weights.get('quality', 0.4))
            )
            consensus_results.append((node_id, final_score))

        # Sort by final consensus score
        consensus_results.sort(key=lambda x: x[1], reverse=True)
        top_candidates = consensus_results[:top_k]
        top_ids = [nid for nid, _ in top_candidates]
        score_map = {nid: score for nid, score in top_candidates}
        docs = self.table.search().where(f"id IN {top_ids}").limit(len(top_ids)).to_list()

        # Resort and attach scores
        sorted_docs = []
        for doc in docs:
            doc['swarm_score'] = score_map.get(doc['id'], 0.0)
            sorted_docs.append(doc)
        sorted_docs.sort(key=lambda x: x['swarm_score'], reverse=True)

        return sorted_docs

    def _get_all_neighbors(self, nodes):
        """Helper to get union of all neighbors for a list of nodes."""
        all_n = set()
        for n in nodes:
            if n in self.graph:
                all_n.update(self.graph.neighbors(n))
        return list(all_n)
    
    def _fetch_vectors_batch(self, ids):
        if not ids: return {}
        # Simple batch fetch simulation for LanceDB
        # In production, chunk this if ids > 1000
        q_ids = [f"'{i}'" for i in ids]
        res = self.table.search().where(f"id IN ({','.join(q_ids)})").limit(len(ids)).to_list()
        return {r['id']: np.array(r['vector']) for r in res}

    @staticmethod
    def _cosine_similarity(a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)