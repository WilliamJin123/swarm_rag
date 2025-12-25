import math
import random
import networkx as nx
import numpy as np
import lancedb
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter, defaultdict
import pandas as pd

from .heuristic_strategies import Heuristics

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
        # Default strategies: Name -> (Function, Weight)
        movement_strategies: Dict[str, Tuple[callable, float]] = None,
        ranking_strategies: Dict[str, Tuple[callable, float]] = None,
        deposit_strategies: Dict[str, Tuple[callable, float]] = None
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
        # 1. Setup Plug-and-Play Defaults
        if movement_strategies is None:
            movement_strategies = {
                "semantic": (Heuristics.semantic_similarity, 0.3),
                "authority": (Heuristics.node_authority, 0.4),
                "diversity": (Heuristics.pheromone_repulsion, 0.3)
            }
        
        if ranking_strategies is None:
            ranking_strategies = {
                "consensus": (Heuristics.consensus_vote, 0.6),
                "semantic": (Heuristics.semantic_rank, 0.4)
            }
        if deposit_strategies is None:
            deposit_strategies = {
                "authority_mark": (Heuristics.deposit_authority, 1.0) 
            }
        
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
        query_pheromones = self.base_pheromones.copy()

        # --- TRAVERSAL LOOP ---
        for step in range(steps):
            current_neighbors = self._get_all_neighbors(agent_locations)
            all_needed_ids = set(agent_locations) | set(current_neighbors)
            vector_cache = self._fetch_vectors_batch(list(all_needed_ids))

            max_pheromone = max(query_pheromones.values()) if query_pheromones else 1.0

            for i in range(n_agents):
                curr_id = agent_locations[i]
                neighbors = list(self.graph.neighbors(curr_id))
                if not neighbors: continue

                scores = []
                valid_neighbors_for_step = []

                for neighbor in neighbors:
                    if neighbor not in vector_cache: continue
                    # This dict contains everything a heuristic might need
                    ctx = {
                        'query_vec': query_vec,
                        'target_vec': vector_cache[neighbor],
                        'target_id': neighbor,
                        'current_id': curr_id,
                        'graph': self.graph,
                        'pheromones': query_pheromones,
                        'max_pheromone': max_pheromone,
                        'step_index': step,
                        'agent_index': i
                    }

                    total_score = 0.0
                    for name, (func, weight) in movement_strategies.items():
                        # Execute the lambda/function
                        val = func(ctx)
                        total_score += val * weight
                    
                    scores.append(max(total_score, 0.001))
                    valid_neighbors_for_step.append(neighbor)

                if not valid_neighbors_for_step: continue

                # Stochastic Move
                next_node = random.choices(valid_neighbors_for_step, weights=scores, k=1)[0]
                agent_locations[i] = next_node
                agent_trajectories[i].append(next_node)

                deposit_ctx = {
                    'query_vec': query_vec,
                    'target_vec': vector_cache[next_node], # We know this exists from movement check
                    'target_id': next_node,
                    'graph': self.graph,
                    'pheromones': query_pheromones
                }

                # 2. Calculate total deposit amount based on strategies
                deposit_amount = 0.0
                for name, (func, weight) in deposit_strategies.items():
                    val = func(deposit_ctx)
                    deposit_amount += val * weight
                
                # 3. Update the map
                query_pheromones[next_node] += deposit_amount
                
            # Decay
            for k in query_pheromones:
                query_pheromones[k] *= decay

        # Consensus
        all_visited = [node for path in agent_trajectories for node in path]
        vote_counts = Counter(all_visited)
        unique_visited = list(vote_counts.keys())
        final_vectors = self._fetch_vectors_batch(unique_visited)

        ranked_results = []
        for node_id, votes in vote_counts.items():
            if node_id not in final_vectors:
                continue

            # --- RANKING CONTEXT ---
            ctx = {
                'query_vec': query_vec,
                'target_vec': final_vectors[node_id],
                'target_id': node_id,
                'votes': votes,
                'total_agents': n_agents,
                'graph': self.graph
            }

            final_score = 0.0
            for name, (func, weight) in ranking_strategies.items():
                final_score += func(ctx) * weight
            
            ranked_results.append((node_id, final_score))
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        return self._format_results(ranked_results[:top_k])

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

    def _format_results():
        pass