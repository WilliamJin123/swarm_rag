import torch
import random
import time
from typing import List, Dict, Any, Optional, Union
from swarm_rag.core.swarm_retriever import SwarmRetriever
from swarm_rag.interfaces.abstract_classes import VectorStore, GraphStore, EmbeddingProvider
from swarm_rag.eval import Evaluator, EvalReporter

# Dummy implementations of abstract classes
class DummyVectorStore(VectorStore):
    def __init__(self, num_nodes=1000, embedding_dim=128):
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        # Generate deterministic embeddings
        torch.manual_seed(42)
        self.embeddings = {
            i: torch.randn(embedding_dim, dtype=torch.float32)
            for i in range(num_nodes)
        }

    def search(self, query_vec: torch.Tensor, limit: int) -> List[Dict[str, Any]]:
        """Mock search that returns deterministic nodes based on query vector"""
        # Convert to tensor if needed
        if not isinstance(query_vec, torch.Tensor):
            query_vec = torch.as_tensor(query_vec, dtype=torch.float32)
        # Use query vector to seed selection for determinism
        query_hash = hash(query_vec.numpy().tobytes()) % (2**32)
        rng = random.Random(query_hash)

        results = []
        seen_ids = set()
        while len(results) < min(limit, self.num_nodes) and len(seen_ids) < self.num_nodes:
            node_id = rng.randint(0, self.num_nodes - 1)
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                emb = self.embeddings[node_id]
                score = torch.dot(query_vec, emb) / (
                    torch.linalg.norm(query_vec) * torch.linalg.norm(emb) + 1e-8
                )
                results.append({'id': node_id, 'score': float(score)})
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def fetch_batch(self, node_ids: List[Any]) -> torch.Tensor:
        """Fetch embeddings for given node IDs, returning stacked tensor matrix"""
        embeddings = []
        for nid in node_ids:
            emb = self.embeddings.get(nid)
            if emb is not None:
                embeddings.append(emb)
            else:
                # Return NaN vector for missing IDs
                embeddings.append(torch.full((self.embedding_dim,), float('nan'), dtype=torch.float32))
        return torch.stack(embeddings)

    def fetch(self, node_id) -> Optional[torch.Tensor]:
        return self.embeddings.get(node_id)

class DegreeView:
    """A simple dict-like view to mimic NetworkX's DegreeView."""
    def __init__(self, graph_dict):
        self._graph = graph_dict
        self._degree_cache = {node: len(neighbors) for node, neighbors in graph_dict.items()}

    def __getitem__(self, node_id):
        return self._degree_cache.get(node_id, 0)

    def __iter__(self):
        return iter(self._degree_cache.items())

    def __repr__(self):
        return f"DegreeView({self._degree_cache})"


class DummyGraphStore(GraphStore):
    def __init__(self, num_nodes=1000, avg_degree=5):
        self.num_nodes = num_nodes
        self.avg_degree = avg_degree
        # Create a simple graph structure
        self.graph = {i: set() for i in range(num_nodes)}
        
        # Add random edges
        random.seed(42)
        for i in range(num_nodes):
            for _ in range(avg_degree):
                neighbor = random.randint(0, num_nodes - 1)
                if neighbor != i:
                    self.graph[i].add(neighbor)
                    self.graph[neighbor].add(i)
        
        # Add the NetworkX-like degree view
        self.degree = DegreeView(self.graph)
    
    def get_neighbors(self, node_id: Any) -> List[Any]:
        """Get neighbors of a node"""
        return list(self.graph.get(node_id, set()))
    
    def contains(self, node_id: Any) -> bool:
        """Check if node exists"""
        return node_id in self.graph
    
    def neighbors(self, node_id: Any) -> List[Any]:
        """Alias for get_neighbors"""
        return self.get_neighbors(node_id)
    
    def get_avg_degree(self):
        # Basic implementation if base class doesn't handle it
        total_degrees = sum(len(n) for n in self.graph.values())
        return total_degrees / max(1, len(self.graph))

class DummyEmbeddingProvider(EmbeddingProvider):
    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
        torch.manual_seed(42)

    def embed_query(self, query: Union[str, Any]) -> torch.Tensor:
        """Generate deterministic embedding based on query hash"""
        if isinstance(query, str):
            seed = hash(query) % (2**32)
        else:
            seed = hash(str(query)) % (2**32)

        torch.manual_seed(seed)
        return torch.randn(self.embedding_dim, dtype=torch.float32)

    def embed_query_batch(self, queries: list[Any]) -> torch.Tensor:
        """Generate embeddings for multiple queries, returning stacked tensor matrix"""
        embeddings = [self.embed_query(q) for q in queries]
        return torch.stack(embeddings)
    


def test_swarm_retriever():
    """Comprehensive test of SwarmRetriever functionality"""
    
    print("=" * 60)
    print("SWARM RETRIEVER TEST SUITE")
    print("=" * 60)
    
    # 1. Initialize dummy components
    print("\n Initializing dummy components...")
    vector_store = DummyVectorStore(num_nodes=1000, embedding_dim=128)
    graph_store = DummyGraphStore(num_nodes=1000, avg_degree=5)
    embedder = DummyEmbeddingProvider(embedding_dim=128)
    evaluator = Evaluator(index_name="SwarmRetriever")
    
    # 2. Initialize SwarmRetriever
    print("\n Initializing SwarmRetriever...")
    retriever = SwarmRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        embedding_provider=embedder,
        cache_neighbors=True, # Enable caching to test locks
        cache_vectors=True
    )

    reporter = EvalReporter()
    queries = [
        "What is quantum entanglement?",
        "Explain black holes",
        "How do neural networks work?"
    ]

    print("\n Testing single query retrieval...")
    start_time = time.time()
    single_results = retriever.retrieve(query=queries[0], n_agents=5, steps=3, top_k=5)
    latency = time.time() - start_time
    ground_truth = [res['id'] for res in single_results[:3]]
    metrics = evaluator.calculate_metrics(single_results, ground_truth, latency)
    reporter.add_run("Single Query", metrics)
    print(f"   ✓ Single query returned {len(single_results)} results in {latency:.3f}s")
    print(f"   ✓ Metrics: MRR={metrics['MRR']:.4f}, Hit@10={metrics['Hit@10']:.4f}")

    # Assertions for single query test
    assert len(single_results) > 0, "Should return results"
    assert len(single_results) <= 5, "Should respect top_k"
    assert all('id' in r and 'score' in r for r in single_results), "Results should have id and score"

    print("\nTesting batch retrieval with automatic strategy selection...")
    start_time = time.time()
    batch_results = retriever.retrieve_batch(queries=queries, n_agents=10, steps=3, top_k=5, max_workers=10)
    latency_per_query = (time.time() - start_time)/len(queries)
    for q, res in zip(queries, batch_results):
        ground_truth_batch = [r['id'] for r in res[:3]]
        batch_metrics = evaluator.calculate_metrics(res, ground_truth_batch, latency_per_query)
        reporter.add_run("Batch Queries", batch_metrics)
        print(f"   ✓ Query '{q[:30]}...' -> {len(res)} results, MRR={batch_metrics['MRR']:.4f}")

    # Assertions for batch test
    assert len(batch_results) == len(queries), "Should return results for all queries"

    large_queries = [f"Test query {i}" for i in range(10)]

    print("\n Testing sequential batch processing...")
    # FIX: Reset timer
    start_time = time.time() 
    sequential_results = retriever.retrieve_batch(large_queries, n_agents=5, steps=2, top_k=3, max_workers=1)
    latency_seq = (time.time() - start_time)/len(large_queries)
    for q, res in zip(large_queries, sequential_results):
        ground_truth_seq = [r['id'] for r in res[:1]]
        seq_metrics = evaluator.calculate_metrics(res, ground_truth_seq, latency_seq)
        reporter.add_run("Sequential Batch", seq_metrics)
    print(f"   ✓ Sequential batch processed {len(large_queries)} queries in {time.time() - start_time:.3f}s")

    print("\n6. Testing custom parallel settings...")
    start_time = time.time()
    parallel_results = retriever.retrieve_batch(
        queries=large_queries,
        n_agents=5,
        steps=2,
        top_k=3,
        max_workers=4
    )
    parallel_time = time.time() - start_time
    avg_latency_per_query = parallel_time / len(large_queries)

    for q, res in zip(large_queries, parallel_results):
        ground_truth = [r['id'] for r in res[:1]]  # simulate 1 correct result
        metrics = evaluator.calculate_metrics(res, ground_truth, avg_latency_per_query)
        reporter.add_run("Parallel Batch (Custom)", metrics)
    print(f"   ✓ Parallel batch with 4 workers completed in {parallel_time:.3f}s")
      
    aggregated = reporter.aggregate(evaluator)
    print("\nAGGREGATED RESULTS")
    
    # Store first df to safely access columns later
    first_df = None
    
    for group, df in aggregated.items():
        if first_df is None: first_df = df
        print(f"\n{group}:")
        print(df.to_string(index=False))
        # Pass explicit metrics to avoid scope errors
        reporter.plot_metrics(df=df, title=group, metrics=list(df.columns))
    
    if len(aggregated) > 1 and first_df is not None:
        # FIX: Use first_df columns instead of loop variable
        reporter.plot_comparison(aggregated_results=aggregated, metrics=list(first_df.columns))

    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)


def test_swarm_retriever_groups():
    print("=" * 60)
    print("SWARM RETRIEVER: HETEROGENEOUS GROUPS TEST")
    print("=" * 60)
    
    # 1. Init
    vector_store = DummyVectorStore()
    graph_store = DummyGraphStore()
    embedder = DummyEmbeddingProvider()
    
    retriever = SwarmRetriever(vector_store, graph_store, embedder)
    
    # 2. Define Groups
    # Group A: 5 Agents, Semantic
    # Group B: 15 Agents, Random (Diversity)
    groups = [
        {
            'count': 5,
            'movement_strategies': {'semantic': ('semantic_similarity', 1.0)},
            'deposit_strategies': {'flat': ('flat', 1.0)}
        },
        {
            'count': 15,
            'movement_strategies': {'random': ('pheromone_repulsion', 1.0)},
            'deposit_strategies': {'flat': ('flat', 1.0)}
        }
    ]
    
    query = "test_query"
    
    # 3. Retrieve
    print("  Running retrieve with agent_groups...")
    start = time.time()
    
    results = retriever.retrieve(
        query=query,
        agent_groups=groups,
        steps=3,
        drop_zone_inc=0.1 # Test new param
    )
    
    duration = time.time() - start
    print(f"  ✓ Retrieval complete in {duration:.3f}s")
    print(f"  ✓ Returned {len(results)} results")
    
    # 4. Verification
    assert len(results) > 0
    
    print("\n  Testing Batch Retrieval with Groups...")
    batch_results = retriever.retrieve_batch(
        queries=["q1", "q2"],
        agent_groups=groups,
        steps=2
    )
    assert len(batch_results) == 2
    print("  ✓ Batch retrieval passed")

if __name__ == "__main__":
    test_swarm_retriever()
    # test_swarm_retriever_groups()
    print("SWARM TESTS PASSED")