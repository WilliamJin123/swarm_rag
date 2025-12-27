
# test_swarm.py
import numpy as np
import random
import time
from typing import List, Dict, Any, Optional, Union
from swarm_rag.core.swarm_retriever import SwarmRetriever
from swarm_rag.interfaces.base import VectorStore, GraphStore, EmbeddingProvider
from swarm_rag.eval import Evaluator, EvalReporter

# Dummy implementations of abstract classes
class DummyVectorStore(VectorStore):
    def __init__(self, num_nodes=1000, embedding_dim=128):
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        # Generate deterministic embeddings
        np.random.seed(42)
        self.embeddings = {
            i: np.random.randn(embedding_dim).astype(np.float32)
            for i in range(num_nodes)
        }
        
    def search(self, query_vec: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """Mock search that returns random nodes with scores"""
        results = []
        for i in range(min(limit, self.num_nodes)):
            node_id = random.randint(0, self.num_nodes - 1)
            # Calculate mock similarity score
            score = np.dot(query_vec, self.embeddings[node_id]) / (
                np.linalg.norm(query_vec) * np.linalg.norm(self.embeddings[node_id])
            )
            results.append({'id': node_id, 'score': float(score)})
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def fetch_batch(self, node_ids: List[Any]) -> List[Optional[np.ndarray]]:
        """Fetch embeddings for given node IDs"""
        return [self.embeddings.get(nid) for nid in node_ids]

    def fetch(self, node_id) -> Optional[np.ndarray]:
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

class DummyEmbeddingProvider(EmbeddingProvider):
    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
        np.random.seed(42)
        
    def embed_query(self, query: Union[str, Any]) -> np.ndarray:
        """Generate deterministic embedding based on query hash"""
        if isinstance(query, str):
            seed = hash(query) % (2**32)
        else:
            seed = hash(str(query)) % (2**32)
        
        np.random.seed(seed)
        return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def embed_query_batch(self, queries: list[Any]) -> List[np.ndarray]:
        """Generate embeddings for multiple queries"""
        return [self.embed_query(q) for q in queries]
    


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

    print("\nTesting batch retrieval with automatic strategy selection...")
    start_time = time.time()
    batch_results = retriever.retrieve_batch(queries=queries, n_agents=10, steps=3, top_k=5, parallel_queries=True)
    latency_per_query = (time.time() - start_time)/len(queries)
    for q, res in zip(queries, batch_results):
        ground_truth_batch = [r['id'] for r in res[:3]]
        batch_metrics = evaluator.calculate_metrics(res, ground_truth_batch, latency_per_query)
        reporter.add_run("Batch Queries", batch_metrics)
        print(f"   ✓ Query '{q[:30]}...' -> {len(res)} results, MRR={batch_metrics['MRR']:.4f}")

    large_queries = [f"Test query {i}" for i in range(10)]

    print("\n Testing sequential batch processing...")
    sequential_results = retriever.retrieve_batch(large_queries, n_agents=5, steps=2, top_k=3, parallel_queries=False)
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
        parallel_queries=True,
        max_concurrent_queries=4
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
    for group, df in aggregated.items():
        print(f"\n{group}:")
        print(df.to_string(index=False))
        reporter.plot_metrics(df=df, title=group, metrics=list(df.columns))
    
    if len(aggregated) > 1:
        reporter.plot_comparison(aggregated_results=aggregated, metrics=list(df.columns))

    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_swarm_retriever()
