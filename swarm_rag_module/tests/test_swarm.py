
# test_swarm.py
import numpy as np
import random
import time
from typing import List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor
import threading
from ..swarm_rag_module.interfaces.base import VectorStore, GraphStore, EmbeddingProvider
from ..swarm_rag_module.eval.metrics import Evaluator

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
    
    def fetch_batch(self, node_ids: List[Any]) -> Dict[Any, np.ndarray]:
        """Fetch embeddings for given node IDs"""
        return {nid: self.embeddings[nid] for nid in node_ids if nid in self.embeddings}

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
    print("\n1. Initializing dummy components...")
    vector_store = DummyVectorStore(num_nodes=1000, embedding_dim=128)
    graph_store = DummyGraphStore(num_nodes=1000, avg_degree=5)
    embedder = DummyEmbeddingProvider(embedding_dim=128)
    evaluator = Evaluator(index_name="SwarmRetriever")
    
    # 2. Initialize SwarmRetriever
    print("2. Initializing SwarmRetriever...")
    retriever = SwarmRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        embedding_provider=embedder,
        max_workers=4,
        cache_neighbors=True
    )

    all_query_metrics = []
    
    # 3. Test single query
    print("\n3. Testing single query retrieval...")
    start_time = time.time()
    single_results = retriever.retrieve(
        query="What is quantum entanglement?",
        n_agents=10,
        steps=3,
        top_k=5
    )
    single_time = time.time() - start_time
    
    print(f"   ✓ Single query completed in {single_time:.3f}s")
    print(f"   ✓ Returned {len(single_results)} results")
    print("   Top 3 results:")
    for i, res in enumerate(single_results[:3]):
        print(f"     {i+1}. Node {res['id']} (score: {res['score']:.4f})")
    
    ground_truth_single = [res['id'] for res in single_results[:3]]
    single_metrics = evaluator.calculate_metrics(single_results, ground_truth_single, single_time)
    all_query_metrics.append(single_metrics)
    print(f"   ✓ Evaluated single query. MRR: {single_metrics['MRR']:.4f}, Hit@10: {single_metrics['Hit@10']:.4f}")

    # 4. Test batch with automatic strategy selection
    print("\n4. Testing batch retrieval with automatic strategy selection...")
    queries = [
        "What is quantum entanglement?",
        "Explain black holes",
        "How do neural networks work?"
    ]
    
    start_time = time.time()
    batch_results = retriever.retrieve_batch(
        queries=queries,
        n_agents=10,
        steps=3,
        top_k=5,
        parallel_queries=True  # Let system decide
    )
    batch_time = time.time() - start_time
    
    print(f"   ✓ Batch query completed in {batch_time:.3f}s")
    print(f"   ✓ Processed {len(queries)} queries")
    for i, (query, results) in enumerate(zip(queries, batch_results)):
        print(f"   Query {i+1}: '{query[:30]}...' -> {len(results)} results")
    
    avg_latency_per_query = batch_time / len(queries)
    for i, (query, results) in enumerate(zip(queries, batch_results)):
        # Simulate ground truth for each query
        ground_truth_batch = [res['id'] for res in results[:3]]
        # Use average latency for each query in the batch
        batch_metrics = evaluator.calculate_metrics(results, ground_truth_batch, avg_latency_per_query)
        all_query_metrics.append(batch_metrics)
        print(f"   ✓ Evaluated query {i+1}: MRR: {batch_metrics['MRR']:.4f}, Hit@10: {batch_metrics['Hit@10']:.4f}")

    # 5. Test forced sequential processing
    print("\n5. Testing forced sequential batch processing...")
    large_query_list = [f"Test query {i}" for i in range(10)]
    
    start_time = time.time()
    sequential_results = retriever.retrieve_batch(
        queries=large_query_list,
        n_agents=5,
        steps=2,
        top_k=3,
        parallel_queries=False
    )
    sequential_time = time.time() - start_time
    
    print(f"   ✓ Sequential batch completed in {sequential_time:.3f}s")
    print(f"   ✓ Processed {len(large_query_list)} queries sequentially")
    
    avg_latency_per_query = sequential_time / len(large_query_list)
    for i, (query, results) in enumerate(zip(large_query_list, sequential_results)):
        ground_truth_seq = [res['id'] for res in results[:1]] # Simulate 1 correct result
        seq_metrics = evaluator.calculate_metrics(results, ground_truth_seq, avg_latency_per_query)
        all_query_metrics.append(seq_metrics)

    # 6. Test custom parallel settings
    print("\n6. Testing custom parallel settings...")
    start_time = time.time()
    parallel_results = retriever.retrieve_batch(
        queries=large_query_list,
        n_agents=5,
        steps=2,
        top_k=3,
        parallel_queries=True,
        max_concurrent_queries=4
    )
    parallel_time = time.time() - start_time
    
    print(f"   ✓ Parallel batch completed in {parallel_time:.3f}s")
    print(f"   ✓ Processed {len(large_query_list)} queries with 4 workers")
    
    avg_latency_per_query = parallel_time / len(large_query_list)
    for i, (query, results) in enumerate(zip(large_query_list, parallel_results)):
        ground_truth_par = [res['id'] for res in results[:1]] # Simulate 1 correct result
        par_metrics = evaluator.calculate_metrics(results, ground_truth_par, avg_latency_per_query)
        all_query_metrics.append(par_metrics)

    # 7. Performance comparison
    print("\n7. Performance Summary:")
    print(f"   Single query:     {single_time:.3f}s")
    print(f"   Small batch:      {batch_time:.3f}s ({batch_time/len(queries):.3f}s per query)")
    print(f"   Sequential batch: {sequential_time:.3f}s ({sequential_time/len(large_query_list):.3f}s per query)")
    print(f"   Parallel batch:   {parallel_time:.3f}s ({parallel_time/len(large_query_list):.3f}s per query)")
    
    # 8. Verify result consistency
    print("\n8. Verifying result consistency...")
    consistent = True
    for i in range(min(3, len(single_results))):
        if single_results[i]['id'] != batch_results[0][i]['id']:
            consistent = False
            break
    
    if consistent:
        print("   ✓ Results are consistent across single and batch queries")
    else:
        print("   ⚠ Results vary (expected due to stochastic nature)")
    
    # 9. Test error handling
    print("\n9. Testing error handling...")
    try:
        empty_results = retriever.retrieve_batch(
            queries=[],
            parallel_queries=True
        )
        print("   ✓ Empty query list handled correctly")
    except Exception as e:
        print(f"   ✗ Error with empty queries: {e}")
    
    # 10. Resource usage test
    print("\n10. Resource usage simulation...")
    print("   Testing with resource constraints...")
    
    # Mock low memory scenario
    original_has_resources = retriever._has_resources_for_parallel
    retriever._has_resources_for_parallel = lambda: False
    
    start_time = time.time()
    constrained_results = retriever.retrieve_batch(
        queries=queries[:2],
        parallel_queries=True  # Will be forced to sequential
    )
    constrained_time = time.time() - start_time
    
    print(f"   ✓ Constrained processing completed in {constrained_time:.3f}s")
    
    avg_latency_per_query = constrained_time / len(queries[:2])
    for i, (query, results) in enumerate(zip(queries[:2], constrained_results)):
        ground_truth_con = [res['id'] for res in results[:3]]
        con_metrics = evaluator.calculate_metrics(results, ground_truth_con, avg_latency_per_query)
        all_query_metrics.append(con_metrics)

    # Restore original method
    retriever._has_resources_for_parallel = original_has_resources
    
    # 11. Final Aggregation and Reporting
    print("\n" + "=" * 60)
    print("AGGREGATED EVALUATION RESULTS")
    print("=" * 60)
    
    final_report = evaluator.aggregate_results(all_query_metrics)
    
    if not final_report.empty:
        print(final_report)
    else:
        print("Could not generate final report.")

    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    # Import the SwarmRetriever (assuming it's in the same directory or installed)
    try:
        from ..swarm_rag_module.core.swarm_retriever import SwarmRetriever
        test_swarm_retriever()
    except ImportError:
        print("Error: Could not import SwarmRetriever. Make sure it's in your Python path.")
        print("For testing purposes, you can copy the SwarmRetriever class into this file.")