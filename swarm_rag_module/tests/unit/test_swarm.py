import torch
import random
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from swarm_rag.core.swarm_retriever import SwarmRetriever
from swarm_rag.interfaces.abstract_classes import VectorStore, GraphStore, EmbeddingProvider
from swarm_rag.eval import Evaluator, EvalReporter

# Dummy implementations of abstract classes
class DummyVectorStore(VectorStore):
    def __init__(self, num_nodes=1000, embedding_dim=128, device: str = "cpu"):
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self._device = device
        # Generate deterministic embeddings
        torch.manual_seed(42)
        self.embeddings = {
            i: torch.randn(embedding_dim, dtype=torch.float32, device=device)
            for i in range(num_nodes)
        }
        # Pre-stack for efficient search
        self._embedding_matrix = torch.stack([self.embeddings[i] for i in range(num_nodes)])

    @property
    def device(self) -> str:
        return self._device

    def search(self, query_vec: torch.Tensor, limit: int):
        """Mock search returning (ids_tensor, scores_tensor)"""
        if not isinstance(query_vec, torch.Tensor):
            query_vec = torch.as_tensor(query_vec, dtype=torch.float32)
        query_vec = query_vec.flatten()
        query_norm = query_vec / (torch.linalg.norm(query_vec) + 1e-8)

        # Compute all similarities
        scores = torch.matmul(self._embedding_matrix, query_norm)

        # Get top-k
        k = min(limit, self.num_nodes)
        top_scores, top_indices = torch.topk(scores, k)

        return top_indices, top_scores

    def search_batch(self, query_vecs: torch.Tensor, limit: int):
        """Batch search returning (ids_tensor, scores_tensor) with shape (n_queries, limit)"""
        if not isinstance(query_vecs, torch.Tensor):
            query_vecs = torch.as_tensor(query_vecs, dtype=torch.float32)

        # Normalize queries
        query_norms = torch.linalg.norm(query_vecs, dim=1, keepdim=True) + 1e-8
        query_vecs_norm = query_vecs / query_norms

        # Compute all similarities: (n_queries, dim) @ (dim, n_nodes) -> (n_queries, n_nodes)
        all_scores = torch.matmul(query_vecs_norm, self._embedding_matrix.t())

        # Get top-k for each query
        k = min(limit, self.num_nodes)
        top_scores, top_indices = torch.topk(all_scores, k, dim=1)

        return top_indices, top_scores

    def fetch_batch(self, node_ids) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch embeddings for given node IDs, returning (matrix, valid_mask)"""
        if isinstance(node_ids, torch.Tensor):
            node_ids = node_ids.tolist()
        embeddings = []
        valid_mask = []
        for nid in node_ids:
            emb = self.embeddings.get(nid)
            if emb is not None:
                embeddings.append(emb)
                valid_mask.append(True)
            else:
                # Return NaN vector for missing IDs
                embeddings.append(torch.full((self.embedding_dim,), float('nan'), dtype=torch.float32))
                valid_mask.append(False)
        return torch.stack(embeddings), torch.tensor(valid_mask, dtype=torch.bool)

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
    def __init__(self, num_nodes=1000, avg_degree=5, device: str = "cpu"):
        self.num_nodes = num_nodes
        self._n_nodes = num_nodes  # Required by SwarmRetriever for pheromone tensor
        self._avg_degree = avg_degree
        self._device = device
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

    @property
    def device(self) -> str:
        return self._device

    def get_neighbors(self, node_id: Any) -> torch.Tensor:
        """Get neighbors of a node as tensor"""
        neighbors = list(self.graph.get(node_id, set()))
        return torch.tensor(neighbors, dtype=torch.long)

    def get_neighbors_batch(self, node_ids):
        """Batch neighbor lookup returning (neighbors_tensor, mask_tensor)"""
        if isinstance(node_ids, torch.Tensor):
            node_ids = node_ids.tolist()

        all_neighbors = []
        max_degree = 0
        for nid in node_ids:
            neighbors = list(self.graph.get(nid, set()))
            all_neighbors.append(neighbors)
            max_degree = max(max_degree, len(neighbors))

        if max_degree == 0:
            max_degree = 1

        # Pad to max_degree
        padded = torch.full((len(node_ids), max_degree), -1, dtype=torch.long)
        mask = torch.zeros((len(node_ids), max_degree), dtype=torch.bool)

        for i, neighbors in enumerate(all_neighbors):
            if neighbors:
                padded[i, :len(neighbors)] = torch.tensor(neighbors, dtype=torch.long)
                mask[i, :len(neighbors)] = True

        return padded, mask

    def contains(self, node_id: Any) -> bool:
        """Check if node exists"""
        return node_id in self.graph

    def neighbors(self, node_id: Any) -> torch.Tensor:
        """Alias for get_neighbors"""
        return self.get_neighbors(node_id)

    def get_avg_degree(self):
        total_degrees = sum(len(n) for n in self.graph.values())
        return total_degrees / max(1, len(self.graph))

    def get_degree(self, node_id: Any) -> int:
        """Get degree of a single node"""
        return len(self.graph.get(node_id, set()))

    def get_degrees_batch(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Batch degree lookup for multiple nodes."""
        if isinstance(node_ids, torch.Tensor):
            node_ids = node_ids.tolist()
        degrees = [self.get_degree(nid) for nid in node_ids]
        return torch.tensor(degrees, dtype=torch.long, device=self._device)

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

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

    # 1. Initialize dummy components (force CPU for unit tests)
    print("\n Initializing dummy components...")
    test_device = "cpu"  # Use CPU for unit tests to avoid GPU dependencies
    vector_store = DummyVectorStore(num_nodes=1000, embedding_dim=128, device=test_device)
    graph_store = DummyGraphStore(num_nodes=1000, avg_degree=5, device=test_device)
    embedder = DummyEmbeddingProvider(embedding_dim=128)
    evaluator = Evaluator(index_name="SwarmRetriever")

    # 2. Initialize SwarmRetriever with explicit device
    print("\n Initializing SwarmRetriever...")
    retriever = SwarmRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        embedding_provider=embedder,
        cache_neighbors=True, # Enable caching to test locks
        cache_vectors=True,
        device=test_device  # Use same device as stores
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

    # 1. Init (force CPU for unit tests)
    test_device = "cpu"
    vector_store = DummyVectorStore(device=test_device)
    graph_store = DummyGraphStore(device=test_device)
    embedder = DummyEmbeddingProvider()

    retriever = SwarmRetriever(vector_store, graph_store, embedder, device=test_device)
    
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

def test_adapter_with_new_api():
    """Test SwarmRetrieverAdapter with use_new_api=True."""
    from swarm_rag.evolution.adapters.swarm_adapter import SwarmRetrieverAdapter

    print("=" * 60)
    print("SWARM RETRIEVER: ADAPTER NEW API TEST")
    print("=" * 60)

    # Initialize components
    test_device = "cpu"
    vector_store = DummyVectorStore(num_nodes=1000, embedding_dim=128, device=test_device)
    graph_store = DummyGraphStore(num_nodes=1000, avg_degree=5, device=test_device)
    embedder = DummyEmbeddingProvider(embedding_dim=128)

    retriever = SwarmRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        embedding_provider=embedder,
        cache_neighbors=True,
        cache_vectors=True,
        device=test_device
    )

    # Test adapter with new API
    adapter = SwarmRetrieverAdapter(retriever, use_new_api=True)

    compiled = {
        "n_agents": 10,
        "steps": 5,  # Must be within bounds [4, 12]
        "decay": 0.9,
        "initial_pool_size": 20,
        "start_subset": 10,  # Must be within bounds [5, 15]
        "top_k": 5,
    }

    # Test single query
    print("\n  1. Testing adapter.retrieve with new API...")
    results = adapter.retrieve("test query", compiled)
    assert isinstance(results, list)
    assert len(results) <= 5
    assert all("id" in r and "score" in r for r in results)
    print(f"    ✓ Single query returned {len(results)} results")

    # Test batch query
    print("\n  2. Testing adapter.retrieve_batch with new API...")
    batch_results = adapter.retrieve_batch(["q1", "q2", "q3"], compiled, max_workers=1)
    assert isinstance(batch_results, list)
    assert len(batch_results) == 3
    for r in batch_results:
        assert isinstance(r, list)
    print(f"    ✓ Batch query returned {len(batch_results)} query results")

    print("\n" + "=" * 60)
    print("ADAPTER NEW API TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)


def test_new_builder_api():
    """Test the new builder pattern API: retriever.query(...).run()"""
    from swarm_rag.interfaces.retriever_types import SingleResult, BatchResult, RetrievalConfig, RunConfig

    print("=" * 60)
    print("SWARM RETRIEVER: NEW BUILDER API TEST")
    print("=" * 60)

    # Initialize components
    test_device = "cpu"
    vector_store = DummyVectorStore(num_nodes=1000, embedding_dim=128, device=test_device)
    graph_store = DummyGraphStore(num_nodes=1000, avg_degree=5, device=test_device)
    embedder = DummyEmbeddingProvider(embedding_dim=128)

    retriever = SwarmRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        embedding_provider=embedder,
        cache_neighbors=True,
        cache_vectors=True,
        device=test_device
    )

    # Test 1: Single string query
    print("\n  1. Testing single string query...")
    result = retriever.query("What is quantum computing?").run()
    assert isinstance(result, SingleResult), f"Expected SingleResult, got {type(result)}"
    assert result.node_ids.dim() == 1, f"node_ids should be 1D for single query, got {result.node_ids.dim()}D"
    assert result.scores.dim() == 1, f"scores should be 1D for single query, got {result.scores.dim()}D"
    print(f"    ✓ Single query returned {result.node_ids.shape[0]} results")

    # Test 2: Batch string queries
    print("\n  2. Testing batch string queries...")
    queries = ["query 1", "query 2", "query 3"]
    result = retriever.query(queries).run()
    assert isinstance(result, BatchResult), f"Expected BatchResult, got {type(result)}"
    assert result.node_ids.dim() == 2, f"node_ids should be 2D for batch, got {result.node_ids.dim()}D"
    assert result.node_ids.shape[0] == 3, f"Expected 3 queries, got {result.node_ids.shape[0]}"
    print(f"    ✓ Batch query returned shape {result.node_ids.shape}")

    # Test 3: Config overrides
    print("\n  3. Testing config overrides...")
    result = retriever.query("test").run(
        config=RetrievalConfig(n_agents=10, steps=2, top_k=5)
    )
    assert isinstance(result, SingleResult)
    assert result.node_ids.shape[0] <= 5, f"top_k=5 but got {result.node_ids.shape[0]} results"
    print(f"    ✓ Config override (top_k=5) returned {result.node_ids.shape[0]} results")

    # Test 4: Run config for batch size
    print("\n  4. Testing run config (batch_size)...")
    queries = [f"query {i}" for i in range(10)]
    result = retriever.query(queries).run(
        run=RunConfig(mode="batched", batch_size=3)
    )
    assert isinstance(result, BatchResult)
    assert result.node_ids.shape[0] == 10, f"Expected 10 queries, got {result.node_ids.shape[0]}"
    print(f"    ✓ Batch with batch_size=3 processed all 10 queries")

    # Test 5: Sequential mode
    print("\n  5. Testing sequential mode...")
    result = retriever.query(["q1", "q2"]).run(
        run=RunConfig(mode="sequential")
    )
    assert isinstance(result, BatchResult)
    assert result.node_ids.shape[0] == 2
    print(f"    ✓ Sequential mode processed 2 queries")

    # Test 6: Precomputed embedding (tensor input)
    print("\n  6. Testing precomputed embedding (tensor input)...")
    embedding = torch.randn(128, dtype=torch.float32)
    result = retriever.query(embedding).run()
    assert isinstance(result, SingleResult)
    print(f"    ✓ Precomputed 1D tensor returned {result.node_ids.shape[0]} results")

    # Test 7: Batch of precomputed embeddings
    print("\n  7. Testing batch of precomputed embeddings...")
    embeddings = torch.randn(5, 128, dtype=torch.float32)
    result = retriever.query(embeddings).run()
    assert isinstance(result, BatchResult)
    assert result.node_ids.shape[0] == 5
    print(f"    ✓ Precomputed 2D tensor (5 queries) returned shape {result.node_ids.shape}")

    # Test 8: Device override
    print("\n  8. Testing device override...")
    result = retriever.query("test").on("cpu").run()
    assert isinstance(result, SingleResult)
    print(f"    ✓ Device override to 'cpu' works")

    print("\n" + "=" * 60)
    print("NEW BUILDER API TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    test_swarm_retriever()
    # test_swarm_retriever_groups()
    test_new_builder_api()
    print("ALL SWARM TESTS PASSED")