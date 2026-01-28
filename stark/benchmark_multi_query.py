"""Benchmark multi-query batching performance."""
import time
import torch
import argparse


def benchmark_batch_sizes(retriever, n_queries=50, batch_sizes=[1, 8, 16, 32, 64]):
    """Benchmark different batch sizes."""
    query_embeddings = torch.randn(n_queries, retriever.vector_store.dim, device=retriever._device)
    initial_pools = [[i % retriever._max_node_id for i in range(j, j+30)]
                     for j in range(n_queries)]

    resolved_agents = retriever._prepare_agents(
        None, 20,
        retriever._DEFAULT_PARAMS['movement_strategies'],
        retriever._DEFAULT_PARAMS['deposit_strategies'],
    )

    results = {}

    for bs in batch_sizes:
        # Warmup
        _ = retriever._retrieve_batch_multi_query_gpu(
            query_embeddings[:8], initial_pools[:8],
            resolved_agents,
            base_seed=42, batch_size=bs, steps=5, decay=0.9,
            drop_zone_inc=0.05, start_subset=10, top_k=20,
            ranking_strategies=retriever._DEFAULT_PARAMS['ranking_strategies'],
        )
        torch.cuda.synchronize()

        # Timed run
        start = time.perf_counter()
        _ = retriever._retrieve_batch_multi_query_gpu(
            query_embeddings, initial_pools,
            resolved_agents,
            base_seed=42, batch_size=bs, steps=5, decay=0.9,
            drop_zone_inc=0.05, start_subset=10, top_k=20,
            ranking_strategies=retriever._DEFAULT_PARAMS['ranking_strategies'],
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        results[bs] = elapsed
        print(f"Batch size {bs:3d}: {elapsed*1000:.1f}ms ({n_queries/elapsed:.1f} queries/sec)")

    return results


def benchmark_sequential_vs_batched(retriever, n_queries=50):
    """Compare sequential vs batched performance."""
    query_embeddings = torch.randn(n_queries, retriever.vector_store.dim, device=retriever._device)
    initial_pools = [[i % retriever._max_node_id for i in range(j, j+30)]
                     for j in range(n_queries)]

    resolved_agents = retriever._prepare_agents(
        None, 20,
        retriever._DEFAULT_PARAMS['movement_strategies'],
        retriever._DEFAULT_PARAMS['deposit_strategies'],
    )

    params = {
        'steps': 5,
        'decay': 0.9,
        'drop_zone_inc': 0.05,
        'start_subset': 10,
        'top_k': 20,
        'ranking_strategies': retriever._DEFAULT_PARAMS['ranking_strategies'],
    }

    # Warmup
    _ = retriever._retrieve_batch_precomputed_sequential(
        query_embeddings[:5], initial_pools[:5],
        resolved_agents, base_seed=42, **params
    )
    torch.cuda.synchronize()

    # Sequential timing
    start = time.perf_counter()
    _ = retriever._retrieve_batch_precomputed_sequential(
        query_embeddings, initial_pools,
        resolved_agents, base_seed=42, **params
    )
    torch.cuda.synchronize()
    seq_time = time.perf_counter() - start

    # Batched timing
    _ = retriever._retrieve_batch_multi_query_gpu(
        query_embeddings[:5], initial_pools[:5],
        resolved_agents, base_seed=42, batch_size=32, **params
    )
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = retriever._retrieve_batch_multi_query_gpu(
        query_embeddings, initial_pools,
        resolved_agents, base_seed=42, batch_size=32, **params
    )
    torch.cuda.synchronize()
    batch_time = time.perf_counter() - start

    print(f"\nSequential: {seq_time*1000:.1f}ms ({n_queries/seq_time:.1f} queries/sec)")
    print(f"Batched:    {batch_time*1000:.1f}ms ({n_queries/batch_time:.1f} queries/sec)")
    print(f"Speedup:    {seq_time/batch_time:.2f}x")

    return seq_time, batch_time


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description="Benchmark multi-query batching")
    parser.add_argument("--dataset", default="prime", choices=["prime", "amazon", "mag"])
    parser.add_argument("--n_queries", type=int, default=50)
    args = parser.parse_args()

    # Load real data
    from load_stark import load_and_download_skb, load_and_download_embeddings, precompute_stark_adjacency
    from swarm_rag.integrations.stark import StarkVectorStore, StarkGraphAdapter
    from swarm_rag.core import SwarmRetriever

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    print(f"Loading {args.dataset} dataset...")
    skb = load_and_download_skb(args.dataset)
    query_embs, doc_embs = load_and_download_embeddings(args.dataset)
    adj_dict = precompute_stark_adjacency(skb, args.dataset)

    vector_store = StarkVectorStore(doc_embs, device="cuda")
    graph_store = StarkGraphAdapter(
        skb, args.dataset,
        adjacency_dict=adj_dict,
        cache_path=os.path.join(BASE_DIR, "adjacency_cache", f"graph_{args.dataset}.npz"),
        device="cuda",
    )

    # Simple embedding provider (we use pre-computed embeddings)
    class DummyEmbedder:
        def __init__(self, dim):
            self.dim = dim
        def embed_query(self, q):
            return torch.randn(self.dim)

    retriever = SwarmRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        embedding_provider=DummyEmbedder(vector_store.dim),
        device="cuda",
    )

    print(f"\nBenchmarking with {args.n_queries} queries...")
    print("\n=== Batch Size Comparison ===")
    benchmark_batch_sizes(retriever, n_queries=args.n_queries)

    print("\n=== Sequential vs Batched ===")
    benchmark_sequential_vs_batched(retriever, n_queries=args.n_queries)
