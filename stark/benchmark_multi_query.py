"""Benchmark multi-query batching performance."""
import time
import torch
import argparse
from typing import Optional


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_gpu_memory():
    """Reset GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


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


def benchmark_new_builder_api(retriever, n_queries=50, batch_sizes=[1, 8, 16, 32, 64]):
    """Benchmark the new builder API with different batch sizes."""
    from swarm_rag.interfaces import RunConfig, RetrievalConfig

    # Generate test embeddings
    query_embeddings = torch.randn(n_queries, retriever.vector_store.dim, device=retriever._device)
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)

    # Get initial pools via similarity search
    pool_ids, _ = retriever.vector_store.search_batch(query_embeddings, limit=30)

    config = RetrievalConfig(
        n_agents=20,
        steps=5,
        decay=0.9,
        initial_pool_size=30,
        start_subset=10,
        top_k=20,
        drop_zone_inc=0.05,
    )

    print("\n=== New Builder API Benchmark ===")
    results = {}

    for bs in batch_sizes:
        reset_gpu_memory()

        run_config: RunConfig = {
            "mode": "sequential" if bs == 1 else "batched",
            "batch_size": bs,
        }

        # Warmup
        _ = retriever.query(query_embeddings[:8], pool=pool_ids[:8]).run(
            config=config, run=run_config
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        reset_gpu_memory()

        # Timed run
        start = time.perf_counter()
        result = retriever.query(query_embeddings, pool=pool_ids).run(
            config=config, run=run_config
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start

        peak_mem = get_gpu_memory_mb()

        mode_label = "seq" if bs == 1 else f"batch={bs}"
        results[bs] = {"time": elapsed, "mem_mb": peak_mem}
        print(f"{mode_label:>10}: {elapsed*1000:.1f}ms ({n_queries/elapsed:.1f} q/s) | Peak mem: {peak_mem:.1f}MB")

    return results


def benchmark_evolution_style(retriever, n_queries=50, n_genomes=5, batch_size=32):
    """
    Benchmark retrieval as used in evolution loop with different batch sizes.

    On GPU, retrieve_batch_with_precomputed always uses batched processing.
    This benchmark compares different internal batch sizes.
    """
    from swarm_rag.evolution.execution.shared_precompute import prepare_shared_context

    print(f"\n=== Evolution-Style Benchmark ({n_genomes} genomes × {n_queries} queries) ===")

    # Generate test data
    query_embeddings = torch.randn(n_queries, retriever.vector_store.dim, device=retriever._device)
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)

    # Simulate ground truth (not used for timing, just for context)
    ground_truth = [[i, i+1] for i in range(n_queries)]
    queries = [f"query_{i}" for i in range(n_queries)]

    # Prepare shared context (done once per generation)
    reset_gpu_memory()

    start = time.perf_counter()
    context = prepare_shared_context(
        retriever=retriever,
        queries=queries,
        ground_truth=ground_truth,
        unique_pool_sizes=[30],
        device=retriever._device,
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    precompute_time = time.perf_counter() - start

    print(f"Pre-compute time: {precompute_time*1000:.1f}ms")

    # Get pools
    initial_pools = context.initial_pools.get(30, [])

    # Benchmark different batch sizes for GPU path
    for internal_batch_size in [8, 16, 32, 64]:
        reset_gpu_memory()

        # The internal batch size for _retrieve_batch_multi_query_gpu is hardcoded to 32
        # Let's test the new builder API instead which supports configurable batch sizes
        from swarm_rag.interfaces import RunConfig, RetrievalConfig

        config = RetrievalConfig(
            n_agents=20,
            steps=5,
            decay=0.9,
            initial_pool_size=30,
            start_subset=10,
            top_k=20,
            drop_zone_inc=0.05,
        )

        # Get pool tensor
        pool_ids, _ = retriever.vector_store.search_batch(context.query_embeddings, limit=30)

        run_config: RunConfig = {"mode": "batched", "batch_size": internal_batch_size}

        # Warmup
        _ = retriever.query(context.query_embeddings[:8], pool=pool_ids[:8]).run(
            config=config, run=run_config
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        reset_gpu_memory()

        start = time.perf_counter()
        for g in range(n_genomes):
            _ = retriever.query(context.query_embeddings, pool=pool_ids).run(
                config=config, run=run_config
            )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start
        mem = get_gpu_memory_mb()

        total_queries = n_genomes * n_queries
        print(f"  batch_size={internal_batch_size:2d}: {elapsed*1000:.1f}ms ({total_queries/elapsed:.1f} q/s) | Mem: {mem:.1f}MB")

    # Also benchmark old API for comparison
    reset_gpu_memory()

    start = time.perf_counter()
    for g in range(n_genomes):
        _ = retriever.retrieve_batch_with_precomputed(
            query_embeddings=context.query_embeddings,
            initial_pools=initial_pools,
            n_agents=20,
            steps=5,
            decay=0.9,
            initial_pool_size=30,
            start_subset=10,
            top_k=20,
            max_workers=4,
        )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    old_api_time = time.perf_counter() - start
    old_api_mem = get_gpu_memory_mb()

    total_queries = n_genomes * n_queries
    print(f"  Old API (batch=32): {old_api_time*1000:.1f}ms ({total_queries/old_api_time:.1f} q/s) | Mem: {old_api_mem:.1f}MB")

    return {
        "precompute_time": precompute_time,
        "old_api": {"time": old_api_time, "mem_mb": old_api_mem},
    }


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
        def __init__(self, dim, device="cuda"):
            self.dim = dim
            self.device = device
        def embed_query(self, q):
            return torch.randn(self.dim, device=self.device)
        def embed_query_batch(self, queries):
            return torch.randn(len(queries), self.dim, device=self.device)

    retriever = SwarmRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        embedding_provider=DummyEmbedder(vector_store.dim),
        device="cuda",
    )

    print(f"\nBenchmarking with {args.n_queries} queries...")
    print("\n=== Batch Size Comparison (Old API) ===")
    benchmark_batch_sizes(retriever, n_queries=args.n_queries)

    print("\n=== Sequential vs Batched (Old API) ===")
    benchmark_sequential_vs_batched(retriever, n_queries=args.n_queries)

    print("\n=== New Builder API ===")
    try:
        benchmark_new_builder_api(retriever, n_queries=args.n_queries)
    except Exception as e:
        print(f"Builder API benchmark failed: {e}")

    print("\n=== Evolution-Style Simulation ===")
    try:
        benchmark_evolution_style(retriever, n_queries=args.n_queries, n_genomes=5)
    except Exception as e:
        print(f"Evolution-style benchmark failed: {e}")
