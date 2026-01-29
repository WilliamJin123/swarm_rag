"""
Test optimizations for STARK Prime to achieve ~20ms/query.
Tests: torch.compile() and reduced step count.
"""
import os
import sys
import time

# Ensure UTF-8 encoding for stdout (Windows compatibility)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Enable profiling
os.environ['SWARM_PROFILE'] = '1'

import torch
from swarm_rag.core import Heuristics, SwarmRetriever, HeuristicRegistry
from swarm_rag.interfaces.enums import HeuristicKey
from swarm_rag.integrations.stark import (
    StarkPreComputedEmbeddingHandler,
    StarkVectorStore,
    StarkGraphAdapter,
)
from swarm_rag.evolution.types.config import STARK_FEATURES
from load_stark import (
    load_and_download_embeddings,
    load_and_download_skb,
    load_and_download_qa,
    precompute_stark_adjacency,
)


def make_config(n_agents=20, steps=4, weight_tensors=None):
    config = {
        "n_agents": n_agents,
        "steps": steps,
        "decay": 0.5,
        "initial_pool_size": 30,
        "start_subset": 10,
        "top_k": 20,
        "movement_strategies": {
            "semantic": (Heuristics.semantic_similarity_unnormalized, 0.5),
            "centrality": (HeuristicRegistry.get_movement("stark_centrality"), 0.2),
            "diversity": (Heuristics.pheromone_repulsion, 0.25),
            "jitter": (Heuristics.random_jitter, 0.05),
        },
        "deposit_strategies": {
            "flat": (HeuristicRegistry.get(HeuristicKey.FLAT), 1.0)
        },
        "ranking_strategies": {
            "visited": (Heuristics.percentage_visited, 0.1),
            "semantic": (Heuristics.semantic_rank, 0.9),
        },
        "feature_config": STARK_FEATURES,
    }
    if weight_tensors is not None:
        config["weight_tensors"] = weight_tensors
    return config


def benchmark_config(retriever, query_ids, config, name, warmup=3, n_runs=1):
    """Run benchmark with given config."""
    # Warmup
    for qid in query_ids[:warmup]:
        retriever.retrieve(qid, **config)
    torch.cuda.synchronize()

    # Reset profiler
    retriever._profiler.reset()

    # Timed run
    total_time = 0
    for _ in range(n_runs):
        start = time.perf_counter()
        for qid in query_ids:
            retriever.retrieve(qid, **config)
        torch.cuda.synchronize()
        total_time += time.perf_counter() - start

    latency = total_time / (len(query_ids) * n_runs) * 1000
    return latency


def main():
    print("=" * 60)
    print("STARK Prime Optimization Test - Target: 20ms/query")
    print("=" * 60)

    dataset = "prime"

    # Load data
    print("\nLoading STARK Prime dataset...")
    qa_data = load_and_download_qa(dataset)
    skb = load_and_download_skb(dataset)
    query_embs, doc_embs = load_and_download_embeddings(dataset)
    adj_dict = precompute_stark_adjacency(skb, dataset)

    # Create stores with dense=True for fused path
    print("Creating stores (dense=True for fast path)...")
    # Use script directory as base for cache paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, "adjacency_cache")
    os.makedirs(cache_dir, exist_ok=True)

    graph_store = StarkGraphAdapter(
        skb, dataset,
        adjacency_dict=adj_dict,
        cache_path=os.path.join(cache_dir, f"graph_{dataset}.npz"),
        device="cuda"
    )
    vector_store = StarkVectorStore(doc_embs, device="cuda", dense=True)
    embedding_provider = StarkPreComputedEmbeddingHandler(query_embs, device="cuda")

    # Create retriever
    retriever = SwarmRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        embedding_provider=embedding_provider,
        device="cuda"
    )

    # Create weight_tensors for batched GPU path
    # Features: [semantic_similarity_unnormalized, stark_centrality, node_centrality, pheromone_repulsion]
    weight_tensors = retriever._create_default_weight_tensors(device="cuda")
    weight_tensors.movement_weights = torch.tensor([[0.5, 0.2, 0.0, 0.25]], device="cuda")

    # Get test queries
    n_queries = 50
    query_ids = [qa_data[i][1] for i in range(n_queries)]

    print(f"\nRunning benchmarks with {n_queries} queries...")
    results = []

    # Test 1: Baseline (4 steps, 20 agents)
    config = make_config(n_agents=20, steps=4, weight_tensors=weight_tensors)
    latency = benchmark_config(retriever, query_ids, config, "baseline")
    results.append(("Baseline (4 steps, 20 agents)", latency))
    print(f"  Baseline: {latency:.2f} ms/query")

    # Test 2: Reduced steps (3 steps)
    config = make_config(n_agents=20, steps=3, weight_tensors=weight_tensors)
    latency = benchmark_config(retriever, query_ids, config, "3 steps")
    results.append(("3 steps, 20 agents", latency))
    print(f"  3 steps:  {latency:.2f} ms/query")

    # Test 3: Reduced steps (2 steps)
    config = make_config(n_agents=20, steps=2, weight_tensors=weight_tensors)
    latency = benchmark_config(retriever, query_ids, config, "2 steps")
    results.append(("2 steps, 20 agents", latency))
    print(f"  2 steps:  {latency:.2f} ms/query")

    # Test 4: Fewer agents (2 steps, 15 agents)
    config = make_config(n_agents=15, steps=2, weight_tensors=weight_tensors)
    latency = benchmark_config(retriever, query_ids, config, "2 steps 15 agents")
    results.append(("2 steps, 15 agents", latency))
    print(f"  2 steps, 15 agents: {latency:.2f} ms/query")

    # Test 5: Minimal (2 steps, 10 agents)
    config = make_config(n_agents=10, steps=2, weight_tensors=weight_tensors)
    latency = benchmark_config(retriever, query_ids, config, "2 steps 10 agents")
    results.append(("2 steps, 10 agents", latency))
    print(f"  2 steps, 10 agents: {latency:.2f} ms/query")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Path to 20ms/query")
    print("=" * 60)
    print(f"{'Configuration':<30} {'Latency':<15} {'vs 20ms':<10}")
    print("-" * 60)
    for name, lat in results:
        delta = lat - 20
        status = "ACHIEVED" if lat <= 20 else f"+{delta:.1f}ms"
        print(f"{name:<30} {lat:>6.2f} ms/query  {status}")
    print("=" * 60)

    # Profile the best configuration
    best_name, best_lat = min(results, key=lambda x: x[1])
    print(f"\nBest configuration: {best_name} ({best_lat:.2f} ms/query)")
    print("\nDetailed profile for best config:")

    # Re-run best config with profiling
    if "2 steps, 10" in best_name:
        config = make_config(n_agents=10, steps=2, weight_tensors=weight_tensors)
    elif "2 steps, 15" in best_name:
        config = make_config(n_agents=15, steps=2, weight_tensors=weight_tensors)
    elif "2 steps, 20" in best_name:
        config = make_config(n_agents=20, steps=2, weight_tensors=weight_tensors)
    elif "3 steps" in best_name:
        config = make_config(n_agents=20, steps=3, weight_tensors=weight_tensors)
    else:
        config = make_config(n_agents=20, steps=4, weight_tensors=weight_tensors)

    retriever._profiler.reset()
    for qid in query_ids[:3]:
        retriever.retrieve(qid, **config)
    torch.cuda.synchronize()
    retriever._profiler.reset()
    for qid in query_ids:
        retriever.retrieve(qid, **config)
    torch.cuda.synchronize()
    print(retriever._profiler.summary())

    # Cleanup
    vector_store.close()
    graph_store.close()


if __name__ == "__main__":
    main()
