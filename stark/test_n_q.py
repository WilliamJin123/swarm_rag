
"""
SwarmRAG evaluation on STaRK datasets.

Usage:
    python test_n_q.py                    # 10 questions, auto device
    python test_n_q.py -n 50              # 50 questions
    python test_n_q.py -f                 # Full QA dataset
    python test_n_q.py -v                 # Verbose per-query output
    python test_n_q.py -c                 # Compare GPU vs CPU
    python test_n_q.py --device cpu       # Force CPU mode
"""
import argparse
import os
import random
import sys
import time
import torch

# Ensure UTF-8 encoding for stdout (Windows compatibility)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from swarm_rag.core import Heuristics, SwarmRetriever, HeuristicRegistry
from swarm_rag.interfaces.enums import HeuristicKey
from swarm_rag.integrations.stark import (
    StarkPreComputedEmbeddingHandler,
    StarkVectorStore,
    StarkGraphAdapter,
)
from swarm_rag.eval import Evaluator
from swarm_rag.utils.device import get_device
from load_stark import (
    load_and_download_embeddings,
    load_and_download_skb,
    load_and_download_qa,
    precompute_stark_adjacency,
)

CONFIG = {
    "n_agents": 25,
    "steps": 5,
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
}


def run_eval(retriever: SwarmRetriever, evaluator: Evaluator, qa_data, indices, verbose=False):
    """Run evaluation, return results list and total time."""
    results = []
    total_time = 0.0

    for i, idx in enumerate(indices):
        query, query_id, answer_ids, _ = qa_data[idx]

        start = time.time()
        retrieved = retriever.retrieve(query=query_id, **CONFIG)
        latency = time.time() - start
        total_time += latency

        metrics = evaluator.calculate_metrics(retrieved, answer_ids, latency)
        results.append(metrics)

        if verbose:
            n_gt = max(len(answer_ids), 5)
            nodes = metrics.get('node_results', [])[:n_gt]
            # Show each node with Y/N marker for hit/miss
            formatted = [f"{n['id']}({'Y' if n['correct'] else 'N'})" for n in nodes]
            hits = [n['id'] for n in nodes if n['correct']]
            misses = [n['id'] for n in nodes if not n['correct']]
            print(f"\n[{i+1}/{len(indices)}] Query {query_id}")
            print(f"  Top {n_gt}: {formatted}")
            print(f"  Hits: {hits}")
            print(f"  Misses: {misses}")
            print(f"  Ground truth: {answer_ids}")
            print(f"  MRR: {metrics['MRR']:.3f} | Recall@20: {metrics['Recall@20']:.2f}")
        elif (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(indices)}] lat={latency:.3f}s Hit@5={metrics['Hit@5']:.2f}")

    return results, total_time


def main():
    parser = argparse.ArgumentParser(description="SwarmRAG STaRK evaluation")
    parser.add_argument("-n", type=int, default=10, help="Number of questions")
    parser.add_argument("-f", "--full", action="store_true", help="Run full QA dataset")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-c", "--compare", action="store_true", help="Compare GPU vs CPU")
    parser.add_argument("-he", "--human_eval", action="store_true", help="Run against the human eval dataset")
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto", help="Device mode")
    parser.add_argument("--dataset", default="prime", help="Dataset: prime, amazon, mag")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--agents", type=int, help="Override n_agents")
    parser.add_argument("--steps", type=int, help="Override steps")
    args = parser.parse_args()

    # Apply overrides
    if args.agents:
        CONFIG["n_agents"] = args.agents
    if args.steps:
        CONFIG["steps"] = args.steps

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Resolve device
    # Note: We map "gpu" input to "cuda" string for PyTorch compatibility
    if args.device == "gpu":
        base_device = "cuda"
    elif args.device == "cpu":
        base_device = "cpu"
    else:
        base_device = get_device()

    # Determine devices to test
    if args.compare and base_device == "cuda":
        devices = ["cuda", "cpu"]
    else:
        devices = [base_device]

    print(f"\n{'='*50}")
    print(f"SwarmRAG STaRK Evaluation")
    print(f"Dataset: {args.dataset} | Mode: {'compare' if args.compare else base_device}")
    print(f"Agents: {CONFIG['n_agents']} | Steps: {CONFIG['steps']}")
    print(f"{'='*50}")

    # Load data
    print(f"\nLoading data... {'[HUMAN_EVAL]' if args.human_eval else ''}")
    qa_data = load_and_download_qa(args.dataset, human_gen=args.human_eval)
    skb = load_and_download_skb(args.dataset)
    query_embs, doc_embs, query_ids, doc_ids = load_and_download_embeddings(args.dataset)
    adj_dict = precompute_stark_adjacency(skb, args.dataset)

    # Determine sample size
    total_q = len(qa_data)
    n_questions = total_q if args.full else min(args.n, total_q)
    indices = list(range(n_questions))
    print(f"Questions: {n_questions}/{total_q}" + (" (full)" if args.full else ""))

    # Use script directory as base for cache paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, "adjacency_cache")
    os.makedirs(cache_dir, exist_ok=True)
    embedding_provider = StarkPreComputedEmbeddingHandler(query_embs, query_ids=query_ids)
    evaluator = Evaluator(k_values=[1, 5, 10, 20])

    all_results = {}

    for device in devices:
        mode_name = "GPU" if device == "cuda" else "CPU"
        print(f"\n--- {mode_name} ---")

        vector_store = StarkVectorStore(doc_embs, doc_ids, device=device, dense=True)

        graph_store = StarkGraphAdapter(
            skb, args.dataset,
            adjacency_dict=adj_dict,
            cache_path=os.path.join(cache_dir, f"graph_{args.dataset}.npz"),
            device=device
        )

        retriever = SwarmRetriever(
            vector_store=vector_store,
            graph_store=graph_store,
            embedding_provider=embedding_provider,
            cache_neighbors=False,
            device=device
        )
        print(f"  device={retriever.device}")

        if device == "cuda":
            print("  warmup...")
            _ = retriever.retrieve(query=qa_data[0][1], **CONFIG)

        results, total_time = run_eval(retriever, evaluator, qa_data, indices, args.verbose)
        avg_lat = total_time / len(results)
        agg = evaluator.aggregate_results(results)

        all_results[mode_name] = {"results": results, "avg_lat": avg_lat, "agg": agg}

        print(f"\n  {mode_name} Results (avg {avg_lat:.3f}s/query):")
        print(evaluator.format_results(agg))

    # Comparison summary
    if args.compare and len(all_results) > 1:
        print(f"\n{'='*50}")
        print("COMPARISON")
        print(f"{'='*50}")
        gpu_lat = all_results["GPU"]["avg_lat"]
        cpu_lat = all_results["CPU"]["avg_lat"]
        speedup = cpu_lat / gpu_lat if gpu_lat > 0 else 0
        print(f"GPU: {gpu_lat:.3f}s/query | CPU: {cpu_lat:.3f}s/query")
        print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()