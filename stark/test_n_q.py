"""
Stark QA Evaluation Script for SwarmRAG

This script tests the SwarmRAG retrieval system on the Stark Prime dataset,
supporting both GPU and CPU modes with detailed metrics reporting.

Usage:
    python test_n_q.py --n 10 --device auto          # Auto-detect GPU/CPU
    python test_n_q.py --n 10 --device gpu           # Force GPU (auto-enables GPU graph store)
    python test_n_q.py --n 10 --device cpu           # Force CPU
    python test_n_q.py --n 10 --compare              # Compare GPU vs CPU
    python test_n_q.py --n 10 --verbose              # Detailed per-query output

Performance Notes:
-----------------
The GPU provides ~8x speedup for vector search operations, but overall retrieval
speedup is modest (~1.0-1.3x) because the bottleneck is CPU graph traversal.

WHY IS GRAPH TRAVERSAL THE BOTTLENECK?
--------------------------------------
Each agent step involves:
  1. Get neighbors from graph (CPU) - O(degree) ~125 avg for prime
  2. Fetch embeddings for neighbors (GPU/CPU)
  3. Compute similarity scores (GPU fast, CPU slower)
  4. Random selection based on scores (CPU)

The graph adjacency lookups happen ~n_agents * steps times per query (e.g., 25*5=125).
These are CPU-bound operations using scipy sparse matrix indexing.

HOW TO IMPROVE THE BOTTLENECK:
------------------------------
1. **GPU Graph Store**: Move adjacency matrix to GPU as torch sparse tensor
   - Requires: torch.sparse_csr_tensor for adjacency
   - Benefit: ~5-10x faster neighbor lookups

2. **Batch Graph Operations**: Fetch all agent neighbors in one operation
   - Current: Sequential per-agent neighbor fetch
   - Improved: Batch all agent locations, fetch all neighbors at once

3. **Graph Caching on GPU**: Keep frequently accessed nodes in GPU memory
   - Use LRU cache with GPU tensors for hot nodes
   - Reduces CPU-GPU transfer overhead

4. **Reduce Agent Steps**: Use larger initial pool, fewer traversal steps
   - Trade exploration depth for speed
   - initial_pool_size=50, steps=2 instead of pool=30, steps=5

5. **Parallel Agent Processing**: Process agents in parallel batches
   - Group agents by location, process similar paths together
   - Use torch.scatter for batched pheromone updates

Example GPU Graph Implementation:
    class GPUGraphStore:
        def __init__(self, adj_dict):
            # Convert to GPU sparse tensor
            self.adj = torch.sparse_csr_tensor(...).cuda()

        def get_neighbors_batch(self, node_ids: torch.Tensor) -> torch.Tensor:
            # Batch lookup on GPU
            return self.adj[node_ids].coalesce().indices()
"""

import random
import time
import argparse
import os
from typing import List, Optional, Dict, Any
import pandas as pd
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from swarm_rag.core import Heuristics, SwarmRetriever, HeuristicRegistry
from swarm_rag.interfaces.enums import HeuristicKey
from swarm_rag.integrations.stark import (
    StarkInMemoryVectorStore,
    StarkPreComputedEmbeddingHandler,
    StarkSKBAdapter,
    StarkGPUVectorStore,
    StarkGPUGraphAdapter,
)
from swarm_rag.eval import Evaluator, EvalReporter
from swarm_rag.utils.device import get_device, clear_device_cache
from load_stark import load_and_download_embeddings, load_and_download_skb, load_and_download_qa, precompute_stark_adjacency


# Default hyperparameters (optimized for Stark Prime)
DEFAULT_CONFIG = {
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


def safe_print(text: str) -> None:
    """Print with ASCII fallback for Windows console compatibility."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


def safe_print_df(df: pd.DataFrame, evaluator: Evaluator = None, style: str = "vertical") -> None:
    """Print DataFrame with ASCII fallback, using vertical format by default."""
    if evaluator and hasattr(evaluator, 'format_results'):
        text = evaluator.format_results(df, style=style)
    else:
        text = df.to_string()
    safe_print(text)


def create_vector_store(doc_embs: Dict, device: str):
    """Create appropriate vector store based on device."""
    if device == "cuda":
        return StarkGPUVectorStore(doc_embs, use_gpu=True)
    else:
        return StarkInMemoryVectorStore(doc_embs)


def run_evaluation(
    retriever: SwarmRetriever,
    evaluator: Evaluator,
    qa_data: list,
    sampled_indices: list,
    config: dict,
    verbose: bool = False,
    mode_name: str = "Test",
) -> Dict[str, Any]:
    """
    Run evaluation on sampled questions.

    Returns dict with 'results', 'avg_latency', 'total_time', 'per_query_data'
    """
    query_results = []
    per_query_data = []
    total_latency = 0.0

    for i, idx in enumerate(sampled_indices):
        query, query_id, answer_ids, _ = qa_data[idx]

        start_time = time.time()
        retrieved_nodes = retriever.retrieve(
            query=query_id,
            n_agents=config["n_agents"],
            steps=config["steps"],
            decay=config["decay"],
            initial_pool_size=config["initial_pool_size"],
            start_subset=config["start_subset"],
            top_k=config["top_k"],
            movement_strategies=config["movement_strategies"],
            deposit_strategies=config["deposit_strategies"],
            ranking_strategies=config["ranking_strategies"],
        )
        latency = time.time() - start_time
        total_latency += latency

        metrics = evaluator.calculate_metrics(
            retrieved_nodes=retrieved_nodes,
            ground_truth_ids=answer_ids,
            latency_sec=latency,
        )
        query_results.append(metrics)

        # Store per-query details
        n_gt = max(len(answer_ids), 5)
        query_data = {
            "query_id": query_id,
            "query_text": query[:50] + "..." if len(query) > 50 else query,
            "ground_truth": answer_ids,
            "retrieved_top_n": [n['id'] for n in retrieved_nodes[:n_gt]],
            "scores_top_n": [n['score'] for n in retrieved_nodes[:n_gt]],
            "hits_in_top20": [aid for aid in answer_ids if aid in [n['id'] for n in retrieved_nodes[:20]]],
            "latency": latency,
            "metrics": metrics,
        }
        per_query_data.append(query_data)

        # Progress output
        if verbose:
            print(f"\n--- [{mode_name}] Query {i + 1}/{len(sampled_indices)} (ID: {query_id}) ---")
            # Display with correctness markers (Y/N)
            node_results = metrics.get('node_results', [])
            top_n = node_results[:n_gt]
            formatted = [f"{n['id']} ({'Y' if n['correct'] else 'N'})" for n in top_n]
            correct_count = sum(1 for n in top_n if n['correct'])
            print(f"  Top {n_gt} IDs: {formatted}")
            print(f"  Correct: {correct_count}/{n_gt}")
            scores_str = [f"{n.get('score', 0):.4f}" for n in top_n]
            print(f"  Top {n_gt} scores: {scores_str}")
            print(f"  Ground truth: {answer_ids}")
            print(f"  Hits in top 20: {query_data['hits_in_top20']}")
            print(f"  Latency: {latency:.4f}s | Hit@1: {metrics['Hit@1']:.2f} | "
                  f"Hit@5: {metrics['Hit@5']:.2f} | Recall@20: {metrics['Recall@20']:.2f} | "
                  f"MRR: {metrics['MRR']:.4f}")
        elif (i + 1) % 5 == 0 or i == 0:
            print(f"  [{mode_name}] Query {i + 1}/{len(sampled_indices)}: "
                  f"latency={latency:.3f}s, Hit@5={metrics['Hit@5']:.2f}")

    avg_latency = total_latency / len(sampled_indices) if sampled_indices else 0

    return {
        "results": query_results,
        "avg_latency": avg_latency,
        "total_time": total_latency,
        "per_query_data": per_query_data,
    }


def test_stark(
    dataset_names: List[str],
    num_questions: int = 10,
    seed: Optional[int] = None,
    human_gen: bool = False,
    device: str = "auto",
    compare: bool = False,
    verbose: bool = False,
    config: Optional[dict] = None,
    plot: bool = False,
    save_plots: Optional[str] = None,
) -> None:
    """
    Unified test function for Stark QA evaluation.

    Args:
        dataset_names: List of datasets to test (e.g., ["prime"])
        num_questions: Number of questions to evaluate
        seed: Random seed for reproducibility
        human_gen: Use human-generated QA data
        device: "auto", "gpu", "cpu", or "cuda" (GPU mode auto-enables GPU graph store)
        compare: Run both GPU and CPU for comparison
        verbose: Show detailed per-query output
        config: Override default hyperparameters
        plot: Display metric graphs after evaluation
        save_plots: Directory to save plots (if provided)
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # Resolve device
    if device == "auto":
        resolved_device = get_device()
    elif device in ("gpu", "cuda"):
        resolved_device = "cuda"
    else:
        resolved_device = "cpu"

    # Merge config
    run_config = DEFAULT_CONFIG.copy()
    if config:
        run_config.update(config)

    # GPU graph store is automatically enabled when using GPU
    use_gpu_graph = (resolved_device == "cuda")

    print(f"\n{'='*70}")
    print(f"SWARM RAG STARK EVALUATION")
    print(f"{'='*70}")
    print(f"Device: {resolved_device} (requested: {device})")
    print(f"Questions: {num_questions}")
    print(f"Agents: {run_config['n_agents']}, Steps: {run_config['steps']}")
    print(f"Compare mode: {compare}")
    print(f"GPU Graph: {use_gpu_graph} (auto-enabled with GPU)")
    print(f"{'='*70}")

    reporter = EvalReporter()

    for dataset_name in dataset_names:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")

        # Load data
        print("\nLoading data...")
        qa_data = load_and_download_qa(dataset_name, human_gen=human_gen)
        skb = load_and_download_skb(dataset_name)
        query_embs, doc_embs = load_and_download_embeddings(dataset_name)
        adjacency_dict = precompute_stark_adjacency(skb, dataset_name)

        # Setup paths
        cache_dir = os.path.join("stark", "adjacency_cache")
        os.makedirs(cache_dir, exist_ok=True)
        graph_cache_path = os.path.join(cache_dir, f"graph_{dataset_name}.npz")

        # Shared components - use GPU graph store automatically when using GPU
        if use_gpu_graph:
            print(f"  Initializing GPU graph store...")
            graph_store = StarkGPUGraphAdapter(
                skb, dataset_name,
                adjacency_dict=adjacency_dict,
                cache_path=graph_cache_path,
                use_gpu=True
            )
            print(f"  GPU graph store: is_gpu={graph_store.is_gpu}, device={graph_store.device}")
        else:
            graph_store = StarkSKBAdapter(
                skb, dataset_name,
                adjacency_dict=adjacency_dict,
                cache_path=graph_cache_path
            )
        embedding_provider = StarkPreComputedEmbeddingHandler(query_embs)

        evaluator = Evaluator(
            k_values=[1, 5, 10, 20],
            index_name=dataset_name,
            stats=['mean', 'std']
        )

        # Sample questions
        total_questions = len(qa_data)
        sampled_indices = list(range(min(num_questions, total_questions)))
        print(f"Evaluating {len(sampled_indices)} questions...")

        results_summary = {}
        devices_to_test = []

        if compare:
            if resolved_device == "cuda":
                devices_to_test = ["cuda", "cpu"]
            else:
                devices_to_test = ["cpu"]
                print("Note: GPU not available, comparison disabled")
        else:
            devices_to_test = [resolved_device]

        # Run evaluation for each device
        for test_device in devices_to_test:
            mode_name = "GPU" if test_device == "cuda" else "CPU"
            print(f"\n--- {mode_name} Evaluation ---")

            # Create vector store
            init_start = time.time()
            vector_store = create_vector_store(doc_embs, test_device)
            init_time = time.time() - init_start
            print(f"  Vector store initialized in {init_time:.2f}s")

            # Create retriever
            retriever = SwarmRetriever(
                vector_store=vector_store,
                graph_store=graph_store,
                embedding_provider=embedding_provider,
                cache_neighbors=False,
                use_gpu=(test_device == "cuda"),
            )
            print(f"  Device: {retriever.device}, GPU enabled: {retriever.is_gpu_enabled}")

            # Warmup for GPU
            if test_device == "cuda":
                print("  Warming up GPU...")
                _ = retriever.retrieve(
                    query=qa_data[0][1],
                    n_agents=10,
                    steps=2,
                    top_k=5,
                    movement_strategies=run_config["movement_strategies"],
                    deposit_strategies=run_config["deposit_strategies"],
                    ranking_strategies=run_config["ranking_strategies"],
                )

            # Run evaluation
            eval_result = run_evaluation(
                retriever=retriever,
                evaluator=evaluator,
                qa_data=qa_data,
                sampled_indices=sampled_indices,
                config=run_config,
                verbose=verbose,
                mode_name=mode_name,
            )

            results_summary[mode_name] = eval_result

            # Show aggregated results
            agg_df = evaluator.aggregate_results(eval_result["results"])
            print(f"\n  {mode_name} Results (avg latency: {eval_result['avg_latency']:.4f}s):")
            safe_print_df(agg_df, evaluator)

            # Generate plots if requested
            if plot or save_plots:
                plot_save_dir = save_plots
                if plot_save_dir:
                    os.makedirs(plot_save_dir, exist_ok=True)

                plot_prefix = f"{dataset_name}_{mode_name}"
                reporter.plot_per_query_metrics(
                    eval_result["results"],
                    title=f"{dataset_name} {mode_name} - Per-Query Metrics",
                    save_path=os.path.join(plot_save_dir, f"{plot_prefix}_per_query.png") if plot_save_dir else None
                )
                reporter.plot_latency_distribution(
                    eval_result["results"],
                    title=f"{dataset_name} {mode_name} - Latency Distribution",
                    save_path=os.path.join(plot_save_dir, f"{plot_prefix}_latency.png") if plot_save_dir else None
                )
                reporter.plot_recall_curve(
                    eval_result["results"],
                    title=f"{dataset_name} {mode_name} - Recall@K Curve",
                    save_path=os.path.join(plot_save_dir, f"{plot_prefix}_recall.png") if plot_save_dir else None
                )
                reporter.plot_metrics(
                    agg_df,
                    f"{dataset_name} {mode_name}",
                    save_path=os.path.join(plot_save_dir, f"{plot_prefix}_bar.png") if plot_save_dir else None
                )
                if plot_save_dir:
                    print(f"  Plots saved to {plot_save_dir}/")

            # Add to reporter
            for result in eval_result["results"]:
                reporter.add_run(f"{dataset_name}_{mode_name}", result)

        # Comparison summary
        if compare and len(results_summary) > 1:
            print(f"\n{'='*70}")
            print("COMPARISON SUMMARY")
            print(f"{'='*70}")

            for mode, data in results_summary.items():
                print(f"\n{mode}:")
                print(f"  Avg Latency: {data['avg_latency']:.4f}s")
                print(f"  Total Time: {data['total_time']:.2f}s")

            if "GPU" in results_summary and "CPU" in results_summary:
                gpu_lat = results_summary["GPU"]["avg_latency"]
                cpu_lat = results_summary["CPU"]["avg_latency"]
                speedup = cpu_lat / gpu_lat if gpu_lat > 0 else 0

                print(f"\n*** GPU Speedup: {speedup:.2f}x ***")
                if speedup > 1.0:
                    print(f"    GPU is {speedup:.2f}x faster than CPU")
                else:
                    print(f"    CPU is {1/speedup:.2f}x faster (graph traversal bottleneck)")

                # Quality comparison
                gpu_results = results_summary["GPU"]["results"]
                cpu_results = results_summary["CPU"]["results"]

                gpu_mrr = sum(r['MRR'] for r in gpu_results) / len(gpu_results)
                cpu_mrr = sum(r['MRR'] for r in cpu_results) / len(cpu_results)
                gpu_recall = sum(r['Recall@20'] for r in gpu_results) / len(gpu_results)
                cpu_recall = sum(r['Recall@20'] for r in cpu_results) / len(cpu_results)

                print(f"\n  Quality Comparison:")
                print(f"    MRR:       GPU={gpu_mrr:.4f}, CPU={cpu_mrr:.4f}")
                print(f"    Recall@20: GPU={gpu_recall:.4f}, CPU={cpu_recall:.4f}")

    # Final report
    if len(dataset_names) > 0:
        print(f"\n{'='*70}")
        print("FINAL AGGREGATED RESULTS")
        print(f"{'='*70}")

        aggregated_results = reporter.aggregate(evaluator)
        for name, df in aggregated_results.items():
            print(f"\n--- {name} ---")
            safe_print_df(df, evaluator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SwarmRAG evaluation on Stark datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_n_q.py --n 10                    # Auto-detect device, 10 questions
  python test_n_q.py --n 20 --device gpu       # Force GPU mode
  python test_n_q.py --n 10 --compare          # Compare GPU vs CPU
  python test_n_q.py --n 5 --verbose           # Detailed per-query output
  python test_n_q.py --datasets prime amazon   # Multiple datasets
        """
    )

    parser.add_argument("--datasets", nargs="+", default=["prime"],
                        help="Datasets to test (default: prime)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of questions (default: 10)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--he", "--human-eval", action='store_true',
                        help="Use human-generated QA data")
    parser.add_argument("--device", choices=["auto", "gpu", "cpu", "cuda"],
                        default="auto",
                        help="Device to use; gpu/cuda auto-enables GPU graph store (default: auto)")
    parser.add_argument("--compare", action='store_true',
                        help="Run both GPU and CPU for comparison")
    parser.add_argument("--verbose", "-v", action='store_true',
                        help="Show detailed per-query output")
    parser.add_argument("--plot", action='store_true',
                        help="Display metric graphs after evaluation")
    parser.add_argument("--save-plots", type=str, default=None,
                        help="Directory to save plots (default: display only)")

    # Hyperparameter overrides
    parser.add_argument("--agents", type=int, default=None,
                        help="Override n_agents (default: 25)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override steps (default: 5)")

    args = parser.parse_args()

    # Build config overrides
    config_overrides = {}
    if args.agents:
        config_overrides["n_agents"] = args.agents
    if args.steps:
        config_overrides["steps"] = args.steps

    test_stark(
        dataset_names=args.datasets,
        num_questions=args.n,
        seed=args.seed,
        human_gen=args.he,
        device=args.device,
        compare=args.compare,
        verbose=args.verbose,
        config=config_overrides if config_overrides else None,
        plot=args.plot,
        save_plots=args.save_plots,
    )
