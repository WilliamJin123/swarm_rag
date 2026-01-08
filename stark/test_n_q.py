
import random
import time
import argparse
import os
from typing import List, Optional
import pandas as pd
import logging

# --- Configure logging to see the warnings from VectorStore ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from swarm_rag.core import Heuristics, SwarmRetriever, HeuristicRegistry
from swarm_rag.interfaces.enums import HeuristicKey
from swarm_rag.integrations.stark import StarkInMemoryVectorStore, StarkPreComputedEmbeddingHandler, StarkSKBAdapter
from swarm_rag.eval import Evaluator, EvalReporter
from load_stark import load_and_download_embeddings, load_and_download_skb, load_and_download_qa, precompute_stark_adjacency

def test_stark_questions(
    dataset_names: List[str], 
    num_questions: int = 10, 
    seed: Optional[int] = None,
    human_gen: bool = False
) -> None:
    # ... (function setup is the same) ...
    if seed is not None:
        random.seed(seed)

    reporter = EvalReporter()

    for dataset_name in dataset_names:
        print(f"\n{'=' * 50}")
        print(f"Testing dataset: {dataset_name}")
        print(f"{'=' * 50}")

        # Load Data
        print("Loading QA data...")
        qa_data = load_and_download_qa(dataset_name, human_gen=human_gen)
        
        print("Loading SKB...")
        skb = load_and_download_skb(dataset_name)
        
        print("Loading Embeddings...")
        query_embs, doc_embs = load_and_download_embeddings(dataset_name)
        
        print("Precomputing Adjacency...")
        adjacency_dict = precompute_stark_adjacency(skb, dataset_name)

        # Ensure cache directory exists
        cache_dir = os.path.join("stark", "adjacency_cache")
        os.makedirs(cache_dir, exist_ok=True)
        graph_cache_path = os.path.join(cache_dir, f"graph_{dataset_name}.npz")

        # Initialize Components
        vector_store = StarkInMemoryVectorStore(doc_embs)
        graph_store = StarkSKBAdapter(skb, dataset_name, adjacency_dict=adjacency_dict, cache_path=graph_cache_path)
        embedding_provider = StarkPreComputedEmbeddingHandler(query_embs)

        retriever = SwarmRetriever(
            vector_store=vector_store,
            graph_store=graph_store,
            embedding_provider=embedding_provider,
            cache_neighbors=False
        )

        evaluator = Evaluator(
            k_values=[1, 5, 10, 20], 
            index_name=dataset_name,
            stats=['mean', 'std']    
        )
        total_questions = len(qa_data)

        if num_questions >= total_questions:
            sampled_indices = list(range(total_questions))
        else:
            # sampled_indices = random.sample(range(total_questions), num_questions)
            sampled_indices = list(range(num_questions))
        
        print(f"Starting evaluation on {len(sampled_indices)} questions...")
        query_results = []

        for i, idx in enumerate(sampled_indices):
            query, query_id, answer_ids, _ = qa_data[idx]
            
            print(f"\n--- Processing Query {i + 1}/{len(sampled_indices)} (ID: {query_id}) ---")

            start_time = time.time()
            retrieved_nodes = retriever.retrieve(
                query=query_id,
                n_agents=20,
                steps=4,
                decay=0.5,
                initial_pool_size=30,
                start_subset=10,
                top_k=20,
                movement_strategies={
                    "semantic": (Heuristics.semantic_similarity_unnormalized, 0.35),
                    "centrality": (HeuristicRegistry.get_movement("stark_centrality"), 0.2),
                    "diversity": (Heuristics.pheromone_repulsion, 0.4),
                    "jitter": (Heuristics.random_jitter, 0.05),
                },
                deposit_strategies={
                    "f": (HeuristicRegistry.get(HeuristicKey.FLAT), 1.0)
                },
                ranking_strategies={
                    "visited": (Heuristics.percentage_visited, 0.2),
                    "semantic": (Heuristics.semantic_rank, 0.8),
                },
            )
            latency = time.time() - start_time

            print(f"  Top 3 retrieved IDs: {[node['id'] for node in retrieved_nodes[:3]]}")
            print(f"  Top 3 scores: {['{:.4f}'.format(node['score']) for node in retrieved_nodes[:3]]}")
            print(f"  Ground truth IDs: {answer_ids}")
            print(f"  Ground truth in top 20: {[aid for aid in answer_ids if aid in [n['id'] for n in retrieved_nodes[:20]]]}")

            metrics = evaluator.calculate_metrics(
                retrieved_nodes=retrieved_nodes,
                ground_truth_ids=answer_ids,
                latency_sec=latency,
            )
            
            print(f"  Latency: {latency:.4f}s | Hit@1: {metrics['Hit@1']:.2f} | Hit@5: {metrics['Hit@5']:.2f} | Recall@20: {metrics['Recall@20']:.2f} | MRR: {metrics['MRR']:.2f}")
            query_results.append(metrics)

        display_df = evaluator.aggregate_results(query_results)
        print(f"\n{'='*20} AGGREGATED RESULTS FOR {dataset_name} {'='*20}")
        print(display_df)

        for result in query_results:
            reporter.add_run(dataset_name, result)

        if len(dataset_names) > 0:
            print("\nFinal Comparison Table:")
            
            aggregated_results = reporter.aggregate(evaluator) 
            
            first_valid_df: pd.DataFrame = None
            for name, df in aggregated_results.items():
                print(f"\n--- {name} ---")
                print(df)
                if first_valid_df is None and not df.empty:
                    first_valid_df = df

            if first_valid_df is not None:
                metric_cols = [c for c in first_valid_df.columns if c not in ['Run', 'Dataset', 'index']]
                reporter.plot_comparison(aggregated_results, metrics=metric_cols)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Swarm RAG tests on Stark datasets.")
    parser.add_argument("--datasets", nargs="+", default=["prime"], help="List of datasets to test")
    parser.add_argument("--n", type=int, default=10, help="Number of questions to test")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random sampling for reproducibility")
    parser.add_argument("--he", "--human-eval", action='store_true', help="Use human-generated QA data")

    args = parser.parse_args()
    
    test_stark_questions(args.datasets, args.n, seed=args.seed, human_gen=args.he)