import time
import argparse
from typing import List, Optional

import pandas as pd

from swarm_rag.core import Heuristics, SwarmRetriever, HeuristicRegistry
from swarm_rag.integrations.stark import StarkInMemoryVectorStore, StarkPreComputedEmbeddingHandler, StarkSKBAdapter
from swarm_rag.eval import Evaluator, EvalReporter
from load_stark import load_and_download_embeddings, load_and_download_skb, load_and_download_qa, precompute_stark_adjacency

def test_stark_questions(dataset_names: List[str], num_questions: int = 10) -> None:
    """
    Test the first n questions of each QA dataset and generate evaluation metrics and plots.
    
    Args:
        dataset_names: List of dataset names to test (e.g., ["prime", "amazon"])
        num_questions: Number of questions to process from the start of the dataset.
    """
    reporter = EvalReporter()

    for dataset_name in dataset_names:
        print(f"\n{'=' * 50}")
        print(f"Testing dataset: {dataset_name}")
        print(f"{'=' * 50}")

        # Load Data
        print("Loading QA data...")
        qa_data = load_and_download_qa(dataset_name)
        
        print("Loading SKB...")
        skb = load_and_download_skb(dataset_name)
        
        print("Loading Embeddings...")
        query_embs, doc_embs = load_and_download_embeddings(dataset_name)
        
        print("Precomputing Adjacency...")
        adjacency_dict = precompute_stark_adjacency(skb, dataset_name)

        # Initialize Components
        vector_store = StarkInMemoryVectorStore(doc_embs)
        graph_store = StarkSKBAdapter(skb, dataset_name, adjacency_dict=adjacency_dict)
        embedding_provider = StarkPreComputedEmbeddingHandler(query_embs)

        retriever = SwarmRetriever(
            vector_store=vector_store,
            graph_store=graph_store,
            embedding_provider=embedding_provider,
            cache_neighbors=False,
        )

        # Initialize Evaluator
        evaluator = Evaluator(k_values=[1, 5, 10, 20], index_name=dataset_name)

        actual_num_questions = min(num_questions, len(qa_data))
        print(f"Starting evaluation on first {actual_num_questions} questions...")
        
        print(f"Starting evaluation on first {actual_num_questions} questions...")

        query_results = []

        for i in range(actual_num_questions):
            query, query_id, answer_ids, _ = qa_data[i]
            
            if i % 5 == 0: print(f"Processing {i + 1}/{actual_num_questions}...")

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
                    "semantic": (Heuristics.semantic_similarity, 0.35),
                    "centrality": (HeuristicRegistry.get_movement("stark_centrality"), 0.2),
                    "diversity": (Heuristics.pheromone_repulsion, 0.4),
                    "jitter": (Heuristics.random_jitter, 0.05),
                },
                deposit_strategies={
                    "semantic_deposit": (Heuristics.deposit_semantic, 1.0)
                },
                ranking_strategies={
                    "visited": (Heuristics.percentage_visited, 0.6),
                    "semantic": (Heuristics.semantic_rank, 0.4),
                },
            )
            latency = time.time() - start_time

            metrics = evaluator.calculate_metrics(
                retrieved_nodes=retrieved_nodes,
                ground_truth_ids=answer_ids,
                latency_sec=latency,
            )
            
            query_results.append(metrics)

        display_df = evaluator.aggregate_results(query_results)
        print(f"\nAggregated results for {dataset_name}:")
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
    
    args = parser.parse_args()
    
    test_stark_questions(args.datasets, args.n)