import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union

from swarm_rag.core import Heuristics, SwarmRetriever
from swarm_rag.integrations.stark import StarkInMemoryVectorStore, StarkPreComputedEmbeddingHandler, StarkSKBAdapter
from swarm_rag.eval.metrics import Evaluator
from load_stark import load_and_download_embeddings, load_and_download_skb, load_and_download_qa
from stark_qa.skb import SKB

# def stark_locality_deposit(ctx):
#     graph: SKB = ctx["graph"]
#     target_id = ctx["target_id"]
#     graph.

# Heuristics.stark_locality_deposit = stark_locality_deposit

def test_first_10_questions(dataset_names: List[str]):
    """
    Test the first 10 questions of each QA dataset and generate evaluation metrics and plots.
    
    Args:
        dataset_names: List of dataset names to test
    """
    # Initialize results storage
    all_results = {}
    
    for dataset_name in dataset_names:
        print(f"\n{'='*50}")
        print(f"Testing dataset: {dataset_name}")
        print(f"{'='*50}")
        
        # Load dataset components
        qa_data = load_and_download_qa(dataset_name)
        skb = load_and_download_skb(dataset_name)
        query_embs, doc_embs = load_and_download_embeddings(dataset_name)
        
        # Initialize components
        vector_store = StarkInMemoryVectorStore(doc_embs)
        graph_store = StarkSKBAdapter(skb)
        embedding_provider = StarkPreComputedEmbeddingHandler(query_embs)
        
        # Initialize retriever with collaborative amplification deposit strategy
        retriever = SwarmRetriever(
            vector_store=vector_store,
            graph_store=graph_store,
            embedding_provider=embedding_provider,
            max_workers=8
        )
        
        # Initialize evaluator
        evaluator = Evaluator(k_values=[1, 5, 10, 20], index_name=dataset_name)
        
        # Process first 10 questions (or fewer if dataset is smaller)
        num_questions = min(10, len(qa_data))
        query_results = []
        
        for i in range(num_questions):
            query, query_id, answer_ids, _ = qa_data[i]
            
            print(f"\nProcessing question {i+1}/{num_questions}: {query[:150]}...")
            
            # Measure retrieval time
            start_time = time.time()
            retrieved_nodes = retriever.retrieve(
                query=query_id,
                n_agents=20,
                steps=4,
                decay=0.5,
                initial_pool_size=30,
                start_subset=10,
                top_k=20,
                movement_strategies = {
                "semantic": (Heuristics.semantic_similarity, 0.3),
                "centrality": (graph_store.log_degree_centrality, 0.4),
                "diversity": (Heuristics.pheromone_repulsion, 0.3)
                },
                deposit_strategies={
                    "collaborative": (Heuristics.deposit_collaborative_amplification, 1.0)
                }
            )
            latency = time.time() - start_time
            
            # Evaluate results
            metrics = evaluator.calculate_metrics(
                retrieved_nodes=retrieved_nodes,
                ground_truth_ids=answer_ids,
                latency_sec=latency
            )
            query_results.append(metrics)
            
            # Print key metrics
            print(f"  Hit@1: {metrics['Hit@1']:.2f}, Hit@5: {metrics['Hit@5']:.2f}, "
                  f"MRR: {metrics['MRR']:.3f}, Latency: {latency:.3f}s")
        
        # Aggregate results for this dataset
        aggregated = evaluator.aggregate_results(query_results)
        all_results[dataset_name] = aggregated
        
        # Print aggregated results
        print(f"\nAggregated results for {dataset_name}:")
        print(aggregated.to_string(index=False))
        
        # Generate plots for this dataset
        plot_metrics(aggregated, dataset_name)
    
    # Generate comparison plots if multiple datasets
    if len(dataset_names) > 1:
        plot_comparison(all_results)

def plot_metrics(df: pd.DataFrame, dataset_name: str):
    """Generate bar plots for evaluation metrics of a single dataset."""
    metrics = ['Hit@1', 'Hit@5', 'Hit@10', 'Hit@20', 
               'Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'MRR']
    
    # Extract values from formatted strings
    values = []
    for metric in metrics:
        if metric in df.columns:
            val_str = df[metric].iloc[0]
            values.append(float(val_str.split(' ± ')[0]))
        else:
            values.append(0.0)
    
    # Create figure
    plt.figure(figsize=(15, 6))
    bars = plt.bar(metrics, values, color='skyblue', edgecolor='navy')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom')
    
    # Formatting
    plt.title(f'Evaluation Metrics for {dataset_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def plot_comparison(all_results: Dict[str, pd.DataFrame]):
    """Generate comparison plots for multiple datasets."""
    metrics = ['Hit@1', 'Hit@5', 'Hit@10', 'Hit@20', 
               'Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'MRR']
    
    # Prepare data for plotting
    plot_data = {}
    for metric in metrics:
        values = []
        labels = []
        for dataset, df in all_results.items():
            if metric in df.columns:
                val_str = df[metric].iloc[0]
                values.append(float(val_str.split(' ± ')[0]))
                labels.append(dataset)
        plot_data[metric] = (labels, values)
    
    # Create subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric, (labels, values)) in enumerate(plot_data.items()):
        ax = axes[idx]
        bars = ax.bar(labels, values, color='lightcoral', edgecolor='darkred')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Formatting
        ax.set_title(metric, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle('Metric Comparison Across Datasets', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    test_first_10_questions(["prime"])
