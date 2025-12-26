import pandas as pd
from typing import List, Dict, Any, Union
import numpy as np

class Evaluator:
    def __init__(self, k_values=[1, 5, 10, 20], diversity_cutoff=20, index_name="Swarm_RAG"):
        """
        Args:
            k_values: List of k values for Hit@K and Recall@K
            diversity_cutoff: Number of top results to consider for diversity metrics
            index_name: Name to use in the final results DataFrame
        """
        self.k_values = k_values
        self.diversity_cutoff = diversity_cutoff
        self.index_name = index_name
        

    def calculate_metrics(self, 
        retrieved_nodes: List[Dict[str, Any]], 
        ground_truth_ids: List[Union[str, int]], 
        latency_sec: float) -> Dict[str, float]:
        """
        Computes all metrics for a SINGLE query.
        
        Args:
            retrieved_nodes: List of dicts, each must have 'id' and 'node_type'.
                             Ordered by relevance (Top 1 first).
            ground_truth_ids: List of strings (the correct node IDs).
            latency_sec: Time taken to retrieve.
        """
        # 1. Setup
        retrieved_ids = [str(n['id']) for n in retrieved_nodes if 'id' in n]
        gt_set = set([str(g) for g in ground_truth_ids])
        
        metrics = {"latency": latency_sec}
        
        # 2. Hit@K and Recall@K
        for k in self.k_values:
            # Slice the top K
            top_k_ids = retrieved_ids[:k]
            
            # Intersection (How many relevant items found?)
            hits = set(top_k_ids).intersection(gt_set)
            num_hits = len(hits)
            
            # Hit@K (Binary: Did we find at least one?)
            metrics[f"Hit@{k}"] = 1.0 if num_hits > 0 else 0.0
            
            # Recall@K (Fraction of total relevant items found)
            metrics[f"Recall@{k}"] = num_hits / len(gt_set) if gt_set else 0.0

        # 3. MRR (Mean Reciprocal Rank)
        # Look for the FIRST relevant item in the entire retrieved list
        mrr = 0.0
        for rank, rid in enumerate(retrieved_ids):
            if rid in gt_set:
                mrr = 1.0 / (rank + 1)
                break
        metrics["MRR"] = mrr

        # 4. Diversity Metrics
        cutoff = min(self.diversity_cutoff, len(retrieved_nodes))
        top_nodes = retrieved_nodes[:cutoff]
        
        # Count unique node types
        node_types = [n.get('node_type', 'unknown') for n in top_nodes]
        metrics["Diversity_Node_Types"] = len(set(node_types))

        unique_relevant = set(retrieved_ids[:20]).intersection(gt_set)
        metrics["Diversity_Count"] = len(unique_relevant)
        
        
        # DR@K (Diversity Recall) - using the same cutoff as diversity
        dr_k = min(20, len(retrieved_ids))  # Standard DR@20
        if dr_k > 0:
            top_k_ids = retrieved_ids[:dr_k]
            hits = set(top_k_ids).intersection(gt_set)
            metrics["DR@20"] = len(hits) / len(gt_set) if gt_set else 0.0
        else:
            metrics["DR@20"] = 0.0

        return metrics

    def aggregate_results(self, results: List[Dict[str, float]]) -> pd.DataFrame:
        """
        Averages the metrics across all queries to produce the final table.
        """
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return pd.DataFrame()
        
        # Calculate mean and std for all numeric columns
        summary = df[numeric_cols].agg(['mean', 'std']).T
        
        # Format for readability - include both mean and std
        formatted = {}
        for metric in summary.index:
            mean_val = summary.loc[metric, 'mean']
            std_val = summary.loc[metric, 'std']
            formatted[metric] = (
                f"{mean_val:.4f}" if pd.isna(std_val) else f"{mean_val:.4f} ± {std_val:.4f}"
            )

        return pd.DataFrame([formatted], index=[self.index_name])