import pandas as pd
from typing import List, Dict, Any

class   Evaluator:
    def __init__(self, k_values=[1, 5, 10, 20]):
        self.k_values = k_values

    def calculate_metrics(self, 
        retrieved_nodes: List[Dict[str, Any]], 
        ground_truth_ids: List[str], 
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
        retrieved_ids = [str(n['id']) for n in retrieved_nodes]
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
            if len(gt_set) > 0:
                metrics[f"Recall@{k}"] = num_hits / len(gt_set)
            else:
                metrics[f"Recall@{k}"] = 0.0

        # 3. MRR (Mean Reciprocal Rank)
        # Look for the FIRST relevant item in the entire retrieved list
        mrr = 0.0
        for rank, rid in enumerate(retrieved_ids):
            if rid in gt_set:
                mrr = 1.0 / (rank + 1)
                break
        metrics["MRR"] = mrr

        # 4. Diversity Metrics (Based on Top 20)
        # We use K=20 as the standard cutoff for diversity
        top_20_nodes = retrieved_nodes[:20]
        
        # Diversity Count: How many unique node types are in the top 20?
        # e.g., finding just "papers" vs finding "papers", "authors", and "institutions"
        node_types = [n.get('node_type', 'unknown') for n in top_20_nodes]
        metrics["Diversity_Count"] = len(set(node_types))
        
        # DR@20 (Diversity Recall)
        # In the GraphFlow paper context, this often equals Recall if 'aspects' aren't explicitly defined.
        # We will map it to Recall@20 to match your table, but you can customize this
        # if you have specific 'aspect' labels in your ground truth.
        metrics["DR@20"] = metrics["Recall@20"]

        return metrics

    def aggregate_results(self, results: List[Dict[str, float]]) -> pd.DataFrame:
        """
        Averages the metrics across all queries to produce the final table.
        """
        df = pd.DataFrame(results)
        
        # Calculate mean for all numeric columns
        summary = df.mean().to_dict()
        
        # Format for readability
        formatted = {
            "Hit@1": summary.get("Hit@1", 0),
            "Hit@5": summary.get("Hit@5", 0),
            "Hit@10": summary.get("Hit@10", 0),
            "Hit@20": summary.get("Hit@20", 0),
            "MRR": summary.get("MRR", 0),
            "Recall@20": summary.get("Recall@20", 0),
            "DR@20": summary.get("DR@20", 0),
            "Diversity_Count": summary.get("Diversity_Count", 0),
            "latency": summary.get("latency", 0)
        }
        
        return pd.DataFrame([formatted], index=["Enhanced_SwarmRAG"])