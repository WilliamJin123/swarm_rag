import pandas as pd
from typing import List, Dict, Any, Union
from .metric_functions import MetricFunctions
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
        

    def calculate_metrics(
        self, 
        retrieved_nodes: List[Dict[str, Any]], 
        ground_truth_ids: List[Union[str, int]], 
        latency_sec: float
    ) -> Dict[str, float]:
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
        gt_ids = [str(g) for g in ground_truth_ids]
        gt_set = set(gt_ids)
        
        metrics: Dict[str, float] = {}
        metrics["latency"] = latency_sec
        
        # 2. Hit@K and Recall@K
        for k in self.k_values:
            metrics[f"Hit@{k}"] = MetricFunctions.hit_at_k(k)(
                retrieved_ids, gt_ids
            )
            metrics[f"Recall@{k}"] = MetricFunctions.recall_at_k(k)(
                retrieved_ids, gt_ids
            )

        # 3. MRR (Mean Reciprocal Rank)
        # Look for the FIRST relevant item in the entire retrieved list
        metrics["MRR"] = MetricFunctions.mrr(
            retrieved_ids, gt_ids
        )

        # 4. Diversity Metrics
        cutoff = min(self.diversity_cutoff, len(retrieved_nodes))

        node_types = {
            retrieved_nodes[i].get("node_type", "unknown")
            for i in range(cutoff)
        }
        metrics["Diversity_Node_Types"] = float(len(node_types))

        metrics["Diversity_Count"] = float(
            len(set(retrieved_ids[:cutoff]) & gt_set)
        )
        
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