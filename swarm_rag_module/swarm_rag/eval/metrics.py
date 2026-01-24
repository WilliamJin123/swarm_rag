import pandas as pd
import torch
from typing import List, Dict, Any, TypedDict, Union
from .metric_functions import MetricFunctions

class RetrievedNode(TypedDict):
    """A node returned by the retriever."""
    id: Union[str, int] 
    node_type: str     

Metrics = TypedDict('Metrics', {
    # Core fields
    'latency': float,
    'MRR': float,
    'Diversity_Node_Types': float,
    'Diversity_Count': float,
    
    # Hit@K / Recall@K
    'Hit@1': float,
    'Recall@1': float,
    'Hit@5': float,
    'Recall@5': float,
    'Hit@10': float,
    'Recall@10': float,
    'Hit@20': float,
    'Recall@20': float,
}, total=False)

def iqr(x):
    t = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x.float()
    return (torch.quantile(t, 0.75) - torch.quantile(t, 0.25)).item()

def rng(x):  # range
    t = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x.float()
    return (torch.max(t) - torch.min(t)).item()

class Evaluator:
    def __init__(
        self, 
        k_values=[1, 5, 10, 20], 
        diversity_cutoff=20, 
        index_name="Swarm_RAG",
        stats: List[str] = None
    ):
        """
        Args:
            k_values: List of k values for Hit@K and Recall@K
            diversity_cutoff: Number of top results to consider for diversity metrics
            index_name: Name to use in the final results DataFrame
            stats: List of statistics to aggregate. Options: 
                   ['mean', 'std', 'var', 'min', 'max', 'range', 'median', 'iqr'].
                   Defaults to ['mean', 'std'].
        """
        self.k_values = k_values
        self.diversity_cutoff = diversity_cutoff
        self.index_name = index_name
        self.stats = stats if stats is not None else ['mean', 'std']
        

    def calculate_metrics(
        self, 
        retrieved_nodes: List[RetrievedNode],
        ground_truth_ids: List[Union[str, int]], 
        latency_sec: float
    ) -> Metrics:
        """
        Computes all metrics for a SINGLE query.
        
        Args:
            retrieved_nodes: List of dicts, each must have 'id' and 'node_type'.
                             Ordered by relevance (Top 1 first).
            ground_truth_ids: List of strings (the correct node IDs).
            latency_sec: Time taken to retrieve.
        Returns:

        """
        # 1. Setup
        retrieved_ids = [str(n['id']) for n in retrieved_nodes if 'id' in n]
        gt_ids = [str(g) for g in ground_truth_ids]
        gt_set = set(gt_ids)
        
        metrics: Metrics = {}
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

        # Per-node correctness tracking
        node_results = []
        for rank, node in enumerate(retrieved_nodes, 1):
            node_id = str(node['id']) if 'id' in node else None
            if node_id:
                node_results.append({
                    'id': node['id'],
                    'rank': rank,
                    'correct': node_id in gt_set,
                    'score': node.get('score', None)
                })

        metrics['node_results'] = node_results

        return metrics

    def aggregate_results(self, results: List[Metrics]) -> pd.DataFrame:
        """
        Aggregates the metrics across all queries to produce the final table
        based on the configured stats.
        """
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return pd.DataFrame()
        
        # Define custom aggregations
        agg_funcs = {}

        # Map string stats to pandas/numpy compatible functions
        valid_stats = {
            'mean': 'mean',
            'std': 'std',
            'var': 'var',
            'min': 'min',
            'max': 'max',
            'median': 'median',
            'sum': 'sum',
            'count': 'count',
            'iqr': iqr,
            'range': rng
        }

        # Filter valid stats
        selected_stats = [s for s in self.stats if s in valid_stats]
        if not selected_stats:
            selected_stats = ['mean', 'std'] # Fallback

        # We use a list of functions for aggregation
        aggs_to_run = [valid_stats[s] for s in selected_stats]
        
        summary = df[numeric_cols].agg(aggs_to_run).T
        
        summary.columns = selected_stats

        formatted = {}
        for metric in summary.index:
            parts = []
            # We prioritize mean and std/var at the front
            
            val_mean = summary.loc[metric, 'mean'] if 'mean' in selected_stats else None
            val_std = summary.loc[metric, 'std'] if 'std' in selected_stats else None
            
            # Primary part
            if val_mean is not None:
                if val_std is not None:
                    # Check for NaN std (single sample)
                    if pd.isna(val_std):
                        parts.append(f"{val_mean:.4f}")
                    else:
                        parts.append(f"{val_mean:.4f} σ {val_std:.4f}")
                else:
                    parts.append(f"{val_mean:.4f}")
            
            extras = []
            for stat in selected_stats:
                if stat in ['mean', 'std']: continue # Already handled
                
                val = summary.loc[metric, stat]
                if pd.isna(val): continue
                
                extras.append(f"{stat}: {val:.4f}")
            
            if extras:
                parts.append(f"[{' | '.join(extras)}]")
            
            formatted[metric] = " ".join(parts)

        return pd.DataFrame([formatted], index=[self.index_name])

    def format_results(self, df: pd.DataFrame, style: str = "vertical") -> str:
        """
        Format aggregated results for terminal display.

        Args:
            df: DataFrame from aggregate_results()
            style: "vertical" (line-by-line) or "table" (original wide table)

        Returns:
            Formatted string for terminal display
        """
        if style == "table":
            return df.to_string()

        # Vertical format - each metric on its own line
        lines = []
        if len(df) == 0:
            return "No results"

        idx_name = df.index[0]
        lines.append(f"[{idx_name}]")
        lines.append("-" * 50)

        # Group metrics by category for readability
        metric_groups = {
            "Hit@K": ["Hit@1", "Hit@5", "Hit@10", "Hit@20"],
            "Recall@K": ["Recall@1", "Recall@5", "Recall@10", "Recall@20"],
            "Other": ["MRR", "latency", "Diversity_Node_Types", "Diversity_Count"]
        }

        for group_name, metrics in metric_groups.items():
            group_has_values = False
            group_lines = []
            for metric in metrics:
                if metric in df.columns:
                    val = df[metric].iloc[0]
                    group_lines.append(f"  {metric:20s}: {val}")
                    group_has_values = True
            if group_has_values:
                lines.append(f"\n{group_name}:")
                lines.extend(group_lines)

        return "\n".join(lines)