from typing import Dict, Any, Optional, List
import pandas as pd
import matplotlib.pyplot as plt
import torch
import re
import os

from .metrics import Evaluator

DEFAULT_METRICS = [
    'Hit@1', 'Hit@5', 'Hit@10', 'Hit@20',
    'Recall@1', 'Recall@5', 'Recall@10', 'Recall@20',
    'MRR'
]

class EvalReporter:
    def __init__(self, metrics = None):
        self.results_by_group = {}
        self.metrics = metrics or DEFAULT_METRICS

    def add_run(self, group: str, metrics: Dict[str, Any]):
        self.results_by_group.setdefault(group, []).append(metrics)

    def aggregate(self, evaluator: Evaluator):
        aggregated = {}
        for group, runs in self.results_by_group.items():
            aggregated[group] = evaluator.aggregate_results(runs)
        return aggregated

    def plot_metrics(self, df, title: str, metrics: Optional[list] = None, save_path: str = None):
        plot_metrics(df, title, metrics or self.metrics, save_path=save_path)

    def plot_comparison(self, aggregated_results: Dict[str, pd.DataFrame], metrics: Optional[list] = None, save_path: str = None):
        plot_comparison(aggregated_results, metrics or self.metrics, save_path=save_path)

    def plot_per_query_metrics(self, results: List[Dict], title: str = "Per-Query Metrics",
                               metrics: List[str] = None, save_path: str = None):
        """Line plot of metrics across queries (latency, Hit@K, Recall@K, MRR)."""
        metrics = metrics or ['latency', 'Hit@5', 'Recall@20', 'MRR']
        n_queries = len(results)

        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3*len(metrics)), sharex=True)
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            values = [r.get(metric, 0) for r in results]
            values_tensor = torch.tensor(values, dtype=torch.float32)
            mean_val = torch.mean(values_tensor).item()
            ax.plot(range(1, n_queries + 1), values, marker='o', markersize=3)
            ax.axhline(y=mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.3f}')
            ax.set_ylabel(metric)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Query Index')
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_latency_distribution(self, results: List[Dict], title: str = "Latency Distribution",
                                  bins: int = 20, save_path: str = None):
        """Histogram showing distribution of query latencies."""
        latencies = [r['latency'] for r in results]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(latencies, bins=bins, edgecolor='black', alpha=0.7)

        # Add statistical markers
        latencies_tensor = torch.tensor(latencies, dtype=torch.float32)
        mean_lat = torch.mean(latencies_tensor).item()
        median_lat = torch.median(latencies_tensor).item()
        ax.axvline(mean_lat, color='r', linestyle='--', label=f'Mean: {mean_lat:.3f}s')
        ax.axvline(median_lat, color='g', linestyle='--', label=f'Median: {median_lat:.3f}s')

        ax.set_xlabel('Latency (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_recall_curve(self, results: List[Dict], title: str = "Recall@K Curve",
                          save_path: str = None):
        """Line plot of Recall@1, @5, @10, @20 (like precision-recall curve)."""
        k_values = [1, 5, 10, 20]
        means = []
        stds = []

        for k in k_values:
            values = [r.get(f'Recall@{k}', 0) for r in results]
            values_tensor = torch.tensor(values, dtype=torch.float32)
            means.append(torch.mean(values_tensor).item())
            stds.append(torch.std(values_tensor).item())

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_values, means, 'b-o', linewidth=2, markersize=8, label='Mean Recall@K')
        ax.fill_between(k_values,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, label='±1 Std Dev')

        ax.set_xlabel('K')
        ax.set_ylabel('Recall@K')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(k_values)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_all(self, results: List[Dict], aggregated_df: pd.DataFrame,
                 dataset_name: str, save_dir: str = None):
        """Generate all available plots."""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # 1. Bar chart of aggregated metrics
        self.plot_metrics(
            aggregated_df,
            f"{dataset_name} - Aggregated Metrics",
            save_path=os.path.join(save_dir, f"{dataset_name}_bar.png") if save_dir else None
        )

        # 2. Per-query line plot
        self.plot_per_query_metrics(
            results,
            title=f"{dataset_name} - Per-Query Metrics",
            save_path=os.path.join(save_dir, f"{dataset_name}_per_query.png") if save_dir else None
        )

        # 3. Latency histogram
        self.plot_latency_distribution(
            results,
            title=f"{dataset_name} - Latency Distribution",
            save_path=os.path.join(save_dir, f"{dataset_name}_latency.png") if save_dir else None
        )

        # 4. Recall@K curve
        self.plot_recall_curve(
            results,
            title=f"{dataset_name} - Recall@K Curve",
            save_path=os.path.join(save_dir, f"{dataset_name}_recall.png") if save_dir else None
        )

def _extract_metric_value(val_str: str) -> float:
    """Extracts the primary metric value (mean) from the formatted string."""
    # Look for the first number (float or int) at the start of the string
    match = re.match(r"^([\d\.]+)", str(val_str))
    if match:
        return float(match.group(1))
    return 0.0

def plot_metrics(df: pd.DataFrame, dataset_name: str, metrics: list, save_path: str = None):
    """Generate bar plots for evaluation metrics of a single dataset."""
    # Extract values from formatted strings
    values = []
    for metric in metrics:
        if metric in df.columns:
            val_str = df[metric].iloc[0]
            values.append(_extract_metric_value(val_str))
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
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_comparison(all_results: Dict[str, pd.DataFrame], metrics: list, save_path: str = None):
    """Generate comparison plots for multiple datasets."""

    # Prepare data for plotting
    plot_data = {}
    for metric in metrics:
        values = []
        labels = []
        for dataset, df in all_results.items():
            if metric in df.columns:
                val_str = df[metric].iloc[0]
                values.append(_extract_metric_value(val_str))
                labels.append(dataset)
        plot_data[metric] = (labels, values)

    # Create subplots
    n_metrics = len(plot_data)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols  # ceiling division
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows))
    axes = axes.flatten()  # flatten in case multiple rows

    for idx, (metric, (labels, values)) in enumerate(plot_data.items()):
        ax = axes[idx]
        bars = ax.bar(labels, values, color='lightcoral', edgecolor='darkred')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')

        ax.set_title(metric, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Remove any unused subplots
    for ax in axes[n_metrics:]:
        ax.remove()

    plt.suptitle('Metric Comparison Across Datasets', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()