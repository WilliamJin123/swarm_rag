from typing import Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt

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

    def plot_metrics(self, df, title: str, metrics: Optional[list] = None):
        plot_metrics(df, title, metrics or self.metrics)

    def plot_comparison(self, aggregated_results: Dict[str, pd.DataFrame], metrics: Optional[list] = None):
        plot_comparison(aggregated_results, metrics or self.metrics)

def plot_metrics(df: pd.DataFrame, dataset_name: str, metrics: list):
    """Generate bar plots for evaluation metrics of a single dataset."""
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

def plot_comparison(all_results: Dict[str, pd.DataFrame], metrics: list):
    """Generate comparison plots for multiple datasets."""
    
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
    plt.show()