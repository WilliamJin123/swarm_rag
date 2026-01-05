import json
import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any, Optional

class ProgressTracker:
    def __init__(
        self, 
        log_path: str = "evolution_log.jsonl", 
        plot_path: str = "evolution_progress.png",
        plot_title: str = "Evolution Progress",
        overwrite: bool = True
    ):
        self.log_path = log_path
        self.plot_path = plot_path
        self.plot_title = plot_title
        self.history: List[Dict[str, Any]] = []
        if overwrite:
            with open(self.log_path, "w") as f:
                pass
        else:
            print(f"  [Tracker] Appending to existing log: {log_path}")

    def log(self, generation: int, train_stats: Dict[str, float], val_stats: Optional[Dict[str, float]] = None, save_path: str = None):
        """
        Logs a single step of evolution.
        """
        if save_path is None:
            save_path = self.log_path
        entry = {
            "generation": generation,
            "timestamp": time.time(),
            **{f"train_{k}": v for k, v in train_stats.items()},
        }
        
        if val_stats:
            entry.update({f"val_{k}": v for k, v in val_stats.items()})

        self.history.append(entry)
        
        # Write to JSONL
        with open(save_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def plot(
        self, 
        save_path: str = None, 
        title: str = None
    ):
        """
        Generates a training vs validation graph.
        """
        if not self.history:
            print("No history to plot.")
            return

        if save_path is None:
            save_path = self.plot_path
        if title is None:
            title = self.plot_title


        df = pd.DataFrame(self.history)
        
        # Create a plot with 2 Y-axes (Quality on left, Cost on right)
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # --- Axis 1: Quality (Maximize) ---
        color = 'tab:blue'
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Quality Score (Higher is Better)', color=color)
        
        # Safe plotting (looks for 'train_best_quality', fallbacks to 'train_best_fitness' if legacy)
        y_train_qual = df.get("train_best_quality", df.get("train_best_fitness"))
        y_train_avg = df.get("train_avg_quality", df.get("train_avg_fitness"))
        
        if y_train_qual is not None:
            ax1.plot(df["generation"], y_train_qual, label="Train Best Quality", color=color, marker="o")
            ax1.plot(df["generation"], y_train_avg, label="Train Avg Quality", color=color, linestyle="--", alpha=0.5)
        
        # Validation Quality
        if "val_best_quality" in df.columns:
            val_df = df.dropna(subset=["val_best_quality"])
            ax1.plot(val_df["generation"], val_df["val_best_quality"], label="Val Quality", color="tab:green", marker="x", linewidth=2)
            
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        # --- Axis 2: Cost (Minimize) ---
        if "train_best_cost" in df.columns:
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:red'
            ax2.set_ylabel('Cost / Latency (ms) (Lower is Better)', color=color)
            ax2.plot(df["generation"], df["train_best_cost"], label="Train Cost", color=color, linestyle=":", alpha=0.6)
            ax2.tick_params(axis='y', labelcolor=color)

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = (ax2.get_legend_handles_labels() if "train_best_cost" in df.columns else ([], []))
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Progress plot saved to {save_path}")

    def print_summary(self, generation: int):
        if not self.history:
            print("No history to summarize.")
            return

        latest = self.history[-1]
        print(f"--- Gen {generation} Summary ---")

        # Sort keys to ensure Train comes before Val, and metrics are alphabetical
        sorted_keys = sorted(latest.keys())

        for key in sorted_keys:
            # Skip metadata keys
            if key in ["generation", "timestamp"]:
                continue

            val = latest[key]
            
            # 1. Smart Value Formatting
            if isinstance(val, (int, float)):
                # If it looks like latency/cost (large number) or integer, use less precision
                if "cost" in key or "latency" in key or abs(val) > 100:
                    val_str = f"{val:.1f}"
                elif isinstance(val, int) or val.is_integer():
                    val_str = f"{int(val)}"
                else:
                    # Standard metric precision
                    val_str = f"{val:.4f}"
            else:
                val_str = str(val)

            # 2. Key Beautification
            # Convert "train_best_quality" -> "Train Best Quality"
            if key.startswith("train_"):
                clean_key = key.replace("train_", "").replace("_", " ").title()
                prefix = "[TRAIN]"
            elif key.startswith("val_"):
                clean_key = key.replace("val_", "").replace("_", " ").title()
                prefix = "[ VAL ]"
            else:
                clean_key = key.replace("_", " ").title()
                prefix = "[     ]"

            # 3. Print aligned
            print(f"  {prefix} {clean_key:<25} : {val_str}")

        print("-" * 45)