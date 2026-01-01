import json
import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any, Optional

class ProgressTracker:
    def __init__(self, log_path: str = "evolution_log.jsonl"):
        self.log_path = log_path
        self.history: List[Dict[str, Any]] = []
        # Clear previous log
        with open(self.log_path, "w") as f:
            pass

    def log(self, generation: int, train_stats: Dict[str, float], val_stats: Optional[Dict[str, float]] = None):
        """
        Logs a single step of evolution.
        """
        entry = {
            "generation": generation,
            "timestamp": time.time(),
            **{f"train_{k}": v for k, v in train_stats.items()},
        }
        
        if val_stats:
            entry.update({f"val_{k}": v for k, v in val_stats.items()})

        self.history.append(entry)
        
        # Write to JSONL
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def plot(self, save_path: str = "evolution_progress.png"):
        """
        Generates a training vs validation graph.
        """
        if not self.history:
            return

        df = pd.DataFrame(self.history)
        
        plt.figure(figsize=(10, 6))
        
        # Plot Fitness
        plt.plot(df["generation"], df["train_best_fitness"], label="Train Best Fitness", marker="o")
        plt.plot(df["generation"], df["train_avg_fitness"], label="Train Avg Fitness", linestyle="--", alpha=0.7)
        
        if "val_best_fitness" in df.columns and df["val_best_fitness"].notna().any():
            # Interpolate validation points since they might be sparse
            val_df = df.dropna(subset=["val_best_fitness"])
            plt.plot(val_df["generation"], val_df["val_best_fitness"], label="Validation Fitness", marker="x", linewidth=2)

        plt.xlabel("Generation")
        plt.ylabel("Fitness Score")
        plt.title("Evolutionary Progress: Training vs Validation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path)
        plt.close()
        print(f"Progress plot saved to {save_path}")

    def print_summary(self, generation: int):
        """Prints a clean summary to console."""
        latest = self.history[-1]
        print(f"--- Gen {generation} Summary ---")
        print(f"  Train Best: {latest.get('train_best_fitness', 0):.4f}")
        print(f"  Train Avg:  {latest.get('train_avg_fitness', 0):.4f}")
        if 'val_best_fitness' in latest:
            print(f"   VAL SCORE: {latest['val_best_fitness']:.4f}")
        print("----------------------------")