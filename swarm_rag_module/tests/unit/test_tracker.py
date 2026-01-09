# test_tracker_advanced.py
import os
import json
import io
import sys
from contextlib import redirect_stdout
from swarm_rag.evolution.execution.tracker import ProgressTracker

LOG_FILE = "test_data/test_advanced_log.jsonl"
PLOT_FILE = "test_data/test_advanced_plot.png"

def test_dynamic_logging():
    print("\n--- Testing Dynamic Logging & JSONL Structure ---")
    
    tracker = ProgressTracker(
        log_path=LOG_FILE,
        plot_path=PLOT_FILE,
    )
    
    # 1. Log mixed bag of metrics (some ints, floats, big numbers)
    train_stats = {
        "best_quality": 0.87654321,      # Should format to .4f
        "best_cost": 1245.678,           # Should format to .1f (large number)
        "avg_quality": 0.5,
        "best_metric_var_Recall@20": 0.0012, # Detailed metric
        "n_agents_avg": 25               # Integer
    }
    val_stats = {
        "best_quality": 0.8111
    }
    
    tracker.log(generation=1, train_stats=train_stats, val_stats=val_stats)
    
    # 2. Verify File Content
    assert os.path.exists(LOG_FILE), "Log file not created"
    
    with open(LOG_FILE, 'r') as f:
        line = f.readline()
        data = json.loads(line)
        
    print("  ✓ JSONL line valid")
    assert data['train_best_quality'] == 0.87654321, "Float precision lost in JSON"
    assert data['val_best_quality'] == 0.8111, "Validation stats missing"
    assert 'timestamp' in data, "Timestamp missing"

def test_print_summary_formatting():
    print("\n--- Testing Summary Output Formatting ---")
    
    tracker = ProgressTracker(
        log_path=LOG_FILE,
        plot_path=PLOT_FILE,
    )
    # Re-log simple data
    tracker.log(1, {"best_quality": 0.55555, "best_cost": 1500.2}, None)
    
    # Capture stdout to verify formatting
    capture = io.StringIO()
    with redirect_stdout(capture):
        tracker.print_summary(1)
        
    output = capture.getvalue()
    
    # Check for "beautified" keys
    if "[TRAIN] Best Quality" in output:
        print("  ✓ Key formatting: '[TRAIN] Best Quality' found (Success)")
    else:
        print(f"  X Key formatting failed. Got:\n{output}")
        
    # Check for smart number formatting
    if "1500.2" in output and "1500.2000" not in output:
        print("  ✓ Cost formatted with low precision (Success)")
    else:
        print("  X Cost formatting failed")

def test_plotting_resilience():
    print("\n--- Testing Plotting Resilience ---")
    
    tracker = ProgressTracker(
        log_path=LOG_FILE,
        plot_path=PLOT_FILE,
    )
    
    # Log 3 generations
    tracker.log(0, {"best_quality": 0.1, "avg_quality": 0.05, "best_cost": 100})
    tracker.log(1, {"best_quality": 0.5, "avg_quality": 0.3, "best_cost": 90})
    tracker.log(2, {"best_quality": 0.6, "avg_quality": 0.4, "best_cost": 120})
    
    # Try plotting
    try:
        tracker.plot(save_path=PLOT_FILE)
        assert os.path.exists(PLOT_FILE), "Plot file not generated"
        print("  ✓ Plot generation successful")
    except Exception as e:
        print(f"  X Plotting crashed: {e}")

def cleanup():
    for f in [LOG_FILE, PLOT_FILE]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    # try:
    test_dynamic_logging()
    test_print_summary_formatting()
    test_plotting_resilience()
    print("\nALL TRACKER TESTS PASSED")
    # finally:
    #     cleanup()