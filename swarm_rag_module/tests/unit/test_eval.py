import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

# Adjust imports to match project structure
from swarm_rag.eval.metrics import Evaluator, RetrievedNode
from swarm_rag.eval.report import EvalReporter, _extract_metric_value, plot_metrics, plot_comparison

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator(
            k_values=[1, 5],
            diversity_cutoff=5,
            index_name="TestIndex",
            stats=['mean', 'std', 'min', 'max']
        )

    def test_calculate_metrics_perfect_match(self):
        """Test metrics when retrieved nodes perfectly match ground truth."""
        # GT: 1, 2, 3
        # Retrieved: 1, 2, 3, 4, 5
        retrieved = [
            {'id': 1, 'node_type': 'A'},
            {'id': 2, 'node_type': 'B'},
            {'id': 3, 'node_type': 'A'},
            {'id': 4, 'node_type': 'C'},
            {'id': 5, 'node_type': 'B'}
        ]
        gt = [1, 2, 3]
        metrics = self.evaluator.calculate_metrics(retrieved, gt, 0.1)

        self.assertEqual(metrics['Hit@1'], 1.0)
        self.assertAlmostEqual(metrics['Recall@1'], 1/3) # 1 found out of 3
        self.assertEqual(metrics['Hit@5'], 1.0)
        self.assertEqual(metrics['Recall@5'], 1.0) # 3 found out of 3
        self.assertEqual(metrics['MRR'], 1.0) # First match at rank 1
        self.assertEqual(metrics['Diversity_Node_Types'], 3.0) # A, B, C in top 5
        self.assertEqual(metrics['Diversity_Count'], 3.0) # 1, 2, 3 intersect
        self.assertEqual(metrics['latency'], 0.1)

    def test_calculate_metrics_partial_match(self):
        """Test metrics with partial overlap."""
        # GT: 1, 2
        # Retrieved: 3, 1, 4 (1 matches at pos 2, which is rank 2)
        retrieved = [
            {'id': 3, 'node_type': 'A'},
            {'id': 1, 'node_type': 'B'},
            {'id': 4, 'node_type': 'C'}
        ]
        gt = [1, 2]
        metrics = self.evaluator.calculate_metrics(retrieved, gt, 0.2)

        self.assertEqual(metrics['Hit@1'], 0.0)
        self.assertEqual(metrics['Hit@5'], 1.0) # Found within top 5
        self.assertEqual(metrics['MRR'], 0.5) # Rank 2 -> 1/2
        self.assertAlmostEqual(metrics['Recall@5'], 0.5) # 1 out of 2 found

    def test_calculate_metrics_no_match(self):
        """Test metrics with zero overlap."""
        retrieved = [{'id': 10}, {'id': 11}]
        gt = [1, 2]
        metrics = self.evaluator.calculate_metrics(retrieved, gt, 0.1)
        self.assertEqual(metrics['Hit@1'], 0.0)
        self.assertEqual(metrics['Recall@5'], 0.0)
        self.assertEqual(metrics['MRR'], 0.0)

    def test_aggregate_results_formatting_with_sigma(self):
        """Test that mean and std are formatted with the Greek sigma symbol."""
        # Result 1: Hit@1 = 1.0
        # Result 2: Hit@1 = 0.0
        # Mean = 0.5, Std = 0.7071
        
        results = [
            {'Hit@1': 1.0, 'latency': 0.1},
            {'Hit@1': 0.0, 'latency': 0.2}
        ]
        
        # Override stats to just mean/std for simpler check
        self.evaluator.stats = ['mean', 'std']
        df = self.evaluator.aggregate_results(results)
        
        self.assertIn('Hit@1', df.columns)
        val = df.loc['TestIndex', 'Hit@1']
        
        # Check format "0.5000 σ 0.7071"
        self.assertIn("σ", val)
        parts = val.split(" σ ")
        self.assertEqual(len(parts), 2)
        self.assertAlmostEqual(float(parts[0]), 0.5, places=3)
        self.assertAlmostEqual(float(parts[1]), 0.7071, places=3)

    def test_aggregate_results_single_sample(self):
        """Test aggregation with a single sample (std is NaN)."""
        # Std should be NaN for single sample, so it should handle it gracefully
        results = [{'Hit@1': 1.0}]
        self.evaluator.stats = ['mean', 'std']
        df = self.evaluator.aggregate_results(results)
        val = df.loc['TestIndex', 'Hit@1']
        
        # Should just be mean, no sigma if std is nan
        self.assertEqual(val, "1.0000")

    def test_aggregate_results_extra_stats(self):
        """Test aggregation with extra stats like min/max."""
        self.evaluator.stats = ['mean', 'min', 'max']
        results = [{'Hit@1': 1.0}, {'Hit@1': 0.0}, {'Hit@1': 0.5}]
        # Mean = 0.5, Min = 0.0, Max = 1.0
        df = self.evaluator.aggregate_results(results)
        val = df.loc['TestIndex', 'Hit@1']
        
        # Format: "0.5000 [min: 0.0000 | max: 1.0000]"
        self.assertTrue(val.startswith("0.5000"))
        self.assertIn("[min: 0.0000 | max: 1.0000]", val)

    def test_empty_results(self):
        """Test aggregation with empty results."""
        df = self.evaluator.aggregate_results([])
        self.assertTrue(df.empty)

    def test_empty_retrieved_list(self):
        """Test metrics with empty retrieved list."""
        metrics = self.evaluator.calculate_metrics([], [1, 2, 3], 0.1)
        self.assertEqual(metrics['Hit@1'], 0.0)
        self.assertEqual(metrics['MRR'], 0.0)

    def test_empty_ground_truth(self):
        """Test metrics with empty ground truth."""
        metrics = self.evaluator.calculate_metrics([{'id': 1}], [], 0.1)
        self.assertEqual(metrics['Hit@1'], 0.0)

class TestEvalReporter(unittest.TestCase):
    def setUp(self):
        self.reporter = EvalReporter()

    def test_extract_metric_value(self):
        """Test the regex extraction of metric values."""
        self.assertEqual(_extract_metric_value("0.5000 σ 0.1000"), 0.5)
        self.assertEqual(_extract_metric_value("0.1234"), 0.1234)
        self.assertEqual(_extract_metric_value("0.9 [min:0]"), 0.9)
        self.assertEqual(_extract_metric_value("invalid"), 0.0)
        self.assertEqual(_extract_metric_value(""), 0.0)

    def test_reporter_aggregation(self):
        """Test the reporter's aggregate method."""
        evaluator = Evaluator(k_values=[1], stats=['mean'])
        
        self.reporter.add_run("GroupA", {'Hit@1': 1.0})
        self.reporter.add_run("GroupA", {'Hit@1': 0.0})
        self.reporter.add_run("GroupB", {'Hit@1': 1.0})
        
        aggregated = self.reporter.aggregate(evaluator)
        
        self.assertIn("GroupA", aggregated)
        self.assertIn("GroupB", aggregated)
        
        val_a = aggregated["GroupA"].iloc[0]['Hit@1']
        self.assertEqual(val_a, "0.5000") # Mean of 1.0 and 0.0
        
        val_b = aggregated["GroupB"].iloc[0]['Hit@1']
        self.assertEqual(val_b, "1.0000")

    @patch('swarm_rag.eval.report.plt')
    def test_plot_metrics(self, mock_plt):
        """Test plot_metrics calls matplotlib correctly."""
        df = pd.DataFrame({'Hit@1': ["0.8000 σ 0.1"], 'Hit@5': ["0.9000"]})
        plot_metrics(df, "Dataset1", ['Hit@1', 'Hit@5'])
        
        mock_plt.figure.assert_called_once()
        mock_plt.bar.assert_called_once()
        mock_plt.show.assert_called_once()

    @patch('swarm_rag.eval.report.plt')
    def test_plot_comparison(self, mock_plt):
        """Test plot_comparison calls matplotlib correctly."""
        df1 = pd.DataFrame({'Hit@1': ["0.8"]})
        df2 = pd.DataFrame({'Hit@1': ["0.9"]})
        data = {"D1": df1, "D2": df2}
        
        # Mock subplots to return figure and axes array
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        
        # The code does axes.flatten()
        mock_axes_array = MagicMock()
        mock_axes_array.flatten.return_value = [mock_ax] * 10 
        
        mock_plt.subplots.return_value = (mock_fig, mock_axes_array)
        
        plot_comparison(data, ['Hit@1'])
        
        mock_plt.subplots.assert_called_once()
        # Should plot bars on the axes
        mock_ax.bar.assert_called()
        mock_plt.show.assert_called_once()

if __name__ == "__main__":
    unittest.main()
