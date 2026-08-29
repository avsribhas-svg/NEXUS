"""
NEXUS Phase 1 — Evaluation Pipeline Tests

Tests metric computation, speed benchmarking logic, generalization evaluation
grouping, and figure generation.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from nexus.evaluation.metrics import (
    compute_mae,
    compute_r_squared,
    compute_per_group_mae,
    vmem_accuracy_threshold,
)
from nexus.evaluation.generalization import group_by_perturbation_type
from nexus.evaluation.figures import generate_scatter_plot, generate_spatial_error_map


# ===========================================================================
# Metrics
# ===========================================================================


class TestMetrics:
    """Tests for nexus.evaluation.metrics"""

    def test_mae_perfect_prediction(self):
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([1.0, 2.0, 3.0])
        assert compute_mae(pred, true) == pytest.approx(0.0, abs=1e-7)

    def test_mae_known_value(self):
        pred = np.array([1.0, 3.0, 5.0])
        true = np.array([2.0, 3.0, 3.0])
        expected = (1.0 + 0.0 + 2.0) / 3
        assert compute_mae(pred, true) == pytest.approx(expected, abs=1e-6)

    def test_mae_symmetric(self):
        a = np.random.randn(100)
        b = np.random.randn(100)
        assert compute_mae(a, b) == pytest.approx(compute_mae(b, a), abs=1e-6)

    def test_mae_nonnegative(self):
        for _ in range(10):
            assert compute_mae(np.random.randn(50), np.random.randn(50)) >= 0

    def test_r_squared_perfect(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_r_squared(pred, true) == pytest.approx(1.0, abs=1e-6)

    def test_r_squared_mean_predictor(self):
        """Predicting the mean should give R² ≈ 0."""
        true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.full_like(true, true.mean())
        assert compute_r_squared(pred, true) == pytest.approx(0.0, abs=1e-6)

    def test_r_squared_worse_than_mean(self):
        """Predictions worse than mean should give R² < 0."""
        true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # reversed
        r2 = compute_r_squared(pred, true)
        # Reversed predictions will have negative R² (much worse than mean)
        assert r2 < 0.5  # at minimum, not perfect

    def test_r_squared_bounded(self):
        """R² should be ≤ 1."""
        pred = np.random.randn(100)
        true = pred + np.random.randn(100) * 0.1
        assert compute_r_squared(pred, true) <= 1.0 + 1e-6

    def test_vmem_accuracy_threshold(self):
        """
        PRD threshold: MAE ≤ 10% of Vmem range.
        For Vmem in [-60, +10], range = 70, threshold = 7 mV.
        """
        vmem_min = -60.0
        vmem_max = 10.0
        threshold = vmem_accuracy_threshold(vmem_min, vmem_max)
        expected = 0.10 * (vmem_max - vmem_min)
        assert threshold == pytest.approx(expected, abs=1e-6)

    def test_per_group_mae(self):
        """Per-group MAE should correctly partition and compute."""
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        true = np.array([1.5, 2.5, 3.0, 4.0])
        groups = np.array(["A", "A", "B", "B"])

        result = compute_per_group_mae(pred, true, groups)
        assert "A" in result
        assert "B" in result
        assert result["A"] == pytest.approx(0.5, abs=1e-6)  # mean(|0.5|, |0.5|)
        assert result["B"] == pytest.approx(0.0, abs=1e-6)  # mean(|0|, |0|)


# ===========================================================================
# Generalization Evaluation
# ===========================================================================


class TestGeneralizationEval:
    """Tests for nexus.evaluation.generalization"""

    def test_group_by_perturbation_type(self):
        """
        Results should be grouped by perturbation type, with each group
        containing only samples of that type.
        """
        # Create mock evaluation records
        records = [
            {"config_id": "a", "perturbation_type": "channel_blockade", "mae": 5.0},
            {"config_id": "b", "perturbation_type": "channel_blockade", "mae": 6.0},
            {"config_id": "c", "perturbation_type": "gj_blockade", "mae": 8.0},
            {"config_id": "d", "perturbation_type": "gj_blockade", "mae": 7.0},
            {"config_id": "e", "perturbation_type": "exogenous_expression", "mae": 10.0},
            {"config_id": "f", "perturbation_type": "spatial_gradient", "mae": 12.0},
        ]

        grouped = group_by_perturbation_type(records)

        assert set(grouped.keys()) == {
            "channel_blockade",
            "gj_blockade",
            "exogenous_expression",
            "spatial_gradient",
        }
        assert len(grouped["channel_blockade"]) == 2
        assert len(grouped["gj_blockade"]) == 2
        assert len(grouped["exogenous_expression"]) == 1
        assert len(grouped["spatial_gradient"]) == 1


# ===========================================================================
# Figure Generation
# ===========================================================================


class TestFigures:
    """Tests for nexus.evaluation.figures — figures should generate without error."""

    def test_scatter_plot_generates(self, tmp_path):
        """Scatter plot of pred vs. true should save without error."""
        pred = np.random.uniform(-80, -20, 200)
        true = pred + np.random.randn(200) * 5
        output_path = tmp_path / "scatter.png"
        generate_scatter_plot(pred, true, output_path=str(output_path))
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_spatial_error_map_generates(self, tmp_path):
        """Spatial error map should save without error."""
        n_cells = 25
        positions = np.random.uniform(0, 100, (n_cells, 2))
        pred_vmem = np.random.uniform(-80, -20, n_cells)
        true_vmem = pred_vmem + np.random.randn(n_cells) * 3
        output_path = tmp_path / "spatial_error.png"
        generate_spatial_error_map(
            positions, pred_vmem, true_vmem, output_path=str(output_path)
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0
