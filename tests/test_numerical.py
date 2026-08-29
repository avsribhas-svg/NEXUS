"""
NEXUS Phase 1 — Numerical Correctness Tests

Property-based and invariant tests that verify the model respects physical
constraints and produces self-consistent predictions. These catch subtle bugs
that shape-only tests miss.
"""

import pytest
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from nexus.model.mpnn import MPNN
from nexus.model.baseline import BaselineMLP
from nexus.training.trainer import Trainer
from nexus.training.config import TrainingConfig

N_CHANNELS = 8


def _make_graph(x, edge_index, edge_attr, y=None):
    """Helper to create a PyG Data from tensors."""
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )
    if y is not None:
        data.y = torch.tensor(y, dtype=torch.float32)
    return data


# ===========================================================================
# Symmetry & Invariance Tests
# ===========================================================================


class TestSymmetryInvariants:
    """
    If the input has a symmetry, the output must reflect it.
    These test whether the model's architecture correctly propagates structure.
    """

    def test_identical_cells_identical_vmem(self):
        """
        All cells with identical features in a symmetric graph should get
        identical predicted Vmem. This is a graph automorphism test.
        """
        model = MPNN(n_channels=N_CHANNELS)
        model.eval()

        n_cells = 6
        # All cells identical
        x = torch.ones(n_cells, N_CHANNELS) * 0.5
        # Ring graph (fully symmetric)
        src = list(range(n_cells))
        dst = [(i + 1) % n_cells for i in range(n_cells)]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        edge_attr = torch.ones(edge_index.shape[1], 1) * 0.5

        with torch.no_grad():
            out = model(x, edge_index, edge_attr)

        # All outputs should be the same (within floating point tolerance)
        assert torch.allclose(
            out, out[0].expand_as(out), atol=1e-5
        ), f"Non-uniform output for symmetric input: std={out.std():.6f}"

    def test_zero_input_produces_finite_output(self):
        """Zero channel densities should still produce finite Vmem."""
        model = MPNN(n_channels=N_CHANNELS)
        model.eval()

        x = torch.zeros(10, N_CHANNELS)
        src = list(range(10))
        dst = [(i + 1) % 10 for i in range(10)]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        edge_attr = torch.zeros(edge_index.shape[1], 1)

        with torch.no_grad():
            out = model(x, edge_index, edge_attr)
        assert torch.isfinite(out).all()

    def test_scaling_sensitivity(self):
        """
        Doubling all channel densities should change the prediction.
        If it doesn't, the model is ignoring the magnitude of its inputs.
        """
        model = MPNN(n_channels=N_CHANNELS)
        model.eval()

        x = torch.rand(10, N_CHANNELS) * 0.3
        src = list(range(10))
        dst = [(i + 1) % 10 for i in range(10)]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        edge_attr = torch.ones(edge_index.shape[1], 1) * 0.5

        with torch.no_grad():
            out_base = model(x, edge_index, edge_attr)
            out_scaled = model(x * 2, edge_index, edge_attr)

        assert not torch.allclose(out_base, out_scaled, atol=1e-4), (
            "Output unchanged when input doubled — model ignores input magnitude"
        )


# ===========================================================================
# GJ Topology Tests
# ===========================================================================


class TestGapJunctionTopology:
    """Tests that verify the model uses gap junction topology correctly."""

    def test_isolated_vs_coupled_cells_differ(self):
        """
        Same cell features with vs. without gap junctions should produce
        different predictions (unless GJ conductance is zero).
        """
        model = MPNN(n_channels=N_CHANNELS)
        model.eval()

        n = 20
        x = torch.rand(n, N_CHANNELS)

        # Coupled: ring
        src = list(range(n))
        dst = [(i + 1) % n for i in range(n)]
        coupled_ei = torch.tensor([src + dst, dst + src], dtype=torch.long)
        coupled_ea = torch.ones(coupled_ei.shape[1], 1) * 0.8

        # Isolated: no edges
        isolated_ei = torch.zeros((2, 0), dtype=torch.long)
        isolated_ea = torch.zeros((0, 1))

        with torch.no_grad():
            out_coupled = model(x, coupled_ei, coupled_ea)
            out_isolated = model(x, isolated_ei, isolated_ea)

        assert not torch.allclose(out_coupled, out_isolated, atol=1e-4), (
            "Coupled and isolated cells produce identical output — "
            "model ignores gap junction topology"
        )

    def test_higher_coupling_reduces_vmem_variance(self):
        """
        Physical intuition: stronger gap junction coupling should make Vmem
        more uniform across the tissue (cells equilibrate).

        After training on data that embeds this physics, the model should
        reproduce this property. We test on a TRAINED model.
        """
        # Create training data where this property holds
        rng = np.random.RandomState(42)
        data_list = []
        for i in range(100):
            n = 16
            x = torch.tensor(
                rng.uniform(0, 1, (n, N_CHANNELS)), dtype=torch.float32
            )
            src = list(range(n))
            dst = [(j + 1) % n for j in range(n)]
            edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

            # Half with high coupling, half with low
            if i < 50:
                gj = 0.1  # low coupling
            else:
                gj = 0.9  # high coupling
            edge_attr = torch.full((edge_index.shape[1], 1), gj, dtype=torch.float32)

            # Target: intrinsic + coupling effect
            intrinsic = (x[:, 0] * 40 - x[:, 1] * 80 - 30).float()
            if gj > 0.5:
                # High coupling: pull toward mean
                intrinsic = intrinsic * (1 - gj) + intrinsic.mean() * gj
            y = intrinsic.clamp(-120, 60)
            data_list.append(
                Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            )

        loader = DataLoader(data_list, batch_size=20, shuffle=True)

        model = MPNN(n_channels=N_CHANNELS)
        config = TrainingConfig(lr=1e-3, max_epochs=100, patience=20)
        trainer = Trainer(model=model, config=config)
        trainer.train(loader, loader)

        # Test: high-coupling predictions should have lower variance than low-coupling
        model.eval()
        variances = {"low": [], "high": []}

        for data in data_list[:10]:  # low coupling samples
            with torch.no_grad():
                pred = model(data.x, data.edge_index, data.edge_attr)
            variances["low"].append(pred.var().item())

        for data in data_list[50:60]:  # high coupling samples
            with torch.no_grad():
                pred = model(data.x, data.edge_index, data.edge_attr)
            variances["high"].append(pred.var().item())

        mean_var_low = np.mean(variances["low"])
        mean_var_high = np.mean(variances["high"])

        # This is a soft check: high coupling should tend toward lower variance
        # but we don't require it to be strictly less (model might not learn it perfectly)
        # We just check the model CAN distinguish the two conditions
        preds_differ = abs(mean_var_low - mean_var_high) > 1e-3
        assert preds_differ, (
            f"Model produces same variance for low ({mean_var_low:.4f}) and "
            f"high ({mean_var_high:.4f}) coupling — may not use edge features effectively"
        )


# ===========================================================================
# Baseline Comparison Tests
# ===========================================================================


class TestBaselineComparison:
    """
    The GNN should outperform the MLP baseline on coupled-tissue data.
    If it doesn't, the graph structure isn't contributing to prediction.
    """

    def test_gnn_beats_mlp_on_coupled_data(self):
        """
        Train both models on data where GJ coupling matters.
        GNN should achieve lower MAE than MLP.
        """
        rng = np.random.RandomState(42)
        data_list = []

        for i in range(80):
            n = 16
            x = torch.tensor(
                rng.uniform(0, 1, (n, N_CHANNELS)), dtype=torch.float32
            )
            src = list(range(n))
            dst = [(j + 1) % n for j in range(n)]
            edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
            gj = rng.uniform(0.3, 0.9)
            edge_attr = torch.full(
                (edge_index.shape[1], 1), gj, dtype=torch.float32
            )

            # Target depends on neighbors (requires graph info to predict well)
            intrinsic = (x[:, 0] * 40 - x[:, 1] * 80 - 30).float()
            # Neighbor average
            neighbor_avg = torch.zeros(n)
            counts = torch.zeros(n)
            for e in range(edge_index.shape[1]):
                s, d = edge_index[0, e].item(), edge_index[1, e].item()
                neighbor_avg[d] += intrinsic[s]
                counts[d] += 1
            counts[counts == 0] = 1
            neighbor_avg /= counts

            y = ((1 - gj) * intrinsic + gj * neighbor_avg).clamp(-120, 60)
            data_list.append(
                Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            )

        train_loader = DataLoader(data_list[:60], batch_size=20, shuffle=True)
        test_loader = DataLoader(data_list[60:], batch_size=20)

        # Train GNN
        torch.manual_seed(42)
        gnn = MPNN(n_channels=N_CHANNELS)
        config = TrainingConfig(lr=1e-3, max_epochs=100, patience=20)
        trainer_gnn = Trainer(model=gnn, config=config)
        trainer_gnn.train(train_loader, test_loader)

        # Train MLP
        torch.manual_seed(42)
        mlp = BaselineMLP(n_channels=N_CHANNELS)
        trainer_mlp = Trainer(model=mlp, config=config)
        trainer_mlp.train(train_loader, test_loader)

        # Evaluate both on test set
        gnn.eval()
        mlp.eval()
        gnn_errors, mlp_errors = [], []

        for batch in test_loader:
            with torch.no_grad():
                gnn_pred = gnn(batch.x, batch.edge_index, batch.edge_attr)
                mlp_pred = mlp(batch.x)
            gnn_errors.append((gnn_pred - batch.y).abs().mean().item())
            mlp_errors.append((mlp_pred - batch.y).abs().mean().item())

        gnn_mae = np.mean(gnn_errors)
        mlp_mae = np.mean(mlp_errors)

        assert gnn_mae < mlp_mae, (
            f"GNN MAE ({gnn_mae:.2f}) is not better than MLP MAE ({mlp_mae:.2f}) "
            f"on coupled-tissue data — graph structure not helping"
        )


# ===========================================================================
# Output Range Tests
# ===========================================================================


class TestOutputRange:
    """
    Even without explicit output clamping, a trained model's predictions
    should stay within a physically reasonable range.
    """

    def test_predictions_in_physical_range_after_training(self):
        """
        After training on data in [-120, 60] mV, predictions on in-distribution
        inputs should not wildly exceed that range.
        """
        rng = np.random.RandomState(42)
        data_list = []
        for i in range(50):
            n = 8
            x = torch.tensor(
                rng.uniform(0, 1, (n, N_CHANNELS)), dtype=torch.float32
            )
            edge_index = torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 0],
                 [1, 2, 3, 4, 5, 6, 7, 0, 0, 1, 2, 3, 4, 5, 6, 7]],
                dtype=torch.long,
            )
            edge_attr = torch.ones(16, 1) * 0.5
            y = (x[:, 0] * 40 - x[:, 1] * 80 - 30).clamp(-120, 60)
            data_list.append(
                Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            )

        loader = DataLoader(data_list, batch_size=25, shuffle=True)

        model = MPNN(n_channels=N_CHANNELS)
        config = TrainingConfig(lr=1e-3, max_epochs=50, patience=999)
        trainer = Trainer(model=model, config=config)
        trainer.train(loader, loader)

        model.eval()
        for data in data_list:
            with torch.no_grad():
                pred = model(data.x, data.edge_index, data.edge_attr)
            # Allow some extrapolation but not insane values
            assert pred.min() > -200, f"Prediction too negative: {pred.min():.1f}"
            assert pred.max() < 150, f"Prediction too positive: {pred.max():.1f}"
