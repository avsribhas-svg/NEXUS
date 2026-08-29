"""
NEXUS Phase 1 — Model Architecture Tests

Tests the MPNN and baseline MLP: forward pass shapes, gradient flow,
permutation equivariance, edge feature sensitivity, and parameter count.
"""

import pytest
import numpy as np
import torch
from torch_geometric.data import Data, Batch

from nexus.model.mpnn import MPNN
from nexus.model.baseline import BaselineMLP
from nexus.model.losses import mae_loss, physics_auxiliary_loss

N_CHANNELS = 8


def _make_pyg_graph(n_cells=32, n_channels=N_CHANNELS, gj_conductance=20.0, seed=42):
    """Create a single PyG Data object for testing."""
    rng = np.random.RandomState(seed)
    x = torch.tensor(rng.uniform(0, 1, (n_cells, n_channels)), dtype=torch.float32)

    # Simple ring topology for predictable structure
    src = list(range(n_cells))
    dst = [(i + 1) % n_cells for i in range(n_cells)]
    # Undirected
    edge_index = torch.tensor(
        [src + dst, dst + src], dtype=torch.long
    )
    n_edges = edge_index.shape[1]
    edge_attr = torch.full((n_edges, 1), gj_conductance / 50.0, dtype=torch.float32)

    y = torch.tensor(
        rng.uniform(-80, -20, n_cells), dtype=torch.float32
    )

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def _make_batch(graphs):
    """Batch multiple PyG Data objects."""
    return Batch.from_data_list(graphs)


# ===========================================================================
# MPNN Tests
# ===========================================================================


class TestMPNN:
    """Tests for nexus.model.mpnn.MPNN"""

    def test_forward_pass_shape_single(self):
        """Output shape should be (n_cells,) for a single graph."""
        model = MPNN(n_channels=N_CHANNELS)
        graph = _make_pyg_graph(n_cells=32)
        out = model(graph.x, graph.edge_index, graph.edge_attr)
        assert out.shape == (32,), f"Expected (32,), got {out.shape}"

    def test_forward_pass_shape_batch(self):
        """Output shape should be (total_nodes,) for a batched graph."""
        model = MPNN(n_channels=N_CHANNELS)
        graphs = [_make_pyg_graph(n_cells=n, seed=i) for i, n in enumerate([16, 32, 24])]
        batch = _make_batch(graphs)
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        total_nodes = 16 + 32 + 24
        assert out.shape == (total_nodes,), f"Expected ({total_nodes},), got {out.shape}"

    def test_forward_pass_variable_sizes(self):
        """Model should handle graphs of different sizes in one batch."""
        model = MPNN(n_channels=N_CHANNELS)
        sizes = [10, 50, 100, 200, 500]
        for n in sizes:
            graph = _make_pyg_graph(n_cells=n, seed=n)
            out = model(graph.x, graph.edge_index, graph.edge_attr)
            assert out.shape == (n,), f"Failed for n_cells={n}: got {out.shape}"

    def test_output_dtype(self):
        """Output should be float32."""
        model = MPNN(n_channels=N_CHANNELS)
        graph = _make_pyg_graph()
        out = model(graph.x, graph.edge_index, graph.edge_attr)
        assert out.dtype == torch.float32

    def test_output_finite(self):
        """No NaN or Inf in output."""
        model = MPNN(n_channels=N_CHANNELS)
        graph = _make_pyg_graph()
        out = model(graph.x, graph.edge_index, graph.edge_attr)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_gradient_flows_to_all_parameters(self):
        """Loss.backward() should produce non-None gradients for all parameters."""
        model = MPNN(n_channels=N_CHANNELS)
        graph = _make_pyg_graph()
        out = model(graph.x, graph.edge_index, graph.edge_attr)
        loss = out.mean()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_gradient_nonzero(self):
        """At least some gradients should be nonzero (model is not a dead network)."""
        model = MPNN(n_channels=N_CHANNELS)
        graph = _make_pyg_graph()
        out = model(graph.x, graph.edge_index, graph.edge_attr)
        loss = out.sum()
        loss.backward()
        any_nonzero = any(
            param.grad.abs().sum() > 0 for param in model.parameters() if param.grad is not None
        )
        assert any_nonzero, "All gradients are zero — dead network"

    def test_permutation_equivariance(self):
        """
        Reordering nodes should reorder outputs identically.
        This is a fundamental property of message-passing GNNs.
        """
        model = MPNN(n_channels=N_CHANNELS)
        model.eval()

        graph = _make_pyg_graph(n_cells=20, seed=42)

        # Random permutation
        perm = torch.randperm(20)
        inv_perm = torch.argsort(perm)

        # Permute graph
        x_perm = graph.x[perm]
        # Remap edge indices
        edge_index_perm = inv_perm[graph.edge_index]  # ← remap node IDs
        # Actually, we need to map old indices to new indices
        # If node old_i is now at position perm[old_i], then:
        # New edge (perm[src], perm[dst]) should have same weight
        remap = torch.zeros(20, dtype=torch.long)
        remap[perm] = torch.arange(20)  # remap[old_id] = new_position... no.
        # perm[i] = old node that is now at position i
        # We want: old node j is now at position inv_perm[j]
        edge_index_perm = torch.stack([
            inv_perm[graph.edge_index[0]],
            inv_perm[graph.edge_index[1]],
        ])

        with torch.no_grad():
            out_orig = model(graph.x, graph.edge_index, graph.edge_attr)
            out_perm = model(x_perm, edge_index_perm, graph.edge_attr)

        # out_perm[inv_perm[i]] should equal out_orig[i]
        out_orig_reordered = out_orig[perm]  # = out at position of perm[i] in original
        # Actually: out_perm[new_pos] should match out_orig[old_pos]
        # where new_pos = inv_perm[old_pos]
        # So out_perm[inv_perm] should == out_orig
        torch.testing.assert_close(
            out_perm[inv_perm], out_orig, atol=1e-5, rtol=1e-5,
        )

    def test_edge_features_affect_output(self):
        """
        Changing edge features (GJ conductance) should change the output.
        This verifies the model actually uses edge information.
        """
        model = MPNN(n_channels=N_CHANNELS)
        model.eval()
        graph = _make_pyg_graph(gj_conductance=20.0)

        with torch.no_grad():
            out_low = model(graph.x, graph.edge_index, graph.edge_attr)
            # Change edge attrs to high conductance
            high_edge_attr = torch.ones_like(graph.edge_attr)
            out_high = model(graph.x, graph.edge_index, high_edge_attr)

        assert not torch.allclose(out_low, out_high, atol=1e-6), (
            "Output unchanged when edge features changed — model ignores edges"
        )

    def test_no_edges_still_works(self):
        """Model should handle a graph with zero edges (isolated cells)."""
        model = MPNN(n_channels=N_CHANNELS)
        x = torch.randn(10, N_CHANNELS)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1))
        out = model(x, edge_index, edge_attr)
        assert out.shape == (10,)
        assert torch.isfinite(out).all()

    def test_single_cell_works(self):
        """Model should handle a graph with a single node."""
        model = MPNN(n_channels=N_CHANNELS)
        x = torch.randn(1, N_CHANNELS)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1))
        out = model(x, edge_index, edge_attr)
        assert out.shape == (1,)

    def test_parameter_count_reasonable(self):
        """
        PRD estimates ~500K parameters. Allow 100K–2M range.
        Too few = model too small. Too many = wrong architecture.
        """
        model = MPNN(n_channels=N_CHANNELS)
        n_params = sum(p.numel() for p in model.parameters())
        assert 100_000 <= n_params <= 2_000_000, (
            f"Parameter count {n_params:,} outside expected range [100K, 2M]"
        )

    def test_residual_connections_active(self):
        """
        With residual connections, deeper models should not degrade to zero output.
        Compare a 2-layer and 6-layer model; both should produce nonzero output.
        """
        for n_layers in [2, 6]:
            model = MPNN(n_channels=N_CHANNELS, n_layers=n_layers)
            graph = _make_pyg_graph()
            out = model(graph.x, graph.edge_index, graph.edge_attr)
            assert out.abs().mean() > 1e-6, (
                f"Output near-zero with {n_layers} layers — residual connections broken"
            )

    def test_deterministic_with_eval_mode(self):
        """Same input should produce identical output in eval mode (no dropout stochasticity)."""
        model = MPNN(n_channels=N_CHANNELS)
        model.eval()
        graph = _make_pyg_graph()
        with torch.no_grad():
            out1 = model(graph.x, graph.edge_index, graph.edge_attr)
            out2 = model(graph.x, graph.edge_index, graph.edge_attr)
        torch.testing.assert_close(out1, out2)


# ===========================================================================
# Baseline MLP Tests
# ===========================================================================


class TestBaselineMLP:
    """Tests for nexus.model.baseline.BaselineMLP"""

    def test_forward_pass_shape(self):
        """Output shape should be (n_cells,)."""
        model = BaselineMLP(n_channels=N_CHANNELS)
        x = torch.randn(32, N_CHANNELS)
        out = model(x)
        assert out.shape == (32,), f"Expected (32,), got {out.shape}"

    def test_per_cell_independence(self):
        """
        MLP processes each cell independently. Changing one cell's features
        should not change another cell's prediction.
        """
        model = BaselineMLP(n_channels=N_CHANNELS)
        model.eval()
        x = torch.randn(10, N_CHANNELS)

        with torch.no_grad():
            out1 = model(x).clone()
            # Modify cell 5
            x_mod = x.clone()
            x_mod[5] = torch.randn(N_CHANNELS)
            out2 = model(x_mod)

        # All cells except 5 should be unchanged
        for i in range(10):
            if i != 5:
                assert torch.allclose(out1[i], out2[i], atol=1e-6), (
                    f"Cell {i} changed when cell 5 was modified"
                )

    def test_gradient_flows(self):
        model = BaselineMLP(n_channels=N_CHANNELS)
        x = torch.randn(16, N_CHANNELS)
        out = model(x)
        out.mean().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ===========================================================================
# Loss Function Tests
# ===========================================================================


class TestLosses:
    """Tests for nexus.model.losses"""

    def test_mae_loss_zero_on_exact(self):
        """MAE should be 0 when prediction equals target."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        loss = mae_loss(pred, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_mae_loss_correct_value(self):
        """MAE should be mean of absolute differences."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 2.0, 5.0])
        loss = mae_loss(pred, target)
        expected = (1.0 + 0.0 + 2.0) / 3
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    def test_mae_loss_symmetric(self):
        """MAE(a, b) == MAE(b, a)."""
        a = torch.randn(100)
        b = torch.randn(100)
        assert mae_loss(a, b).item() == pytest.approx(mae_loss(b, a).item(), abs=1e-6)

    def test_mae_loss_nonnegative(self):
        """MAE is always >= 0."""
        for _ in range(10):
            loss = mae_loss(torch.randn(50), torch.randn(50))
            assert loss.item() >= 0

    def test_physics_loss_zero_for_constant_vmem(self):
        """
        If all predicted Vmem are identical, gap junction currents are zero,
        so the physics loss should be zero.
        """
        n_cells = 20
        pred = torch.full((n_cells,), -50.0)
        # Ring topology
        src = list(range(n_cells))
        dst = [(i + 1) % n_cells for i in range(n_cells)]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        conductances = torch.ones(edge_index.shape[1])

        loss = physics_auxiliary_loss(pred, edge_index, conductances)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_physics_loss_nonzero_for_varying_vmem(self):
        """Non-constant Vmem with nonzero GJ conductance should produce nonzero physics loss."""
        n_cells = 20
        pred = torch.linspace(-80, -20, n_cells)
        src = list(range(n_cells))
        dst = [(i + 1) % n_cells for i in range(n_cells)]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        conductances = torch.ones(edge_index.shape[1])

        loss = physics_auxiliary_loss(pred, edge_index, conductances)
        assert loss.item() > 0

    def test_physics_loss_zero_with_zero_conductance(self):
        """Zero GJ conductance should produce zero physics loss regardless of Vmem."""
        n_cells = 20
        pred = torch.linspace(-80, -20, n_cells)
        src = list(range(n_cells))
        dst = [(i + 1) % n_cells for i in range(n_cells)]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        conductances = torch.zeros(edge_index.shape[1])

        loss = physics_auxiliary_loss(pred, edge_index, conductances)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
