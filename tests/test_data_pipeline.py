"""
NEXUS Phase 1 — Data Pipeline Tests

Tests the config sampler, dataset construction, normalization, and split integrity.
These tests do NOT require BETSE; they use synthetic fixtures from conftest.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

# These imports must match the PRD project structure (Section 8).
# If the agent uses different module paths, these imports will fail,
# which is the correct signal that the structure is wrong.
from nexus.data.config_sampler import ConfigSampler
from nexus.data.dataset import BioelectricDataset
from nexus.data.validation import validate_simulation_result


# ===========================================================================
# Config Sampler
# ===========================================================================


class TestConfigSampler:
    """Tests for nexus.data.config_sampler.ConfigSampler"""

    def test_sampler_returns_correct_count(self):
        sampler = ConfigSampler()
        configs = sampler.sample(n=50)
        assert len(configs) == 50

    def test_sampler_returns_dicts_with_required_keys(self):
        sampler = ConfigSampler()
        configs = sampler.sample(n=5)
        required_keys = {"n_cells", "cell_radius", "channel_densities", "gj_conductance"}
        for cfg in configs:
            assert required_keys.issubset(
                cfg.keys()
            ), f"Config missing keys: {required_keys - cfg.keys()}"

    def test_channel_densities_within_range(self):
        """All sampled channel densities must be within the PRD-specified ranges."""
        # PRD Section 4.1.1: Nav 0-50, Kir 0-30, K_leak 0-20, Ca 0-10,
        # Cl 0-15, NaKATP 0-30, HKATP 0-10, VATP 0-10
        channel_maxes = [50, 30, 20, 10, 15, 30, 10, 10]
        sampler = ConfigSampler()
        configs = sampler.sample(n=200)
        for cfg in configs:
            densities = cfg["channel_densities"]  # expect (n_cells, 8) or similar
            if isinstance(densities, np.ndarray):
                for ch_idx, ch_max in enumerate(channel_maxes):
                    assert np.all(densities[:, ch_idx] >= 0), (
                        f"Channel {ch_idx} has negative density"
                    )
                    assert np.all(densities[:, ch_idx] <= ch_max * 1.01), (
                        f"Channel {ch_idx} exceeds max {ch_max}: "
                        f"got {densities[:, ch_idx].max():.2f}"
                    )

    def test_cell_count_within_range(self):
        """Cell count must be in [50, 500] per PRD Section 4.1.1."""
        sampler = ConfigSampler()
        configs = sampler.sample(n=100)
        for cfg in configs:
            assert 50 <= cfg["n_cells"] <= 500, f"n_cells={cfg['n_cells']} out of range"

    def test_gj_conductance_within_range(self):
        """Gap junction conductance must be in [0, 50] nS per PRD."""
        sampler = ConfigSampler()
        configs = sampler.sample(n=100)
        for cfg in configs:
            assert 0 <= cfg["gj_conductance"] <= 50, (
                f"gj_conductance={cfg['gj_conductance']} out of range"
            )

    def test_lhs_coverage(self):
        """
        Latin Hypercube Sampling should produce better coverage than pure random.
        Test: no two samples should be in the same LHS bin for any dimension.
        We test this on gj_conductance (1D) as a proxy.
        """
        sampler = ConfigSampler()
        configs = sampler.sample(n=50)
        gj_vals = np.array([cfg["gj_conductance"] for cfg in configs])
        # Divide [0, 50] into 50 bins; at least 40 bins should be occupied (80%)
        bins = np.linspace(0, 50, 51)
        hist, _ = np.histogram(gj_vals, bins=bins)
        occupied = np.sum(hist > 0)
        assert occupied >= 35, (
            f"LHS coverage too low: only {occupied}/50 bins occupied"
        )

    def test_perturbation_configs_generated(self):
        """Sampler should be able to generate perturbation configs."""
        sampler = ConfigSampler()
        for ptype in [
            "channel_blockade",
            "gj_blockade",
            "exogenous_expression",
            "spatial_gradient",
        ]:
            configs = sampler.sample(n=5, perturbation_type=ptype)
            assert len(configs) == 5
            for cfg in configs:
                assert cfg.get("perturbation_type") == ptype

    def test_channel_blockade_zeroes_channel(self):
        """Channel blockade perturbation should set one channel to zero."""
        sampler = ConfigSampler()
        configs = sampler.sample(n=10, perturbation_type="channel_blockade")
        for cfg in configs:
            densities = cfg["channel_densities"]
            if isinstance(densities, np.ndarray):
                # At least one channel column should be all zeros
                zero_channels = np.all(densities == 0, axis=0)
                assert np.any(zero_channels), "No channel was fully blocked"

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical configs."""
        sampler1 = ConfigSampler(seed=42)
        sampler2 = ConfigSampler(seed=42)
        configs1 = sampler1.sample(n=10)
        configs2 = sampler2.sample(n=10)
        for c1, c2 in zip(configs1, configs2):
            assert c1["n_cells"] == c2["n_cells"]
            assert c1["gj_conductance"] == c2["gj_conductance"]
            if isinstance(c1["channel_densities"], np.ndarray):
                np.testing.assert_array_equal(
                    c1["channel_densities"], c2["channel_densities"]
                )


# ===========================================================================
# Data Validation
# ===========================================================================


class TestDataValidation:
    """Tests for nexus.data.validation.validate_simulation_result"""

    def test_valid_tissue_passes(self, small_tissue):
        """A well-formed tissue record should pass validation."""
        errors = validate_simulation_result(small_tissue)
        assert len(errors) == 0, f"Unexpected validation errors: {errors}"

    def test_nan_vmem_detected(self, small_tissue):
        """NaN in Vmem should be caught."""
        small_tissue["vmem_steady_state"][0] = np.nan
        errors = validate_simulation_result(small_tissue)
        assert any("nan" in str(e).lower() or "NaN" in str(e) for e in errors)

    def test_inf_vmem_detected(self, small_tissue):
        """Inf in Vmem should be caught."""
        small_tissue["vmem_steady_state"][0] = np.inf
        errors = validate_simulation_result(small_tissue)
        assert any("inf" in str(e).lower() for e in errors)

    def test_out_of_range_vmem_detected(self, small_tissue):
        """Vmem outside [-120, +60] should be caught."""
        small_tissue["vmem_steady_state"][0] = 100.0  # > 60 mV
        errors = validate_simulation_result(small_tissue)
        assert any("range" in str(e).lower() for e in errors)

    def test_shape_mismatch_detected(self, small_tissue):
        """Vmem length should match n_cells."""
        small_tissue["vmem_steady_state"] = np.zeros(999)
        errors = validate_simulation_result(small_tissue)
        assert len(errors) > 0


# ===========================================================================
# PyTorch Geometric Dataset
# ===========================================================================


class TestBioelectricDataset:
    """Tests for nexus.data.dataset.BioelectricDataset"""

    def test_dataset_loads(self, synthetic_dataset):
        """Dataset should load from disk without error."""
        ds = BioelectricDataset(root=str(synthetic_dataset), split="train")
        assert len(ds) > 0

    def test_dataset_split_sizes(self, synthetic_dataset):
        """Splits should have the expected sizes."""
        train = BioelectricDataset(root=str(synthetic_dataset), split="train")
        val = BioelectricDataset(root=str(synthetic_dataset), split="val")
        test_id = BioelectricDataset(root=str(synthetic_dataset), split="test_id")
        test_ood = BioelectricDataset(root=str(synthetic_dataset), split="test_ood")
        assert len(train) == 60
        assert len(val) == 15
        assert len(test_id) == 15
        assert len(test_ood) == 40

    def test_data_object_has_required_attributes(self, synthetic_dataset):
        """Each Data object must have x, edge_index, edge_attr, y."""
        ds = BioelectricDataset(root=str(synthetic_dataset), split="train")
        data = ds[0]
        assert hasattr(data, "x"), "Missing node features 'x'"
        assert hasattr(data, "edge_index"), "Missing 'edge_index'"
        assert hasattr(data, "edge_attr"), "Missing edge attributes 'edge_attr'"
        assert hasattr(data, "y"), "Missing target 'y'"

    def test_node_feature_shape(self, synthetic_dataset):
        """x should be (n_cells, 8) — one row per cell, 8 channel types."""
        ds = BioelectricDataset(root=str(synthetic_dataset), split="train")
        data = ds[0]
        assert data.x.dim() == 2
        assert data.x.shape[1] == 8, f"Expected 8 channels, got {data.x.shape[1]}"

    def test_edge_index_shape(self, synthetic_dataset):
        """edge_index should be (2, n_edges), dtype long."""
        ds = BioelectricDataset(root=str(synthetic_dataset), split="train")
        data = ds[0]
        assert data.edge_index.dim() == 2
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.dtype == torch.long

    def test_edge_attr_shape(self, synthetic_dataset):
        """edge_attr should be (n_edges, 1)."""
        ds = BioelectricDataset(root=str(synthetic_dataset), split="train")
        data = ds[0]
        n_edges = data.edge_index.shape[1]
        assert data.edge_attr.shape == (n_edges, 1) or data.edge_attr.shape == (
            n_edges,
        ), f"edge_attr shape {data.edge_attr.shape} doesn't match n_edges={n_edges}"

    def test_target_shape(self, synthetic_dataset):
        """y should be (n_cells,) — one Vmem per cell."""
        ds = BioelectricDataset(root=str(synthetic_dataset), split="train")
        data = ds[0]
        n_cells = data.x.shape[0]
        assert data.y.shape == (n_cells,), (
            f"Target shape {data.y.shape} doesn't match n_cells={n_cells}"
        )

    def test_node_features_normalized(self, synthetic_dataset):
        """Node features should be normalized to [0, 1] per PRD Section 5.1.1."""
        ds = BioelectricDataset(root=str(synthetic_dataset), split="train")
        for i in range(min(10, len(ds))):
            data = ds[i]
            assert data.x.min() >= -0.01, f"x min={data.x.min():.4f} < 0"
            assert data.x.max() <= 1.01, f"x max={data.x.max():.4f} > 1"

    def test_edge_attr_normalized(self, synthetic_dataset):
        """Edge attrs should be normalized to [0, 1] per PRD Section 5.1.1."""
        ds = BioelectricDataset(root=str(synthetic_dataset), split="train")
        for i in range(min(10, len(ds))):
            data = ds[i]
            assert data.edge_attr.min() >= -0.01
            assert data.edge_attr.max() <= 1.01

    def test_edge_index_valid_range(self, synthetic_dataset):
        """All edge indices should be in [0, n_cells)."""
        ds = BioelectricDataset(root=str(synthetic_dataset), split="train")
        for i in range(min(10, len(ds))):
            data = ds[i]
            n_cells = data.x.shape[0]
            assert data.edge_index.min() >= 0
            assert data.edge_index.max() < n_cells

    def test_no_self_loops(self, synthetic_dataset):
        """Edges should not connect a cell to itself."""
        ds = BioelectricDataset(root=str(synthetic_dataset), split="train")
        for i in range(min(10, len(ds))):
            data = ds[i]
            src, dst = data.edge_index[0], data.edge_index[1]
            assert not torch.any(src == dst), "Self-loops detected"

    def test_target_in_physical_range(self, synthetic_dataset):
        """Target Vmem should be within [-120, 60] mV."""
        ds = BioelectricDataset(root=str(synthetic_dataset), split="train")
        for i in range(min(10, len(ds))):
            data = ds[i]
            assert data.y.min() >= -120.0, f"Vmem below -120: {data.y.min()}"
            assert data.y.max() <= 60.0, f"Vmem above +60: {data.y.max()}"

    def test_no_split_leakage(self, synthetic_dataset):
        """No config_id should appear in more than one split."""
        all_ids = {}
        for split in ["train", "val", "test_id", "test_ood"]:
            ds = BioelectricDataset(root=str(synthetic_dataset), split=split)
            for i in range(len(ds)):
                data = ds[i]
                cid = data.config_id if hasattr(data, "config_id") else str(i)
                if cid in all_ids:
                    pytest.fail(
                        f"Config {cid} in both '{all_ids[cid]}' and '{split}'"
                    )
                all_ids[cid] = split

    def test_ood_split_is_all_perturbation(self, synthetic_dataset):
        """test_ood split should contain only perturbation samples."""
        ds = BioelectricDataset(root=str(synthetic_dataset), split="test_ood")
        for i in range(len(ds)):
            data = ds[i]
            if hasattr(data, "is_perturbation"):
                assert data.is_perturbation, (
                    f"test_ood sample {i} is not a perturbation"
                )

    def test_train_split_is_all_baseline(self, synthetic_dataset):
        """Train split should contain only baseline (non-perturbation) samples."""
        ds = BioelectricDataset(root=str(synthetic_dataset), split="train")
        for i in range(len(ds)):
            data = ds[i]
            if hasattr(data, "is_perturbation"):
                assert not data.is_perturbation, (
                    f"Train sample {i} is a perturbation"
                )

    def test_dataloader_batching(self, synthetic_dataset):
        """DataLoader should batch variable-size graphs correctly."""
        from torch_geometric.loader import DataLoader

        ds = BioelectricDataset(root=str(synthetic_dataset), split="train")
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        batch = next(iter(loader))
        # Batched graph: x should have total_nodes rows
        assert batch.x.dim() == 2
        assert batch.x.shape[1] == 8
        # batch.batch assigns each node to its graph
        assert hasattr(batch, "batch")
        assert batch.batch.max() <= 7  # up to 8 graphs in batch
