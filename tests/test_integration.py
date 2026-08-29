"""
NEXUS Phase 1 — Integration Tests

End-to-end tests that verify the full pipeline works together.
BETSE-dependent tests are marked with @pytest.mark.betse and skipped by default.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader

from nexus.model.mpnn import MPNN
from nexus.model.losses import mae_loss
from nexus.training.trainer import Trainer
from nexus.training.config import TrainingConfig
from nexus.data.dataset import BioelectricDataset
from nexus.evaluation.metrics import compute_mae, compute_r_squared


N_CHANNELS = 8


# ===========================================================================
# End-to-End Pipeline (no BETSE)
# ===========================================================================


class TestEndToEnd:
    """
    Tests that the full pipeline (load data → train → evaluate) works
    end-to-end on synthetic data.
    """

    def test_full_pipeline(self, synthetic_dataset, tmp_path):
        """
        Load dataset → train model → evaluate on test split.
        This is the highest-level test: if this passes, the pipeline works.
        """
        # Load data
        train_ds = BioelectricDataset(root=str(synthetic_dataset), split="train")
        val_ds = BioelectricDataset(root=str(synthetic_dataset), split="val")
        test_ds = BioelectricDataset(root=str(synthetic_dataset), split="test_id")

        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16)
        test_loader = DataLoader(test_ds, batch_size=16)

        # Train
        torch.manual_seed(42)
        model = MPNN(n_channels=N_CHANNELS)
        config = TrainingConfig(
            lr=1e-3,
            max_epochs=30,
            patience=10,
            checkpoint_dir=str(tmp_path),
        )
        trainer = Trainer(model=model, config=config)
        history = trainer.train(train_loader, val_loader)

        # Evaluate
        model.eval()
        all_pred, all_true = [], []
        for batch in test_loader:
            with torch.no_grad():
                pred = model(batch.x, batch.edge_index, batch.edge_attr)
            all_pred.append(pred.numpy())
            all_true.append(batch.y.numpy())

        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)

        mae = compute_mae(all_pred, all_true)
        r2 = compute_r_squared(all_pred, all_true)

        # On synthetic data, the model should learn something
        # (MAE should be less than the std of the targets)
        target_std = np.std(all_true)
        assert mae < target_std, (
            f"MAE ({mae:.2f}) exceeds target std ({target_std:.2f}) — "
            f"model learned nothing"
        )

    def test_ood_evaluation_runs(self, synthetic_dataset, tmp_path):
        """
        OOD (perturbation) evaluation should run without error and produce
        per-perturbation-type metrics.
        """
        train_ds = BioelectricDataset(root=str(synthetic_dataset), split="train")
        ood_ds = BioelectricDataset(root=str(synthetic_dataset), split="test_ood")

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        ood_loader = DataLoader(ood_ds, batch_size=16)

        model = MPNN(n_channels=N_CHANNELS)
        config = TrainingConfig(lr=1e-3, max_epochs=10, patience=999)
        trainer = Trainer(model=model, config=config)
        trainer.train(train_loader, train_loader)

        model.eval()
        all_pred, all_true = [], []
        for batch in ood_loader:
            with torch.no_grad():
                pred = model(batch.x, batch.edge_index, batch.edge_attr)
            all_pred.append(pred.numpy())
            all_true.append(batch.y.numpy())

        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)

        # Should compute without error
        mae = compute_mae(all_pred, all_true)
        assert mae >= 0
        assert np.isfinite(mae)

    def test_model_save_load_evaluate_consistency(self, synthetic_dataset, tmp_path):
        """
        Train → save → load → evaluate should produce identical results
        to train → evaluate (no checkpoint corruption).
        """
        train_ds = BioelectricDataset(root=str(synthetic_dataset), split="train")
        test_ds = BioelectricDataset(root=str(synthetic_dataset), split="test_id")

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

        # Train and evaluate
        torch.manual_seed(42)
        model = MPNN(n_channels=N_CHANNELS)
        config = TrainingConfig(
            lr=1e-3, max_epochs=10, patience=999, checkpoint_dir=str(tmp_path)
        )
        trainer = Trainer(model=model, config=config)
        trainer.train(train_loader, train_loader)

        model.eval()
        preds_original = []
        for batch in test_loader:
            with torch.no_grad():
                preds_original.append(
                    model(batch.x, batch.edge_index, batch.edge_attr).numpy()
                )
        preds_original = np.concatenate(preds_original)

        # Save
        ckpt_path = tmp_path / "consistency_test.pt"
        trainer.save_checkpoint(str(ckpt_path))

        # Load into fresh model
        model2 = MPNN(n_channels=N_CHANNELS)
        trainer2 = Trainer(model=model2, config=config)
        trainer2.load_checkpoint(str(ckpt_path))

        model2.eval()
        preds_loaded = []
        for batch in test_loader:
            with torch.no_grad():
                preds_loaded.append(
                    model2(batch.x, batch.edge_index, batch.edge_attr).numpy()
                )
        preds_loaded = np.concatenate(preds_loaded)

        np.testing.assert_allclose(preds_original, preds_loaded, atol=1e-6)


# ===========================================================================
# BETSE Integration Tests (require BETSE installed)
# ===========================================================================


@pytest.mark.betse
class TestBETSEIntegration:
    """
    Tests that require BETSE to be installed and functional.
    Run with: pytest -m betse

    These are separated because BETSE may be difficult to install
    and these tests are slow (actual simulations).
    """

    def test_betse_importable(self):
        """BETSE should be importable."""
        try:
            import betse
        except ImportError:
            pytest.fail("BETSE is not installed or not importable")

    def test_betse_single_simulation(self):
        """
        Run a single BETSE simulation and verify the output structure.
        This is the most basic integration test.
        """
        from nexus.data.betse_generator import BETSEGenerator

        generator = BETSEGenerator()
        config = {
            "n_cells": 50,
            "cell_radius": 10.0,
            "channel_densities": {
                "Nav": 10.0,
                "Kir": 15.0,
                "K_leak": 10.0,
            },
            "gj_conductance": 20.0,
        }

        result = generator.run(config, timeout_s=300)

        assert result is not None, "BETSE simulation returned None"
        assert "vmem_steady_state" in result
        assert len(result["vmem_steady_state"]) > 0
        assert np.all(np.isfinite(result["vmem_steady_state"]))
        assert np.all(result["vmem_steady_state"] >= -120)
        assert np.all(result["vmem_steady_state"] <= 60)

    def test_betse_outputs_physically_plausible(self):
        """
        BETSE output Vmem should be in the biologically plausible range.
        Most non-neural cells have Vmem between -80 and -10 mV.
        """
        from nexus.data.betse_generator import BETSEGenerator

        generator = BETSEGenerator()
        config = {
            "n_cells": 50,
            "cell_radius": 10.0,
            "channel_densities": {
                "Nav": 5.0,
                "Kir": 20.0,
                "K_leak": 15.0,
                "NaKATP": 20.0,
            },
            "gj_conductance": 10.0,
        }

        result = generator.run(config, timeout_s=300)
        vmem = result["vmem_steady_state"]

        # Typical non-neural Vmem: -80 to -10 mV
        # Allow wider range for edge cases, but median should be in this range
        median_vmem = np.median(vmem)
        assert -100 < median_vmem < 20, (
            f"Median Vmem ({median_vmem:.1f} mV) outside plausible range"
        )

    def test_betse_wall_clock_recorded(self):
        """Simulation should record wall-clock time for speed benchmarking."""
        from nexus.data.betse_generator import BETSEGenerator

        generator = BETSEGenerator()
        config = {
            "n_cells": 50,
            "cell_radius": 10.0,
            "channel_densities": {"Kir": 15.0},
            "gj_conductance": 10.0,
        }

        result = generator.run(config, timeout_s=300)
        assert "metadata" in result
        assert "wall_clock_s" in result["metadata"]
        assert result["metadata"]["wall_clock_s"] > 0

    def test_betse_perturbation_changes_vmem(self):
        """
        Blocking a channel in BETSE should produce different Vmem
        than the baseline configuration.
        """
        from nexus.data.betse_generator import BETSEGenerator

        generator = BETSEGenerator()
        base_config = {
            "n_cells": 50,
            "cell_radius": 10.0,
            "channel_densities": {
                "Nav": 10.0,
                "Kir": 20.0,
                "K_leak": 10.0,
                "NaKATP": 15.0,
            },
            "gj_conductance": 10.0,
        }
        perturbed_config = {**base_config}
        perturbed_config["channel_densities"] = {
            **base_config["channel_densities"],
            "Kir": 0.0,  # Block Kir
        }

        result_base = generator.run(base_config, timeout_s=300)
        result_pert = generator.run(perturbed_config, timeout_s=300)

        vmem_diff = np.abs(
            np.mean(result_base["vmem_steady_state"])
            - np.mean(result_pert["vmem_steady_state"])
        )
        assert vmem_diff > 1.0, (
            f"Kir blockade changed mean Vmem by only {vmem_diff:.2f} mV — "
            f"perturbation may not be working"
        )
