"""
NEXUS Phase 1 — Training Loop Tests

Tests that the training loop works correctly: loss decreases, model can overfit
a tiny dataset, checkpoints save and restore, early stopping triggers, and
LR scheduling is active.
"""

import pytest
import numpy as np
import torch
import os
import tempfile
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from nexus.model.mpnn import MPNN
from nexus.model.baseline import BaselineMLP
from nexus.model.losses import mae_loss
from nexus.training.trainer import Trainer
from nexus.training.config import TrainingConfig

N_CHANNELS = 8


def _make_tiny_dataset(n_samples=20, n_cells=16, seed=42):
    """Create a tiny in-memory dataset for fast training tests."""
    rng = np.random.RandomState(seed)
    data_list = []

    for i in range(n_samples):
        x = torch.tensor(
            rng.uniform(0, 1, (n_cells, N_CHANNELS)), dtype=torch.float32
        )
        # Ring topology
        src = list(range(n_cells))
        dst = [(j + 1) % n_cells for j in range(n_cells)]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        edge_attr = torch.full((edge_index.shape[1], 1), 0.4, dtype=torch.float32)

        # Deterministic target: a known function of input features
        # so the model has something learnable
        y = (x[:, 0] * 40 - x[:, 1] * 80 + x[:, 2] * (-20) - 50).float()
        y = y.clamp(-120, 60)

        data_list.append(
            Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        )

    return data_list


# ===========================================================================
# Training Loop Tests
# ===========================================================================


class TestTrainer:
    """Tests for nexus.training.trainer.Trainer"""

    def test_loss_decreases(self):
        """
        Training for several epochs on a learnable dataset should decrease loss.
        This is the most basic sanity check for the training loop.
        """
        data = _make_tiny_dataset(n_samples=40, n_cells=16)
        train_data = data[:30]
        val_data = data[30:]
        train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=8)

        model = MPNN(n_channels=N_CHANNELS)
        config = TrainingConfig(
            lr=1e-3,
            max_epochs=30,
            patience=999,  # disable early stopping for this test
        )
        trainer = Trainer(model=model, config=config)

        history = trainer.train(train_loader, val_loader)

        assert len(history["train_loss"]) >= 10
        # Loss at end should be lower than loss at start
        first_5 = np.mean(history["train_loss"][:5])
        last_5 = np.mean(history["train_loss"][-5:])
        assert last_5 < first_5, (
            f"Loss did not decrease: first 5 epochs avg={first_5:.4f}, "
            f"last 5 epochs avg={last_5:.4f}"
        )

    def test_can_overfit_tiny_dataset(self):
        """
        A model should be able to memorize a very small dataset (N=10).
        If it can't, the architecture or training loop is broken.
        Final MAE should be < 5 mV on the training set.
        """
        data = _make_tiny_dataset(n_samples=10, n_cells=8)
        loader = DataLoader(data, batch_size=10, shuffle=True)

        model = MPNN(n_channels=N_CHANNELS)
        config = TrainingConfig(
            lr=1e-3,
            max_epochs=200,
            patience=999,
        )
        trainer = Trainer(model=model, config=config)
        history = trainer.train(loader, loader)

        final_loss = history["train_loss"][-1]
        assert final_loss < 5.0, (
            f"Could not overfit tiny dataset: final MAE={final_loss:.2f} mV "
            f"(expected < 5 mV)"
        )

    def test_checkpoint_save_and_load(self, tmp_path):
        """
        Saving and loading a checkpoint should produce identical predictions.
        """
        data = _make_tiny_dataset(n_samples=10, n_cells=8)
        loader = DataLoader(data, batch_size=10)

        model = MPNN(n_channels=N_CHANNELS)
        config = TrainingConfig(
            lr=1e-3,
            max_epochs=5,
            patience=999,
            checkpoint_dir=str(tmp_path),
        )
        trainer = Trainer(model=model, config=config)
        trainer.train(loader, loader)

        # Get predictions from trained model
        model.eval()
        batch = next(iter(loader))
        with torch.no_grad():
            pred_before = model(batch.x, batch.edge_index, batch.edge_attr).clone()

        # Save
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        # Load into a fresh model
        model2 = MPNN(n_channels=N_CHANNELS)
        trainer2 = Trainer(model=model2, config=config)
        trainer2.load_checkpoint(str(checkpoint_path))

        model2.eval()
        with torch.no_grad():
            pred_after = model2(batch.x, batch.edge_index, batch.edge_attr)

        torch.testing.assert_close(pred_before, pred_after)

    def test_early_stopping_triggers(self):
        """
        With patience=5, if validation loss doesn't improve for 5 epochs,
        training should stop before max_epochs.
        """
        data = _make_tiny_dataset(n_samples=20, n_cells=8)
        train_loader = DataLoader(data[:10], batch_size=10)
        # Validation set with different distribution — loss won't improve
        val_data = _make_tiny_dataset(n_samples=10, n_cells=8, seed=999)
        # Override targets to be very different
        for d in val_data:
            d.y = torch.full_like(d.y, 999.0)
        val_loader = DataLoader(val_data, batch_size=10)

        model = MPNN(n_channels=N_CHANNELS)
        config = TrainingConfig(
            lr=1e-5,  # very low LR so val loss won't improve
            max_epochs=200,
            patience=5,
        )
        trainer = Trainer(model=model, config=config)
        history = trainer.train(train_loader, val_loader)

        # Should have stopped well before 200 epochs
        assert len(history["train_loss"]) < 50, (
            f"Trained for {len(history['train_loss'])} epochs — "
            f"early stopping didn't trigger (patience=5)"
        )

    def test_lr_decreases_over_training(self):
        """LR scheduler should decrease the learning rate over training."""
        data = _make_tiny_dataset(n_samples=20, n_cells=8)
        loader = DataLoader(data, batch_size=10)

        model = MPNN(n_channels=N_CHANNELS)
        config = TrainingConfig(
            lr=1e-3,
            max_epochs=50,
            patience=999,
        )
        trainer = Trainer(model=model, config=config)
        history = trainer.train(loader, loader)

        if "lr" in history:
            assert history["lr"][-1] < history["lr"][0], (
                f"LR didn't decrease: start={history['lr'][0]}, end={history['lr'][-1]}"
            )

    def test_gradient_clipping_active(self):
        """
        With gradient clipping enabled, parameter updates should be bounded
        even with adversarial inputs that produce large gradients.
        """
        model = MPNN(n_channels=N_CHANNELS)
        config = TrainingConfig(
            lr=1.0,  # absurdly high LR
            max_epochs=1,
            patience=999,
            grad_clip_norm=1.0,
        )

        # Create data with extreme values
        x = torch.randn(8, N_CHANNELS) * 100
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_attr = torch.ones(4, 1)
        y = torch.randn(8) * 1000
        data = [Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)]
        loader = DataLoader(data, batch_size=1)

        trainer = Trainer(model=model, config=config)

        # Should not produce NaN parameters after one step with extreme data
        trainer.train(loader, loader)
        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), (
                f"Parameter {name} is non-finite after training with extreme data "
                f"— gradient clipping may not be active"
            )

    def test_history_contains_required_keys(self):
        """Training history should contain train_loss and val_loss."""
        data = _make_tiny_dataset(n_samples=20, n_cells=8)
        loader = DataLoader(data, batch_size=10)

        model = MPNN(n_channels=N_CHANNELS)
        config = TrainingConfig(lr=1e-3, max_epochs=5, patience=999)
        trainer = Trainer(model=model, config=config)
        history = trainer.train(loader, loader)

        assert "train_loss" in history, "History missing 'train_loss'"
        assert "val_loss" in history, "History missing 'val_loss'"
        assert len(history["train_loss"]) == 5
        assert len(history["val_loss"]) == 5

    def test_best_model_saved(self, tmp_path):
        """
        Trainer should save the best model (lowest val loss), not just the latest.
        """
        data = _make_tiny_dataset(n_samples=20, n_cells=8)
        loader = DataLoader(data, batch_size=10)

        model = MPNN(n_channels=N_CHANNELS)
        config = TrainingConfig(
            lr=1e-3,
            max_epochs=20,
            patience=999,
            checkpoint_dir=str(tmp_path),
        )
        trainer = Trainer(model=model, config=config)
        trainer.train(loader, loader)

        # Best checkpoint should exist
        best_path = tmp_path / "best.pt"
        assert best_path.exists() or any(
            "best" in f.name for f in tmp_path.iterdir()
        ), "No 'best' checkpoint found"

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical training trajectories."""
        results = []
        for _ in range(2):
            torch.manual_seed(42)
            np.random.seed(42)

            data = _make_tiny_dataset(n_samples=20, n_cells=8, seed=42)
            loader = DataLoader(data, batch_size=10, shuffle=False)

            model = MPNN(n_channels=N_CHANNELS)
            config = TrainingConfig(lr=1e-3, max_epochs=10, patience=999)
            trainer = Trainer(model=model, config=config)
            history = trainer.train(loader, loader)
            results.append(history["train_loss"])

        np.testing.assert_allclose(results[0], results[1], atol=1e-5)

    def test_baseline_mlp_trains(self):
        """The baseline MLP should also train through the same Trainer."""
        data = _make_tiny_dataset(n_samples=20, n_cells=8)
        loader = DataLoader(data, batch_size=10)

        model = BaselineMLP(n_channels=N_CHANNELS)
        config = TrainingConfig(lr=1e-3, max_epochs=20, patience=999)
        trainer = Trainer(model=model, config=config)
        history = trainer.train(loader, loader)

        first_5 = np.mean(history["train_loss"][:5])
        last_5 = np.mean(history["train_loss"][-5:])
        assert last_5 < first_5, "Baseline MLP loss did not decrease"
