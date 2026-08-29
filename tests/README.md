# NEXUS Phase 1 — Test Suite

## Purpose

This test suite is the acceptance criterion for the NEXUS Phase 1 build. When all tests pass, the code is correct. Use it as the eval in an agent loop: generate code → run tests → iterate on failures → repeat until green.

## Running

```bash
# All tests except BETSE-dependent ones (fast, ~2 min)
pytest

# Specific test file
pytest tests/test_model.py

# Specific test class
pytest tests/test_model.py::TestMPNN

# Specific test
pytest tests/test_model.py::TestMPNN::test_forward_pass_shape_single

# Include BETSE integration tests (slow, requires BETSE installed)
pytest -m betse

# Everything including BETSE
pytest -m ""

# Verbose with full tracebacks
pytest -v --tb=long
```

## Test Structure

| File | What it tests | Depends on |
|---|---|---|
| `conftest.py` | Shared fixtures, synthetic data generators | numpy, scipy |
| `test_data_pipeline.py` | Config sampling, dataset loading, splits, normalization | `nexus.data.*` |
| `test_model.py` | MPNN + MLP architecture, shapes, gradients, equivariance, edge usage | `nexus.model.*` |
| `test_training.py` | Training loop, loss decrease, overfitting, checkpointing, early stopping | `nexus.training.*` |
| `test_evaluation.py` | Metrics, generalization grouping, figure generation | `nexus.evaluation.*` |
| `test_numerical.py` | Physics invariants, symmetry, topology sensitivity, baseline comparison | `nexus.model.*`, `nexus.training.*` |
| `test_integration.py` | End-to-end pipeline, BETSE integration | All modules |

## Import Contract

Tests import from the module structure defined in the PRD (Section 8):

```
nexus.data.config_sampler.ConfigSampler
nexus.data.dataset.BioelectricDataset
nexus.data.validation.validate_simulation_result
nexus.data.betse_generator.BETSEGenerator        # BETSE tests only
nexus.model.mpnn.MPNN
nexus.model.baseline.BaselineMLP
nexus.model.losses.mae_loss
nexus.model.losses.physics_auxiliary_loss
nexus.training.trainer.Trainer
nexus.training.config.TrainingConfig
nexus.evaluation.metrics.compute_mae
nexus.evaluation.metrics.compute_r_squared
nexus.evaluation.metrics.compute_per_group_mae
nexus.evaluation.metrics.vmem_accuracy_threshold
nexus.evaluation.generalization.group_by_perturbation_type
nexus.evaluation.figures.generate_scatter_plot
nexus.evaluation.figures.generate_spatial_error_map
```

If an import fails, the module doesn't exist yet or is named differently. The import path IS the specification.

## Expected Signatures

### `ConfigSampler(seed=None)`
- `.sample(n: int, perturbation_type: str | None = None) -> list[dict]`
- Each dict has at minimum: `n_cells`, `cell_radius`, `channel_densities`, `gj_conductance`
- `channel_densities` is `np.ndarray` of shape `(n_cells, 8)` or a dict mapping channel names to per-cell arrays

### `BioelectricDataset(root: str, split: str)`
- Subclass of `torch_geometric.data.InMemoryDataset`
- `split` is one of: `"train"`, `"val"`, `"test_id"`, `"test_ood"`
- Each item is a `torch_geometric.data.Data` with: `x`, `edge_index`, `edge_attr`, `y`

### `validate_simulation_result(record: dict) -> list[str]`
- Returns a list of error strings. Empty list = valid.

### `MPNN(n_channels: int, n_layers: int = 6)`
- Forward: `model(x, edge_index, edge_attr) -> Tensor` of shape `(n_nodes,)`
- `x`: `(n_nodes, n_channels)`, `edge_index`: `(2, n_edges)`, `edge_attr`: `(n_edges, 1)`

### `BaselineMLP(n_channels: int)`
- Forward: `model(x) -> Tensor` of shape `(n_nodes,)`

### `mae_loss(pred, target) -> Tensor`
### `physics_auxiliary_loss(pred, edge_index, conductances) -> Tensor`

### `TrainingConfig(lr, max_epochs, patience, checkpoint_dir=None, grad_clip_norm=1.0)`

### `Trainer(model, config)`
- `.train(train_loader, val_loader) -> dict` with keys `"train_loss"`, `"val_loss"`, optionally `"lr"`
- `.save_checkpoint(path: str)`
- `.load_checkpoint(path: str)`

### `compute_mae(pred: np.ndarray, true: np.ndarray) -> float`
### `compute_r_squared(pred: np.ndarray, true: np.ndarray) -> float`
### `compute_per_group_mae(pred, true, groups) -> dict[str, float]`
### `vmem_accuracy_threshold(vmem_min: float, vmem_max: float) -> float`
### `group_by_perturbation_type(records: list[dict]) -> dict[str, list]`
### `generate_scatter_plot(pred, true, output_path: str)`
### `generate_spatial_error_map(positions, pred_vmem, true_vmem, output_path: str)`

## What "All Tests Pass" Means

1. **Data pipeline works**: configs are valid, datasets load, splits are clean, normalization is correct.
2. **Model architecture is correct**: shapes, gradients, equivariance, edge sensitivity, parameter count.
3. **Training loop works**: loss decreases, overfitting succeeds, checkpoints round-trip, early stopping fires, LR schedules.
4. **Metrics are correct**: MAE, R², per-group metrics compute correctly on known inputs.
5. **Physics invariants hold**: symmetric inputs → symmetric outputs, topology affects predictions, GNN beats MLP on coupled data.
6. **Pipeline is end-to-end**: data → train → evaluate runs without error and produces meaningful results.
