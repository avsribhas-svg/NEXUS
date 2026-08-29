"""
NEXUS Phase 1 — Test Fixtures

Shared fixtures for the entire test suite. These create minimal synthetic data
that mimics the structure of real BETSE outputs without requiring BETSE itself.
Most tests run against these fixtures, making the test loop fast.

BETSE-dependent tests are marked with @pytest.mark.betse and skipped by default.
Run them explicitly with: pytest -m betse
"""

import pytest
import numpy as np
import torch
import os
import json
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants matching the PRD specification
# ---------------------------------------------------------------------------

N_CHANNELS = 8  # Nav, Kir, K_leak, Ca, Cl, NaKATP, HKATP, VATP
CHANNEL_NAMES = ["Nav", "Kir", "K_leak", "Ca", "Cl", "NaKATP", "HKATP", "VATP"]
VMEM_MIN = -120.0  # mV, physical lower bound
VMEM_MAX = 60.0     # mV, physical upper bound
VMEM_RANGE = VMEM_MAX - VMEM_MIN  # 180 mV


def _make_hexagonal_grid(n_cells: int, cell_radius: float = 10.0):
    """Generate a hexagonal grid of cell positions and nearest-neighbor edges."""
    # Approximate a hex grid by laying out cells in rows
    side = int(np.ceil(np.sqrt(n_cells)))
    positions = []
    for row in range(side):
        for col in range(side):
            if len(positions) >= n_cells:
                break
            x = col * cell_radius * 2 + (row % 2) * cell_radius
            y = row * cell_radius * np.sqrt(3)
            positions.append([x, y])
        if len(positions) >= n_cells:
            break
    positions = np.array(positions[:n_cells], dtype=np.float32)

    # Nearest-neighbor edges (distance < 2.5 * cell_radius)
    from scipy.spatial import distance_matrix
    dist = distance_matrix(positions, positions)
    threshold = 2.5 * cell_radius
    src, dst = np.where((dist < threshold) & (dist > 0))
    edge_index = np.stack([src, dst], axis=0)

    return positions, edge_index


def _make_synthetic_tissue(
    n_cells: int = 64,
    cell_radius: float = 10.0,
    gj_conductance: float = 20.0,
    seed: int = 42,
    perturbation_type: str | None = None,
):
    """
    Create one synthetic tissue record matching the PRD schema (Section 4.1.2).
    Vmem is computed via a simplified physical model (not BETSE) so tests have
    known-correct ground truth without needing the real simulator.
    """
    rng = np.random.RandomState(seed)

    positions, edge_index = _make_hexagonal_grid(n_cells, cell_radius)
    n_edges = edge_index.shape[1]

    # Channel densities: random within PRD ranges
    channel_maxes = np.array([50, 30, 20, 10, 15, 30, 10, 10], dtype=np.float32)
    channel_densities = rng.uniform(0, 1, size=(n_cells, N_CHANNELS)).astype(np.float32)
    channel_densities *= channel_maxes[None, :]

    # Apply perturbation if specified
    if perturbation_type == "channel_blockade":
        channel_densities[:, 0] = 0.0  # Block Nav
    elif perturbation_type == "gj_blockade":
        gj_conductance = 0.0
    elif perturbation_type == "exogenous_expression":
        channel_densities[:n_cells // 4, 0] = 200.0  # 4x training max for Nav subset
    elif perturbation_type == "spatial_gradient":
        gradient = np.linspace(0, 1, n_cells).astype(np.float32)
        channel_densities[:, 0] = gradient * 50.0  # Nav gradient

    conductances = np.full(n_edges, gj_conductance, dtype=np.float32)
    if perturbation_type == "gj_blockade":
        conductances[:] = 0.0

    # Simplified Vmem ground truth: weighted sum of channel contributions + GJ coupling
    # This is NOT physically accurate — it's a deterministic function the model should learn
    # Nav depolarizes, K channels hyperpolarize, pumps set resting potential
    channel_contributions = np.array(
        [+40, -80, -70, +30, -60, -70, -40, -30], dtype=np.float32
    )  # mV contribution at max density
    vmem_intrinsic = (channel_densities / channel_maxes[None, :]) @ (
        channel_contributions * channel_maxes / channel_maxes.max()
    )

    # GJ coupling: simple averaging with neighbors
    if gj_conductance > 0 and n_edges > 0:
        # Build adjacency for averaging
        adj = np.zeros((n_cells, n_cells), dtype=np.float32)
        for e in range(n_edges):
            i, j = edge_index[0, e], edge_index[1, e]
            adj[i, j] = conductances[e]
        row_sum = adj.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        adj_norm = adj / row_sum
        coupling_strength = min(gj_conductance / 50.0, 0.5)
        vmem = (1 - coupling_strength) * vmem_intrinsic + coupling_strength * (
            adj_norm @ vmem_intrinsic
        )
    else:
        vmem = vmem_intrinsic

    # Clamp to physical range
    vmem = np.clip(vmem, VMEM_MIN, VMEM_MAX)

    return {
        "config_id": f"synth_{seed}_{perturbation_type or 'baseline'}",
        "tissue_geometry": {
            "n_cells": n_cells,
            "cell_radius": cell_radius,
            "cell_positions": positions,
        },
        "channel_densities": channel_densities,
        "gap_junctions": {
            "edge_index": edge_index,
            "conductances": conductances,
        },
        "vmem_steady_state": vmem,
        "vmem_timeseries": None,
        "is_perturbation": perturbation_type is not None,
        "perturbation_type": perturbation_type,
        "metadata": {
            "betse_version": "synthetic",
            "sim_duration_s": 30.0,
            "wall_clock_s": 0.01,
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_tissue():
    """A single small tissue (16 cells) for quick tests."""
    return _make_synthetic_tissue(n_cells=16, seed=42)


@pytest.fixture
def medium_tissue():
    """A medium tissue (64 cells) for shape/scaling tests."""
    return _make_synthetic_tissue(n_cells=64, seed=99)


@pytest.fixture
def large_tissue():
    """A larger tissue (256 cells) for performance/scaling tests."""
    return _make_synthetic_tissue(n_cells=256, seed=137)


@pytest.fixture
def uniform_tissue():
    """All cells have identical channel densities. Model should predict identical Vmem."""
    tissue = _make_synthetic_tissue(n_cells=25, seed=0, gj_conductance=50.0)
    # Override: make all cells identical
    tissue["channel_densities"][:] = tissue["channel_densities"][0]
    # Recompute Vmem: with identical cells and strong GJ coupling, all Vmem should be equal
    channel_maxes = np.array([50, 30, 20, 10, 15, 30, 10, 10], dtype=np.float32)
    channel_contributions = np.array(
        [+40, -80, -70, +30, -60, -70, -40, -30], dtype=np.float32
    )
    vmem_val = (tissue["channel_densities"][0] / channel_maxes) @ (
        channel_contributions * channel_maxes / channel_maxes.max()
    )
    vmem_val = np.clip(vmem_val, VMEM_MIN, VMEM_MAX)
    tissue["vmem_steady_state"][:] = vmem_val
    return tissue


@pytest.fixture
def isolated_cells_tissue():
    """Zero gap junction conductance. Cells are electrically independent."""
    return _make_synthetic_tissue(n_cells=36, seed=55, gj_conductance=0.0)


@pytest.fixture
def perturbation_tissues():
    """One tissue per perturbation type."""
    return {
        ptype: _make_synthetic_tissue(n_cells=36, seed=200 + i, perturbation_type=ptype)
        for i, ptype in enumerate(
            ["channel_blockade", "gj_blockade", "exogenous_expression", "spatial_gradient"]
        )
    }


@pytest.fixture
def synthetic_dataset(tmp_path):
    """
    A small synthetic dataset (100 baseline + 40 perturbation) persisted to disk.
    Returns the directory path.
    """
    data_dir = tmp_path / "synthetic"
    for split in ["train", "val", "test_id", "test_ood"]:
        (data_dir / split).mkdir(parents=True)

    manifest = []

    # 60 train, 15 val, 15 test_id (baseline)
    for i, (split, count) in enumerate(
        [("train", 60), ("val", 15), ("test_id", 15)]
    ):
        for j in range(count):
            seed = i * 1000 + j
            tissue = _make_synthetic_tissue(n_cells=25, seed=seed)
            path = data_dir / split / f"{tissue['config_id']}.npz"
            np.savez(
                path,
                config_id=tissue["config_id"],
                n_cells=tissue["tissue_geometry"]["n_cells"],
                cell_positions=tissue["tissue_geometry"]["cell_positions"],
                channel_densities=tissue["channel_densities"],
                edge_index=tissue["gap_junctions"]["edge_index"],
                conductances=tissue["gap_junctions"]["conductances"],
                vmem_steady_state=tissue["vmem_steady_state"],
                is_perturbation=tissue["is_perturbation"],
            )
            manifest.append(
                {
                    "config_id": tissue["config_id"],
                    "split": split,
                    "path": str(path.relative_to(data_dir)),
                    "n_cells": tissue["tissue_geometry"]["n_cells"],
                    "is_perturbation": tissue["is_perturbation"],
                    "perturbation_type": tissue["perturbation_type"],
                }
            )

    # 40 test_ood (10 per perturbation type)
    for pi, ptype in enumerate(
        ["channel_blockade", "gj_blockade", "exogenous_expression", "spatial_gradient"]
    ):
        for j in range(10):
            seed = 5000 + pi * 100 + j
            tissue = _make_synthetic_tissue(
                n_cells=25, seed=seed, perturbation_type=ptype
            )
            path = data_dir / "test_ood" / f"{tissue['config_id']}.npz"
            np.savez(
                path,
                config_id=tissue["config_id"],
                n_cells=tissue["tissue_geometry"]["n_cells"],
                cell_positions=tissue["tissue_geometry"]["cell_positions"],
                channel_densities=tissue["channel_densities"],
                edge_index=tissue["gap_junctions"]["edge_index"],
                conductances=tissue["gap_junctions"]["conductances"],
                vmem_steady_state=tissue["vmem_steady_state"],
                is_perturbation=tissue["is_perturbation"],
            )
            manifest.append(
                {
                    "config_id": tissue["config_id"],
                    "split": "test_ood",
                    "path": str(path.relative_to(data_dir)),
                    "n_cells": tissue["tissue_geometry"]["n_cells"],
                    "is_perturbation": tissue["is_perturbation"],
                    "perturbation_type": tissue["perturbation_type"],
                }
            )

    # Write manifest
    import csv
    with open(data_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)

    return data_dir


@pytest.fixture
def device():
    """Return the available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
