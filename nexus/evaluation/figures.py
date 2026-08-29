import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_scatter_plot(pred: np.ndarray, true: np.ndarray, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true, pred, s=8, alpha=0.5)
    lo = min(np.min(pred), np.min(true))
    hi = max(np.max(pred), np.max(true))
    ax.plot([lo, hi], [lo, hi], 'k--')
    ax.set_xlabel("True Vmem (mV)")
    ax.set_ylabel("Predicted Vmem (mV)")
    ax.set_title("Predicted vs. True Vmem")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def generate_spatial_error_map(positions: np.ndarray, pred_vmem: np.ndarray, true_vmem: np.ndarray, output_path: str) -> None:
    errors = np.abs(pred_vmem - true_vmem)
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c=errors, cmap="viridis", s=60)
    fig.colorbar(scatter, label="|error| (mV)")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("Spatial Prediction Error")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
