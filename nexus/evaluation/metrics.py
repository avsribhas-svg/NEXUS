import numpy as np

def compute_mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))

def compute_r_squared(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)

def compute_per_group_mae(pred: np.ndarray, true: np.ndarray, groups: np.ndarray) -> dict:
    result = {}
    unique_groups = np.unique(groups)
    for group in unique_groups:
        mask = groups == group
        result[group] = float(np.mean(np.abs(pred[mask] - true[mask])))
    return result

def vmem_accuracy_threshold(vmem_min: float, vmem_max: float) -> float:
    return float(0.10 * (vmem_max - vmem_min))
