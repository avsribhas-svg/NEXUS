import os
import numpy as np

def save_record(record: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {
        "config_id": np.array(str(record["config_id"])),
        "n_cells": np.array(int(record["tissue_geometry"]["n_cells"])),
        "cell_radius": np.array(float(record["tissue_geometry"]["cell_radius"])),
        "cell_positions": np.asarray(record["tissue_geometry"]["cell_positions"], dtype=np.float32),
        "channel_densities": np.asarray(record["channel_densities"], dtype=np.float32),
        "edge_index": np.asarray(record["gap_junctions"]["edge_index"], dtype=np.int64),
        "conductances": np.asarray(record["gap_junctions"]["conductances"], dtype=np.float32),
        "vmem_steady_state": np.asarray(record["vmem_steady_state"], dtype=np.float32),
        "is_perturbation": np.array(bool(record["is_perturbation"])),
        "perturbation_type": np.array("" if record.get("perturbation_type") is None else str(record["perturbation_type"])),
        "betse_version": np.array(str(record.get("metadata", {}).get("betse_version", "unknown"))),
        "wall_clock_s": np.array(float(record.get("metadata", {}).get("wall_clock_s", 0.0))),
        "converged_fraction": np.array(float(record.get("metadata", {}).get("converged_fraction", 1.0))),
        "max_dvmem_mv": np.array(float(record.get("metadata", {}).get("max_dvmem_mv", 0.0)))
    }
    if record.get("vmem_timeseries") is not None:
        arrays["vmem_timeseries"] = np.asarray(record["vmem_timeseries"], dtype=np.float32)
    np.savez_compressed(path, **arrays)
    return path

def load_record(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    
    def scalar(key, cast, default=None):
        if key not in z.files:
            return default
        return cast(z[key].item()) if z[key].ndim == 0 else cast(z[key])
    
    pt = scalar("perturbation_type", str, "")
    pt = None if pt == "" else pt
    
    return {
        "config_id": scalar("config_id", str, ""),
        "tissue_geometry": {
            "n_cells": scalar("n_cells", int, 0),
            "cell_radius": scalar("cell_radius", float, 0.0),
            "cell_positions": z["cell_positions"],
        },
        "channel_densities": z["channel_densities"],
        "gap_junctions": {
            "edge_index": z["edge_index"],
            "conductances": z["conductances"],
        },
        "vmem_steady_state": z["vmem_steady_state"],
        "vmem_timeseries": z["vmem_timeseries"] if "vmem_timeseries" in z.files else None,
        "is_perturbation": scalar("is_perturbation", bool, False),
        "perturbation_type": pt,
        "metadata": {
            "betse_version": scalar("betse_version", str, "unknown"),
            "wall_clock_s": scalar("wall_clock_s", float, 0.0),
            "converged_fraction": scalar("converged_fraction", float, 1.0),
            "max_dvmem_mv": scalar("max_dvmem_mv", float, 0.0),
        },
    }
