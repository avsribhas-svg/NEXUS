import numpy as np

def validate_simulation_result(record: dict) -> list:
    errors = []

    if "vmem_steady_state" not in record:
        errors.append("missing key: vmem_steady_state")
        return errors

    vmem = np.array(record["vmem_steady_state"])

    if np.any(np.isnan(vmem)):
        errors.append("vmem_steady_state contains NaN values")

    if np.any(np.isinf(vmem)):
        errors.append("vmem_steady_state contains Inf values")

    finite = np.isfinite(vmem)
    if np.any(finite & ((vmem < -120.0) | (vmem > 60.0))):
        errors.append("vmem_steady_state outside physical range [-120, 60] mV")

    if "tissue_geometry" in record and "n_cells" in record["tissue_geometry"]:
        n_cells = record["tissue_geometry"]["n_cells"]
        if vmem.shape[0] != n_cells:
            errors.append(f"shape mismatch: vmem_steady_state length {vmem.shape[0]} does not match n_cells {n_cells}")

    if "channel_densities" in record:
        channel_densities = np.array(record["channel_densities"])
        if channel_densities.ndim != 2 or channel_densities.shape[1] != 8:
            errors.append("channel_densities has wrong shape")
        elif "tissue_geometry" in record and "n_cells" in record["tissue_geometry"]:
            n_cells = record["tissue_geometry"]["n_cells"]
            if channel_densities.shape[0] != n_cells:
                errors.append(f"shape mismatch: channel_densities rows do not match vmem_steady_state length")

    if "gap_junctions" in record and "edge_index" in record["gap_junctions"]:
        edge_index = np.array(record["gap_junctions"]["edge_index"])
        if edge_index.size > 0:
            if np.any(edge_index < 0) or np.any(edge_index >= vmem.shape[0]):
                errors.append("edge_index refers to a node outside range")

    return errors
