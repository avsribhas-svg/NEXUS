import math
import numpy as np

CHANNEL_NAMES = ["Nav", "Kir", "K_leak", "Ca", "Cl", "NaKATP", "HKATP", "VATP"]
CHANNEL_MAXES = {"Nav": 50.0, "Kir": 30.0, "K_leak": 20.0, "Ca": 10.0,
                 "Cl": 15.0, "NaKATP": 30.0, "HKATP": 10.0, "VATP": 10.0}
WORLD_ALPHA = 2.28
DM_BASE = 1.0e-18
ALPHA_NAK_BASE = 1.0e-7
KIR_MAX_DM = 3.0e-16
KLEAK_MAX_DM = 2.0e-17
GJ_SURFACE_CLOSED = 1.0e-9
GJ_SURFACE_OPEN = 1.0e-7
GJ_CONDUCTANCE_MAX = 50.0
FRAC_MAX = 4.0

def world_size_from_n_cells(n_cells: int, cell_radius_um: float) -> float:
    return float(WORLD_ALPHA * cell_radius_um * math.sqrt(max(1, n_cells)) * 1e-6)

def normalize_channel_densities(channel_densities) -> dict:
    if isinstance(channel_densities, dict):
        return {name: float(channel_densities.get(name, 0.0)) for name in CHANNEL_NAMES}
    elif isinstance(channel_densities, np.ndarray) and channel_densities.ndim == 2:
        densities = channel_densities.mean(axis=0)
        return dict(zip(CHANNEL_NAMES, densities))
    elif isinstance(channel_densities, np.ndarray) and channel_densities.ndim == 1:
        return dict(zip(CHANNEL_NAMES, channel_densities))
    else:
        raise ValueError("Invalid input type for normalize_channel_densities")

def channel_densities_to_betse(densities: dict) -> dict:
    def frac(name):
        return min(FRAC_MAX, max(0.0, float(densities.get(name, 0.0)) / CHANNEL_MAXES[name]))
    
    return {
        "Dm_Na": DM_BASE * (1.0 + 3.0 * frac("Nav")),
        "Dm_K": DM_BASE,
        "Dm_Cl": DM_BASE * (0.1 + 1.9 * frac("Cl")),
        "Dm_Ca": DM_BASE * (0.1 + 1.9 * frac("Ca")),
        "alpha_NaK": ALPHA_NAK_BASE * (0.5 + 1.0 * frac("NaKATP")),
        "kir_max_dm": KIR_MAX_DM * frac("Kir"),
        "kleak_max_dm": KLEAK_MAX_DM * frac("K_leak"),
    }

def gj_conductance_to_surface_area(gj_conductance: float) -> float:
    frac_gj = min(1.0, max(0.0, float(gj_conductance) / GJ_CONDUCTANCE_MAX))
    return float(GJ_SURFACE_CLOSED + frac_gj * (GJ_SURFACE_OPEN - GJ_SURFACE_CLOSED))

def group_cells_by_density(channel_densities, n_groups: int = 8) -> list:
    arr = np.asarray(channel_densities, dtype=np.float64)
    n_cells = arr.shape[0]
    spread = arr.std(axis=0)
    
    if spread.max() <= 1e-12:
        return [(list(range(n_cells)), {name: float(arr[:, j].mean()) for j, name in enumerate(CHANNEL_NAMES)})]
    
    key_col = int(np.argmax(spread))
    order = np.argsort(arr[:, key_col], kind="stable")
    k = int(min(max(1, n_groups), n_cells))
    parts = np.array_split(order, k)
    
    groups = []
    for part in parts:
        if len(part) > 0:
            groups.append(([int(i) for i in part], {name: float(arr[part, j].mean()) for j, name in enumerate(CHANNEL_NAMES)}))
    
    return groups

def resample_densities_to_mesh(channel_densities, cell_centres) -> np.ndarray:
    centres = np.asarray(cell_centres, dtype=np.float64)
    n_actual = int(centres.shape[0])
    
    if isinstance(channel_densities, dict):
        row = np.array([float(channel_densities.get(name, 0.0)) for name in CHANNEL_NAMES], dtype=np.float64)
        return np.tile(row, (n_actual, 1)).astype(np.float32)
    else:
        src = np.asarray(channel_densities, dtype=np.float64)
        if src.ndim == 1:
            row = src
            return np.tile(row, (n_actual, 1)).astype(np.float32)
        elif src.ndim == 2:
            n_requested = int(src.shape[0])
            if n_requested == 0:
                return np.zeros((n_actual, 8), dtype=np.float32)
            
            order = np.lexsort((centres[:, 1], centres[:, 0]))
            ranks = np.arange(n_actual)
            src_rows = np.minimum((ranks * n_requested) // max(1, n_actual), n_requested - 1)
            out = np.zeros((n_actual, 8), dtype=np.float64)
            out[order] = src[src_rows]
            return out.astype(np.float32)
        else:
            raise ValueError("Invalid input type for channel_densities")
