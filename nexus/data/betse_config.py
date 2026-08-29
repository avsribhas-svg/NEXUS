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
        return min(1.0, max(0.0, float(densities.get(name, 0.0)) / CHANNEL_MAXES[name]))
    
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
