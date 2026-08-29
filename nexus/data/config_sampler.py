import numpy as np
from scipy.stats import qmc

CHANNEL_NAMES = ["Nav", "Kir", "K_leak", "Ca", "Cl", "NaKATP", "HKATP", "VATP"]
CHANNEL_MAXES = np.array([50.0, 30.0, 20.0, 10.0, 15.0, 30.0, 10.0, 10.0], dtype=np.float32)
UNMAPPED_CHANNEL_INDICES = (6, 7)

class ConfigSampler:
    def __init__(self, seed=None, zero_unmapped_channels=True):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.zero_unmapped_channels = zero_unmapped_channels

    def sample(self, n: int, perturbation_type=None) -> list:
        sampler = qmc.LatinHypercube(d=11, seed=self.seed)
        u = sampler.random(n)
        
        configs = []
        for i in range(n):
            n_cells = int(round(float(np.exp(np.log(50.0) + u[i, 0] * (np.log(500.0) - np.log(50.0))))))
            n_cells = int(min(500, max(50, n_cells)))
            
            cell_radius = float(5.0 + u[i, 1] * 10.0)
            
            gj_conductance = float(u[i, 2] * 50.0)
            
            per_channel = (u[i, 3:11] * CHANNEL_MAXES).astype(np.float32)
            channel_densities = np.tile(per_channel, (n_cells, 1)).astype(np.float32)
            
            if self.zero_unmapped_channels:
                for ch_index in UNMAPPED_CHANNEL_INDICES:
                    channel_densities[:, ch_index] = 0.0
            
            n_perturbable = 6 if self.zero_unmapped_channels else 8
            
            if perturbation_type == "channel_blockade":
                ch = int(self.rng.integers(0, n_perturbable))
                channel_densities[:, ch] = 0.0
            elif perturbation_type == "gj_blockade":
                gj_conductance = 0.0
            elif perturbation_type == "exogenous_expression":
                ch = int(self.rng.integers(0, n_perturbable))
                k = max(1, n_cells // 4)
                channel_densities[:k, ch] = float(CHANNEL_MAXES[ch]) * 4.0
            elif perturbation_type == "spatial_gradient":
                ch = int(self.rng.integers(0, n_perturbable))
                channel_densities[:, ch] = (np.linspace(0.0, 1.0, n_cells) * float(CHANNEL_MAXES[ch])).astype(np.float32)
            
            config = {
                "config_id": f"cfg_{i:05d}",
                "n_cells": n_cells,
                "cell_radius": cell_radius,
                "channel_densities": channel_densities,
                "gj_conductance": gj_conductance,
                "perturbation_type": perturbation_type,
                "is_perturbation": perturbation_type is not None,
            }
            
            configs.append(config)
        
        return configs
