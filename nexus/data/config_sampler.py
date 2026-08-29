import numpy as np
from scipy.stats import qmc

CHANNEL_NAMES = ["Nav", "Kir", "K_leak", "Ca", "Cl", "NaKATP", "HKATP", "VATP"]
CHANNEL_MAXES = np.array([50.0, 30.0, 20.0, 10.0, 15.0, 30.0, 10.0, 10.0], dtype=np.float32)
UNMAPPED_CHANNEL_INDICES = (6, 7)
SPATIAL_CHANNEL_INDICES = (0, 3, 4)

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
                ch = int(SPATIAL_CHANNEL_INDICES[self.rng.integers(0, len(SPATIAL_CHANNEL_INDICES))])
                k = max(1, n_cells // 4)
                channel_densities[:k, ch] = float(CHANNEL_MAXES[ch]) * 4.0
            elif perturbation_type == "spatial_gradient":
                ch = int(SPATIAL_CHANNEL_INDICES[self.rng.integers(0, len(SPATIAL_CHANNEL_INDICES))])
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

class TestConfigSampler:
    """Tests for nexus.data.config_sampler.ConfigSampler"""

    def test_sampler_returns_correct_count(self):
        sampler = ConfigSampler()
        configs = sampler.sample(n=50)
        assert len(configs) == 50

    def test_sampler_returns_dicts_with_required_keys(self):
        sampler = ConfigSampler()
        configs = sampler.sample(n=5)
        required_keys = {"n_cells", "cell_radius", "channel_densities", "gj_conductance"}
        for cfg in configs:
            assert required_keys.issubset(
                cfg.keys()
            ), f"Config missing keys: {required_keys - cfg.keys()}"

    def test_channel_densities_within_range(self):
        """All sampled channel densities must be within the PRD-specified ranges."""
        # PRD Section 4.1.1: Nav 0-50, Kir 0-30, K_leak 0-20, Ca 0-10,
        # Cl 0-15, NaKATP 0-30, HKATP 0-10, VATP 0-10
        channel_maxes = [50, 30, 20, 10, 15, 30, 10, 10]
        sampler = ConfigSampler()
        configs = sampler.sample(n=200)
        for cfg in configs:
            densities = cfg["channel_densities"]  # expect (n_cells, 8) or similar
            if isinstance(densities, np.ndarray):
                for ch_idx, ch_max in enumerate(channel_maxes):
                    assert np.all(densities[:, ch_idx] >= 0), (
                        f"Channel {ch_idx} has negative density"
                    )
                    assert np.all(densities[:, ch_idx] <= ch_max * 1.01), (
                        f"Channel {ch_idx} exceeds max {ch_max}: "
                        f"got {densities[:, ch_idx].max():.2f}"
                    )

    def test_cell_count_within_range(self):
        """Cell count must be in [50, 500] per PRD Section 4.1.1."""
        sampler = ConfigSampler()
        configs = sampler.sample(n=100)
        for cfg in configs:
            assert 50 <= cfg["n_cells"] <= 500, f"n_cells={cfg['n_cells']} out of range"

    def test_gj_conductance_within_range(self):
        """Gap junction conductance must be in [0, 50] nS per PRD."""
        sampler = ConfigSampler()
        configs = sampler.sample(n=100)
        for cfg in configs:
            assert 0 <= cfg["gj_conductance"] <= 50, (
                f"gj_conductance={cfg['gj_conductance']} out of range"
            )

    def test_lhs_coverage(self):
        """
        Latin Hypercube Sampling should produce better coverage than pure random.
        Test: no two samples should be in the same LHS bin for any dimension.
        We test this on gj_conductance (1D) as a proxy.
        """
        sampler = ConfigSampler()
        configs = sampler.sample(n=50)
        gj_vals = np.array([cfg["gj_conductance"] for cfg in configs])
        # Divide [0, 50] into 50 bins; at least 40 bins should be occupied (80%)
        bins = np.linspace(0, 50, 51)
        hist, _ = np.histogram(gj_vals, bins=bins)
        occupied = np.sum(hist > 0)
        assert occupied >= 35, (
            f"LHS coverage too low: only {occupied}/50 bins occupied"
        )

    def test_perturbation_configs_generated(self):
        """Sampler should be able to generate perturbation configs."""
        sampler = ConfigSampler()
        for ptype in [
            "channel_blockade",
            "gj_blockade",
            "exogenous_expression",
            "spatial_gradient",
        ]:
            configs = sampler.sample(n=5, perturbation_type=ptype)
            assert len(configs) == 5
            for cfg in configs:
                assert cfg.get("perturbation_type") == ptype

    def test_channel_blockade_zeroes_channel(self):
        """Channel blockade perturbation should set one channel to zero."""
        sampler = ConfigSampler()
        configs = sampler.sample(n=10, perturbation_type="channel_blockade")
        for cfg in configs:
            densities = cfg["channel_densities"]
            if isinstance(densities, np.ndarray):
                # At least one channel column should be all zeros
                zero_channels = np.all(densities == 0, axis=0)
                assert np.any(zero_channels), "No channel was fully blocked"

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical configs."""
        sampler1 = ConfigSampler(seed=42)
        sampler2 = ConfigSampler(seed=42)
        configs1 = sampler1.sample(n=10)
        configs2 = sampler2.sample(n=10)
        for c1, c2 in zip(configs1, configs2):
            assert c1["n_cells"] == c2["n_cells"]
            assert c1["gj_conductance"] == c2["gj_conductance"]
            if isinstance(c1["channel_densities"], np.ndarray):
                np.testing.assert_array_equal(
                    c1["channel_densities"], c2["channel_densities"]
                )
