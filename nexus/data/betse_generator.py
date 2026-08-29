import os
import glob
import time
import shutil
import tempfile
import subprocess
import numpy as np
from ruamel.yaml import YAML
from nexus.data.betse_config import (
    world_size_from_n_cells,
    normalize_channel_densities,
    channel_densities_to_betse,
    gj_conductance_to_surface_area,
)

class BETSEGenerator:

    def __init__(self, work_dir=None, keep_files=False):
        self.work_dir = work_dir
        self.keep_files = keep_files

    @staticmethod
    def _disable_events(node):
        if isinstance(node, dict):
            if "event happens" in node:
                node["event happens"] = False
            for value in node.values():
                BETSEGenerator._disable_events(value)
        elif isinstance(node, list):
            for element in node:
                BETSEGenerator._disable_events(element)

    def run(self, config: dict, timeout_s: float = 300.0, capture_timeseries: bool = False):
        work = None
        try:
            t_start = time.time()
            work = tempfile.mkdtemp(prefix="betse_", dir=self.work_dir)
            sim_dir = os.path.join(work, "sim")

            scaffold_arg = os.path.join(sim_dir, "config.yaml")
            log_path = os.path.join(work, "betse.log")
            proc = subprocess.run(["betse", "--log-file", log_path, "config", scaffold_arg], cwd=work,
                                  capture_output=True, text=True, timeout=timeout_s)
            if proc.returncode != 0:
                return None
            cfg_path = os.path.join(sim_dir, "config.yaml")

            yaml = YAML()
            with open(cfg_path) as f:
                y = yaml.load(f)

            densities = normalize_channel_densities(config["channel_densities"])
            betse_params = channel_densities_to_betse(densities)
            n_cells = int(config.get("n_cells", 100))
            cell_radius_um = float(config.get("cell_radius", 5.0))
            gj_conductance = float(config.get("gj_conductance", 20.0))
            world_size = world_size_from_n_cells(n_cells, cell_radius_um)

            y["general options"]["simulate extracellular spaces"] = False
            y["general options"]["ion profile"] = "mammal"
            y["world options"]["world size"] = float(world_size)
            y["world options"]["cell radius"] = float(cell_radius_um * 1e-6)
            y["world options"]["lattice disorder"] = 0.0
            dm = y["tissue profile definition"]["tissue"]["default"]["diffusion constants"]
            dm["Dm_Na"] = float(betse_params["Dm_Na"])
            dm["Dm_K"] = float(betse_params["Dm_K"])
            dm["Dm_Cl"] = float(betse_params["Dm_Cl"])
            dm["Dm_Ca"] = float(betse_params["Dm_Ca"])
            y["internal parameters"]["alpha_NaK"] = float(betse_params["alpha_NaK"])
            y["variable settings"]["gap junctions"]["gap junction surface area"] = float(
                gj_conductance_to_surface_area(gj_conductance))
            for ch in y["general network"]["channels"]:
                if ch["name"] == "Nav":
                    ch["max Dm"] = 0.0
                    ch["init active"] = True
                elif ch["name"] == "Kv":
                    ch["name"] = "Kir"
                    ch["channel type"] = "Kir2p1"
                    ch["max Dm"] = float(betse_params["kir_max_dm"])
                    ch["init active"] = True
                elif ch["name"] == "K_Leak":
                    ch["max Dm"] = float(betse_params["kleak_max_dm"])
                    ch["init active"] = True
            BETSEGenerator._disable_events(y)

            with open(cfg_path, "w") as f:
                yaml.dump(y, f)

            for stage in ("seed", "init", "sim"):
                remaining = timeout_s - (time.time() - t_start)
                if remaining <= 0:
                    return None
                proc = subprocess.run(["betse", "--log-file", log_path, stage, "config.yaml"], cwd=sim_dir,
                                      capture_output=True, text=True, timeout=remaining)
                if proc.returncode != 0:
                    return None

            sim_files = sorted(glob.glob(os.path.join(sim_dir, "SIMS", "*.betse.gz")))
            if not sim_files:
                return None
            from betse.lib.pickle import pickles
            loaded = pickles.load(sim_files[0])
            sim, cells = loaded[0], loaded[1]

            vmem = np.asarray(sim.vm_ave, dtype=np.float64) * 1000.0
            positions = np.asarray(cells.cell_centres, dtype=np.float64) * 1e6
            nn = np.asarray(cells.cell_nn_i, dtype=np.int64)
            keep = nn[:, 0] != nn[:, 1]
            edge_index = nn[keep].T.astype(np.int64)
            gjopen = np.asarray(sim.gjopen, dtype=np.float64)[keep]
            conductances = (gjopen * gj_conductance).astype(np.float64)
            n_cells_actual = int(vmem.shape[0])

            vm_series = getattr(sim, "vm_ave_time", None)
            converged_fraction = 1.0
            max_dvmem_mv = 0.0
            if vm_series is not None and len(vm_series) >= 2:
                last = np.asarray(vm_series[-1], dtype=np.float64) * 1000.0
                prev = np.asarray(vm_series[-2], dtype=np.float64) * 1000.0
                if last.shape == prev.shape and last.size > 0:
                    dv = np.abs(last - prev)
                    max_dvmem_mv = float(dv.max())
                    converged_fraction = float(np.mean(dv < 0.1))

            timeseries = None
            if capture_timeseries and vm_series is not None and len(vm_series) > 0:
                timeseries = (np.asarray(vm_series, dtype=np.float64) * 1000.0).astype(np.float32)

            density_row = np.array([densities[k] for k in
                ["Nav", "Kir", "K_leak", "Ca", "Cl", "NaKATP", "HKATP", "VATP"]], dtype=np.float32)
            channel_densities = np.tile(density_row, (n_cells_actual, 1)).astype(np.float32)

            import betse
            return {
                "config_id": str(config.get("config_id", "betse_run")),
                "tissue_geometry": {
                    "n_cells": n_cells_actual,
                    "cell_radius": cell_radius_um,
                    "cell_positions": positions.astype(np.float32),
                },
                "channel_densities": channel_densities,
                "gap_junctions": {
                    "edge_index": edge_index,
                    "conductances": conductances.astype(np.float32),
                },
                "vmem_steady_state": vmem.astype(np.float32),
                "vmem_timeseries": timeseries,
                "is_perturbation": bool(config.get("is_perturbation", False)),
                "perturbation_type": config.get("perturbation_type", None),
                "metadata": {
                    "betse_version": str(getattr(betse, "__version__", "unknown")),
                    "sim_duration_s": float(config.get("sim_duration_s", 0.0)),
                    "wall_clock_s": float(time.time() - t_start),
                    "converged_fraction": converged_fraction,
                    "max_dvmem_mv": max_dvmem_mv,
                },
            }

        except Exception:
            return None

        finally:
            if work is not None and not self.keep_files:
                shutil.rmtree(work, ignore_errors=True)
