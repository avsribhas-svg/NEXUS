import os
import sys
import json
import time
import argparse
import tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.data import Data
from nexus.data.config_sampler import ConfigSampler
from nexus.data.betse_generator import BETSEGenerator
from nexus.model.mpnn import MPNN
from nexus.model.baseline import BaselineMLP

CHANNEL_MAXES = np.array([50.0, 30.0, 20.0, 10.0, 15.0, 30.0, 10.0, 10.0], dtype=np.float32)
GJ_MAX = 50.0

def record_to_data(rec) -> Data:
    x = torch.tensor(np.asarray(rec["channel_densities"]) / CHANNEL_MAXES, dtype=torch.float32)
    edge_index = torch.tensor(np.asarray(rec["gap_junctions"]["edge_index"]), dtype=torch.long)
    edge_attr = torch.tensor(np.asarray(rec["gap_junctions"]["conductances"]), dtype=torch.float32).reshape(-1, 1) / GJ_MAX
    y = torch.tensor(np.asarray(rec["vmem_steady_state"]), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def stats(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "median": 0.0, "q1": 0.0, "q3": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    else:
        return {
            "n": int(a.size),
            "median": float(np.median(a)),
            "q1": float(np.percentile(a, 25)),
            "q3": float(np.percentile(a, 75)),
            "mean": float(a.mean()),
            "min": float(a.min()),
            "max": float(a.max())
        }

def main() -> int:
    parser = argparse.ArgumentParser(description="Speed benchmark for BETSE and model inference.")
    parser.add_argument("--n", type=int, default=100, help="Number of simulations to run")
    parser.add_argument("--seed", type=int, default=909, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--arch", type=str, default="mpnn", choices=["mpnn", "mlp"], help="Model architecture")
    parser.add_argument("--timeout", type=float, default=600.0, help="Timeout for each simulation in seconds")
    parser.add_argument("--out", type=str, default="outputs/speed_benchmark.json", help="Output file path")
    args = parser.parse_args()

    sampler = ConfigSampler(seed=args.seed)
    configs = sampler.sample(args.n)

    work = tempfile.mkdtemp(prefix="speedbench_")
    gen = BETSEGenerator(work_dir=work)
    betse_times, records = [], []
    for i, cfg in enumerate(configs):
        t0 = time.time()
        rec = gen.run(cfg, timeout_s=args.timeout)
        dt = time.time() - t0
        if rec is None:
            print("  [%3d/%d] FAILED after %.1f s" % (i + 1, len(configs), dt), flush=True)
            continue
        betse_times.append(dt)
        records.append(rec)
        print("  [%3d/%d] %d cells, %.1f s" % (i + 1, len(configs), len(rec["vmem_steady_state"]), dt), flush=True)

    model = MPNN(n_channels=8) if args.arch == "mpnn" else BaselineMLP(n_channels=8)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    needs_graph = args.arch == "mpnn"

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    model_times = {dev: [] for dev in devices}
    for dev in devices:
        m = model.to(torch.device(dev))
        for _ in range(3):
            d = record_to_data(records[0]).to(torch.device(dev))
            with torch.no_grad():
                _ = m(d.x, d.edge_index, d.edge_attr) if needs_graph else m(d.x)
        for rec in records:
            t0 = time.time()
            d = record_to_data(rec).to(torch.device(dev))
            with torch.no_grad():
                _ = m(d.x, d.edge_index, d.edge_attr) if needs_graph else m(d.x)
            if dev == "cuda":
                torch.cuda.synchronize()
            model_times[dev].append(time.time() - t0)

    report = {
        "n_requested": args.n,
        "n_completed": len(records),
        "arch": args.arch,
        "checkpoint": args.checkpoint,
        "cells": stats([len(r["vmem_steady_state"]) for r in records]),
        "betse_seconds": stats(betse_times),
        "model_seconds": {dev: stats(times) for dev, times in model_times.items()},
        "speedup": {dev: float(np.median(betse_times) / max(1e-9, np.median(times))) for dev, times in model_times.items()}
    }

    print("Cell-count median:", report["cells"]["median"])
    print("BETSE median (q1, q3):", report["betse_seconds"]["q1"], report["betse_seconds"]["q3"])
    for s in report["model_seconds"]:
        print(f"{s} model median (q1, q3): {report['model_seconds'][s]['q1']}, {report['model_seconds'][s]['q3']} speedup: {report['speedup'][s]}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out, flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
