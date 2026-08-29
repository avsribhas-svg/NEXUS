import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import sys
import csv
import time
import argparse
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus.data.config_sampler import ConfigSampler
from nexus.data.betse_generator import BETSEGenerator
from nexus.data.validation import validate_simulation_result
from nexus.utils.io import save_record

PERTURBATION_TYPES = ["channel_blockade", "gj_blockade", "exogenous_expression", "spatial_gradient"]
CSV_FIELDS = ["config_id", "status", "perturbation_type", "is_perturbation", "n_cells_target",
              "n_cells_actual", "wall_clock_s", "converged_fraction", "max_dvmem_mv", "errors"]

def build_work_list(n_baseline: int, n_per_perturbation: int, seed: int, timeseries_count: int) -> list:
    work = []
    # Baseline configurations
    base_cfgs = ConfigSampler(seed=seed).sample(n=n_baseline)
    for i, cfg in enumerate(base_cfgs):
        cfg["config_id"] = "base_%06d" % i
        work.append((cfg, i < timeseries_count))
    # Perturbation configurations, one sampler per type so the types are independent
    for k, ptype in enumerate(PERTURBATION_TYPES):
        cfgs = ConfigSampler(seed=seed + 1 + k).sample(n=n_per_perturbation, perturbation_type=ptype)
        for i, cfg in enumerate(cfgs):
            cfg["config_id"] = "%s_%06d" % (ptype, i)
            work.append((cfg, False))
    return work

def run_one(cfg: dict, capture_timeseries: bool, out_dir: str, timeout_s: float) -> dict:
    config_id = str(cfg["config_id"])
    path = os.path.join(out_dir, config_id + ".npz")
    row = {
        "config_id": config_id,
        "status": "",
        "perturbation_type": "" if cfg.get("perturbation_type") is None else str(cfg["perturbation_type"]),
        "is_perturbation": bool(cfg.get("is_perturbation", False)),
        "n_cells_target": int(cfg.get("n_cells", 0)),
        "n_cells_actual": 0,
        "wall_clock_s": 0.0,
        "converged_fraction": 0.0,
        "max_dvmem_mv": 0.0,
        "errors": ""
    }
    if os.path.exists(path):
        row["status"] = "skipped"
        return row
    t0 = time.time()
    try:
        record = BETSEGenerator().run(cfg, timeout_s=timeout_s, capture_timeseries=capture_timeseries)
        row["wall_clock_s"] = float(time.time() - t0)
        if record is None:
            row["status"] = "betse_failed"
            return row
        errors = validate_simulation_result(record)
        row["n_cells_actual"] = int(record["tissue_geometry"]["n_cells"])
        row["converged_fraction"] = float(record["metadata"].get("converged_fraction", 0.0))
        row["max_dvmem_mv"] = float(record["metadata"].get("max_dvmem_mv", 0.0))
        if errors:
            row["status"] = "invalid"
            row["errors"] = "; ".join(str(e) for e in errors)[:300]
            return row
        save_record(record, path)
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = "exception"
        row["errors"] = repr(exc)[:300]
        row["wall_clock_s"] = float(time.time() - t0)
    return row

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-baseline", type=int, default=11500)
    parser.add_argument("--n-per-perturbation", type=int, default=575)
    parser.add_argument("--out", type=str, default="data/synthetic/_staged")
    parser.add_argument("--jobs", type=int, default=18)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeseries-count", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.csv")
    done = set()
    if os.path.exists(results_path):
        with open(results_path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("status") in ("ok", "invalid", "betse_failed", "exception"):
                    done.add(r["config_id"])
    work = build_work_list(args.n_baseline, args.n_per_perturbation, args.seed, args.timeseries_count)
    work = [(c, t) for (c, t) in work if c["config_id"] not in done]
    if args.limit > 0:
        work = work[:args.limit]
    print(f"Total configurations to run: {len(work)}, Already done: {len(done)}, Jobs: {args.jobs}")
    write_header = not os.path.exists(results_path) or os.path.getsize(results_path) == 0
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        parallel = Parallel(n_jobs=args.jobs, backend="loky", return_as="generator_unordered", batch_size=1)
        tasks = (delayed(run_one)(c, t, args.out, args.timeout) for (c, t) in work)
        completed = 0
        successes = 0
        n_timed = 0
        total_wall_clock_s = 0.0
        status_counts = {}
        start_time = time.time()
        for row in parallel(tasks):
            writer.writerow(row)
            f.flush()
            completed += 1
            status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
            if row["status"] == "ok":
                successes += 1
            if row["status"] != "skipped":
                n_timed += 1
                total_wall_clock_s += float(row["wall_clock_s"])
            if completed % 10 == 0:
                mean_wall_clock = total_wall_clock_s / max(1, n_timed)
                remaining = ((len(work) - completed) * mean_wall_clock) / max(1, args.jobs)
                print("completed %d/%d  ok=%d  mean=%.1fs  eta=%.2fh"
                      % (completed, len(work), successes, mean_wall_clock, remaining / 3600.0),
                      flush=True)
        print(f"Summary: Total completed: {completed}, Successes: {successes}, Failures: {completed - successes}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
