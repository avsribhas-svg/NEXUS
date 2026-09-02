import os
import sys
import json
import time
import argparse
import subprocess

ABLATIONS = [
    {"tag": "abl_mlp_baseline", "flags": ["--arch", "mlp"]},
    {"tag": "abl_depth_k2",     "flags": ["--arch", "mpnn", "--n-layers", "2"]},
    {"tag": "abl_depth_k4",     "flags": ["--arch", "mpnn", "--n-layers", "4"]},
    {"tag": "abl_depth_k6",     "flags": ["--arch", "mpnn", "--n-layers", "6"]},
    {"tag": "abl_depth_k8",     "flags": ["--arch", "mpnn", "--n-layers", "8"]},
    {"tag": "abl_size_1000",    "flags": ["--arch", "mpnn", "--train-size", "1000"]},
    {"tag": "abl_size_2000",    "flags": ["--arch", "mpnn", "--train-size", "2000"]},
    {"tag": "abl_size_4000",    "flags": ["--arch", "mpnn", "--train-size", "4000"]},
    {"tag": "abl_size_8000",    "flags": ["--arch", "mpnn", "--train-size", "8000"]},
    {"tag": "abl_no_normalize", "flags": ["--arch", "mpnn", "--no-normalize"]},
    {"tag": "abl_physics_loss", "flags": ["--arch", "mpnn", "--physics-weight", "0.01"]},
]

def run_one(ab, args) -> bool:
    summary_path = os.path.join(args.out, ab["tag"], "summary.json")
    if os.path.exists(summary_path) and not args.force:
        print("SKIP %s (already done)" % ab["tag"], flush=True)
        return True
    cmd = [sys.executable, os.path.join("scripts", "train.py"),
           "--data", args.data, "--seed", str(args.seed), "--device", args.device,
           "--epochs", str(args.epochs), "--out", args.out, "--tag", ab["tag"]] + ab["flags"]
    print("RUN  %s" % ab["tag"], flush=True)
    print("     " + " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        for line in proc.stdout.splitlines()[-8:]:
            print("     " + line, flush=True)
    else:
        for line in proc.stderr.splitlines()[-15:]:
            print("     " + line, flush=True)
    print("     exit %d in %.1f s" % (proc.returncode, time.time() - t0), flush=True)
    return proc.returncode == 0

def collect(args) -> list:
    rows = []
    for ab in ABLATIONS:
        summary_path = os.path.join(args.out, ab["tag"], "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                d = json.load(f)
            row = {
                "tag": ab["tag"],
                "arch": d.get("arch"),
                "n_layers": d.get("n_layers"),
                "train_size": d.get("train_size"),
                "normalized": d.get("normalized"),
                "physics_weight": d.get("physics_weight"),
                "n_parameters": d.get("n_parameters"),
                "epochs_run": d.get("epochs_run"),
                "train_seconds": d.get("train_seconds"),
                "test_id_mae": d["results"]["test_id"]["mae"],
                "test_id_r2": d["results"]["test_id"]["r2"],
                "test_ood_mae": d["results"]["test_ood"]["mae"],
                "test_ood_r2": d["results"]["test_ood"]["r2"]
            }
            rows.append(row)
    return rows

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/synthetic")
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    
    if not args.collect_only:
        for ab in ABLATIONS:
            run_one(ab, args)
    
    rows = collect(args)
    
    print("%-20s %-6s %6s %7s %6s %8s %9s %9s %9s" % ("tag","arch","K","n_train","norm","params","test_id","test_ood","epochs"), flush=True)
    for r in rows:
        print("%-20s %-6s %6s %7s %6s %8d %9.4f %9.4f %9d"
              % (r["tag"], r["arch"], str(r["n_layers"]), str(r["train_size"]), str(r["normalized"]),
                 r["n_parameters"], r["test_id_mae"], r["test_ood_mae"], r["epochs_run"]), flush=True)
    
    with open(os.path.join(args.out, "ablations.json"), 'w') as f:
        json.dump(rows, f, indent=2)
    print("wrote", os.path.join(args.out, "ablations.json"), flush=True)
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
