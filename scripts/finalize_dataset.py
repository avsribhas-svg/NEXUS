import os
import csv
import sys
import shutil
import argparse

PERTURBATION_TYPES = ["channel_blockade", "gj_blockade", "exogenous_expression", "spatial_gradient"]
BASELINE_SPLITS = [("train", 8000), ("val", 1000), ("test_id", 1000)]
MANIFEST_FIELDS = ["config_id", "split", "path", "n_cells", "is_perturbation", "perturbation_type"]

def read_ok_rows(results_path: str) -> list:
    with open(results_path, newline="") as f:
        reader = csv.DictReader(f)
        return sorted([row for row in reader if row["status"] == "ok"], key=lambda x: x["config_id"])

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", type=str, default="data/synthetic/_staged")
    parser.add_argument("--out", type=str, default="data/synthetic")
    parser.add_argument("--move", action="store_true")
    args = parser.parse_args()
    
    results_path = os.path.join(args.staged, "results.csv")
    if not os.path.exists(results_path):
        print(f"Error: {results_path} does not exist.")
        return 1
    
    rows = read_ok_rows(results_path)
    baseline_rows = [r for r in rows if r["perturbation_type"] == ""]
    pert_rows_by_type = {ptype: [r for r in rows if r["perturbation_type"] == ptype] for ptype in PERTURBATION_TYPES}
    
    os.makedirs(os.path.join(args.out, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "val"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "test_id"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "test_ood"), exist_ok=True)
    
    assignments = []
    cursor = 0
    for split_name, target in BASELINE_SPLITS:
        chunk = baseline_rows[cursor:cursor + target]
        cursor += len(chunk)
        for r in chunk:
            assignments.append((r, split_name))
    
    surplus_baseline = len(baseline_rows) - cursor
    
    surplus_pert = 0
    for ptype in PERTURBATION_TYPES:
        available = pert_rows_by_type[ptype]
        for r in available[:500]:
            assignments.append((r, "test_ood"))
        surplus_pert += max(0, len(available) - 500)
    
    manifest = []
    missing = 0
    for r, split_name in assignments:
        src = os.path.join(args.staged, r["config_id"] + ".npz")
        if not os.path.exists(src):
            missing += 1
            continue
        rel = os.path.join(split_name, r["config_id"] + ".npz")
        dst = os.path.join(args.out, rel)
        if args.move:
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)
        manifest.append({
            "config_id": r["config_id"],
            "split": split_name,
            "path": rel.replace("\\", "/"),
            "n_cells": r["n_cells_actual"],
            "is_perturbation": r["is_perturbation"],
            "perturbation_type": r["perturbation_type"],
        })
    
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    
    print(f"Total successful rows read: {len(rows)}")
    for split_name, target in BASELINE_SPLITS + [("test_ood", 2000)]:
        count = sum(1 for r, s in assignments if s == split_name)
        print(f"{split_name}: {count} entries received")
    print(f"Surplus baseline count: {surplus_baseline}")
    print(f"Surplus perturbation count: {surplus_pert}")
    print(f"Missing file count: {missing}")
    
    for split_name, target in BASELINE_SPLITS + [("test_ood", 2000)]:
        count = sum(1 for r, s in assignments if s == split_name)
        if count < target:
            print(f"{split_name} received fewer entries than its target: {target} vs {count}")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
