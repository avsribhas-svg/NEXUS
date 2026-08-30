import os
import sys
import json
import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import inspect
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus.data.dataset import BioelectricDataset
from nexus.model.mpnn import MPNN
from nexus.model.baseline import BaselineMLP
from nexus.evaluation.metrics import compute_mae, compute_r_squared

CHANNEL_NAMES = ["Nav", "Kir", "K_leak", "Ca", "Cl", "NaKATP", "HKATP", "VATP"]

def infer_perturbed_channel(x, ptype) -> str:
    x = x.detach().cpu().numpy()
    if ptype == "gj_blockade":
        return "none"
    elif ptype == "channel_blockade":
        for j in range(6):
            if np.all(x[:, j] == 0.0):
                return CHANNEL_NAMES[j]
        return "none"
    else:
        sd = x.std(axis=0)
        return CHANNEL_NAMES[int(np.argmax(sd))]

def predict_split(model, dataset, device, batch_size=32) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    needs_graph = len(inspect.signature(model.forward).parameters) >= 3
    model.eval()
    preds, trues, degs, gidx, graphs = [], [], [], [], []
    g = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr) if needs_graph else model(batch.x)
        n = batch.x.shape[0]
        deg = torch.zeros(n, device=device)
        if batch.edge_index.shape[1] > 0:
            deg.index_add_(0, batch.edge_index[1], torch.ones(batch.edge_index.shape[1], device=device))
        preds.append(out.detach().cpu().numpy())
        trues.append(batch.y.detach().cpu().numpy())
        degs.append(deg.detach().cpu().numpy())
        gidx.append((batch.batch + g).detach().cpu().numpy())
        g += int(batch.num_graphs)
    return {
        "pred": np.concatenate(preds),
        "true": np.concatenate(trues),
        "deg": np.concatenate(degs),
        "graph": np.concatenate(gidx)
    }

def degree_breakdown(res) -> list:
    d = np.clip(res["deg"].astype(int), 0, 8)
    unique_degrees = np.unique(d)
    return [
        {
            "degree": int(k) if k < 8 else 8,
            "n": int((d == k).sum()),
            "mae": float(compute_mae(res["pred"][d == k], res["true"][d == k]))
        }
        for k in unique_degrees
    ]

def paired_boundary_interior(res) -> dict:
    d = res["deg"].astype(int)
    g = res["graph"].astype(int)
    err = np.abs(res["pred"] - res["true"])
    boundary_mask = (d <= 5)
    interior_mask = (d >= 6)
    
    b_list, i_list = [], []
    for graph in np.unique(g):
        b_err = err[(g == graph) & boundary_mask]
        i_err = err[(g == graph) & interior_mask]
        
        if len(b_err) > 0 and len(i_err) > 0:
            b_list.append(np.mean(b_err))
            i_list.append(np.mean(i_err))
    
    return {
        "n_graphs": int(len(b_list)),
        "boundary_mae": float(np.mean(b_list)) if b_list else 0.0,
        "interior_mae": float(np.mean(i_list)) if i_list else 0.0,
        "mean_paired_difference": float(np.mean(np.array(i_list) - np.array(b_list))) if b_list and i_list else 0.0
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/synthetic")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--arch", type=str, default="mpnn", choices=["mpnn", "mlp"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="outputs/evaluation.json")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = MPNN(n_channels=8) if args.arch == "mpnn" else BaselineMLP(n_channels=8)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    report = {"arch": args.arch, "checkpoint": args.checkpoint,
              "splits": {}, "degree": {}, "paired": {}, "ood": {}}
    ood_ds = None
    ood_res = None

    for split_name in ["test_id", "test_ood"]:
        ds = BioelectricDataset(root=args.data, split=split_name)
        res = predict_split(model, ds, torch.device(device))
        report["splits"][split_name] = {
            "n_cells": int(res["pred"].shape[0]),
            "mae": float(compute_mae(res["pred"], res["true"])),
            "r2": float(compute_r_squared(res["pred"], res["true"]))
        }
        report["degree"][split_name] = degree_breakdown(res)
        report["paired"][split_name] = paired_boundary_interior(res)
        print("%s  MAE %.4f  R2 %.4f" % (split_name, report["splits"][split_name]["mae"], report["splits"][split_name]["r2"]), flush=True)
        print("  degree breakdown:", flush=True)
        for row in report["degree"][split_name]:
            print("    deg %d  n %7d  MAE %7.4f" % (row["degree"], row["n"], row["mae"]), flush=True)
        p = report["paired"][split_name]
        print("  paired within-graph: %d graphs, boundary MAE %.4f, interior MAE %.4f, mean difference %.4f"
              % (p["n_graphs"], p["boundary_mae"], p["interior_mae"], p["mean_paired_difference"]), flush=True)
        if split_name == "test_ood":
            ood_ds = ds
            ood_res = res

    ood_files = sorted(glob.glob(os.path.join(args.data, "test_ood", "*.npz")))
    labels = np.empty(ood_res["pred"].shape[0], dtype=object)
    for i in range(len(ood_ds)):
        z = np.load(ood_files[i], allow_pickle=True)
        ptype = str(z["perturbation_type"])
        ch = infer_perturbed_channel(ood_ds[i].x, ptype)
        labels[ood_res["graph"] == i] = ptype + " / " + ch

    for label in np.unique(labels):
        mask = (labels == label)
        report["ood"][str(label)] = {
            "n": int(mask.sum()),
            "mae": float(compute_mae(ood_res["pred"][mask], ood_res["true"][mask]))
        }
        print("  %-38s n %7d  MAE %7.4f" % (label, report["ood"][str(label)]["n"], report["ood"][str(label)]["mae"]), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out, flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
