import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus.data.dataset import BioelectricDataset
from nexus.data.experimental import load_experimental_records, apply_operation
from nexus.model.mpnn import MPNN
from nexus.model.baseline import BaselineMLP
from nexus.evaluation.metrics import compute_mae

def predict_graph_means(model, ds, device, needs_graph, batch_size=32) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    means = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr) if needs_graph else model(batch.x)
            nb = int(batch.num_graphs)
            for g in range(nb):
                means.append(float(out[batch.batch == g].mean().item()))
    return np.array(means)

def predict_one(model, data, device, needs_graph) -> float:
    d = data.to(device)
    with torch.no_grad():
        out = model(d.x, d.edge_index, d.edge_attr) if needs_graph else model(d.x)
    return float(out.mean().item())

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate experimental records against model predictions.")
    parser.add_argument("--data", type=str, default="data/synthetic", help="Path to the dataset directory")
    parser.add_argument("--records", type=str, default="data/experimental/experimental_records.json", help="Path to the experimental records file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--arch", type=str, default="mpnn", choices=["mpnn","mlp"], help="Model architecture")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., cuda or cpu)")
    parser.add_argument("--tolerance", type=float, default=3.0, help="Tolerance for matching baseline tissues")
    parser.add_argument("--max-ensemble", type=int, default=40, help="Maximum number of baselines in the ensemble")
    parser.add_argument("--outdir", type=str, default="outputs/figures", help="Output directory for figures")
    parser.add_argument("--out", type=str, default="outputs/experimental_validation.json", help="Output file for results")
    args = parser.parse_args()
    
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available(): 
        device = "cpu"
    device = torch.device(device)
    
    needs_graph = args.arch == "mpnn"
    os.makedirs(args.outdir, exist_ok=True)

    model = MPNN(n_channels=8) if args.arch == "mpnn" else BaselineMLP(n_channels=8)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    ds = BioelectricDataset(root=args.data, split="train")
    true_means = np.array([float(ds[i].y.mean().item()) for i in range(len(ds))])

    recs = load_experimental_records(args.records)
    results = []

    for r in recs["perturbation_pairs"]:
        v0 = r["vmem_control_mv"]
        idx = np.where(np.abs(true_means - v0) <= args.tolerance)[0]
        if idx.size == 0:
            results.append({"id": r["id"], "n_matched": 0, "status": "no baseline within tolerance"})
            continue
        if idx.size > args.max_ensemble:
            idx = idx[np.linspace(0, idx.size - 1, args.max_ensemble).astype(int)]
        deltas = []
        for i in idx:
            base = ds[int(i)]
            pert = apply_operation(base, r["model_operation"])
            vb = predict_one(model, base, device, needs_graph)
            vp = predict_one(model, pert, device, needs_graph)
            deltas.append(vp - vb)
        deltas = np.array(deltas)
        row = {
            "id": r["id"],
            "family": r["perturbation_family"],
            "operation": r["model_operation"],
            "n_matched": int(idx.size),
            "delta_exp_mv": float(r["delta_vmem_mv"]),
            "delta_pred_median_mv": float(np.median(deltas)),
            "delta_pred_q1_mv": float(np.percentile(deltas, 25)),
            "delta_pred_q3_mv": float(np.percentile(deltas, 75)),
            "abs_error_mv": float(abs(np.median(deltas) - r["delta_vmem_mv"])),
            "sign_correct": bool(np.sign(np.median(deltas)) == np.sign(r["delta_vmem_mv"])),
            "status": "ok"
        }
        results.append(row)
        print("%-32s exp %+7.2f  pred %+7.2f [%+.2f, %+.2f]  err %6.2f  sign %s  n=%d" 
              % (row["id"], row["delta_exp_mv"], row["delta_pred_median_mv"],
                 row["delta_pred_q1_mv"], row["delta_pred_q3_mv"],
                 row["abs_error_mv"], "OK" if row["sign_correct"] else "WRONG", row["n_matched"]), flush=True)

    ok = [r for r in results if r["status"] == "ok"]
    anchors = [a["vmem_mv"] for a in recs["baseline_anchors"]]
    vmem_range = float(max(anchors) - min(anchors))
    summary = {
        "n_records": len(results),
        "n_evaluated": len(ok),
        "delta_mae_mv": float(np.mean([r["abs_error_mv"] for r in ok])) if ok else None,
        "n_sign_correct": int(sum(1 for r in ok if r["sign_correct"])),
        "experimental_vmem_range_mv": vmem_range,
        "accuracy_threshold_mv": 0.10 * vmem_range,
        "meets_threshold": bool(ok) and float(np.mean([r["abs_error_mv"] for r in ok])) <= 0.10 * vmem_range
    }
    print("n_records:", summary["n_records"])
    print("n_evaluated:", summary["n_evaluated"])
    print("delta_mae_mv:", summary["delta_mae_mv"])
    print("n_sign_correct:", summary["n_sign_correct"])
    print("experimental_vmem_range_mv:", summary["experimental_vmem_range_mv"])
    print("accuracy_threshold_mv:", summary["accuracy_threshold_mv"])
    print("meets_threshold:", summary["meets_threshold"])

    fig, ax = plt.subplots(figsize=(7, 7))
    xs = [r["delta_exp_mv"] for r in ok]
    ys = [r["delta_pred_median_mv"] for r in ok]
    lo_err = [r["delta_pred_median_mv"] - r["delta_pred_q1_mv"] for r in ok]
    hi_err = [r["delta_pred_q3_mv"] - r["delta_pred_median_mv"] for r in ok]
    ax.errorbar(xs, ys, yerr=[lo_err, hi_err], fmt="o", capsize=4, markersize=8)
    lim = max(abs(min(xs + ys)), abs(max(xs + ys))) * 1.3 + 1
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, label="perfect agreement")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    for r in ok:
        ax.annotate(r["id"], (r["delta_exp_mv"], r["delta_pred_median_mv"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Measured change in Vmem (mV)")
    ax.set_ylabel("Predicted change in Vmem (mV), median of matched ensemble")
    ax.set_title("Experimental validation: predicted vs measured perturbation response")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig7_experimental_validation.png"), dpi=150)
    plt.close(fig)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({"summary": summary, "records": results}, f, indent=2)
    print("wrote", args.out)
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
