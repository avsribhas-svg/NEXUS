import os
import sys
import json
import glob
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus.data.dataset import BioelectricDataset
from nexus.model.mpnn import MPNN
from nexus.model.baseline import BaselineMLP
from nexus.evaluation.metrics import compute_mae, compute_r_squared
from nexus.evaluation.figures import generate_scatter_plot, generate_spatial_error_map

def load_model(arch, checkpoint, device):
    model = MPNN(n_channels=8) if arch == "mpnn" else BaselineMLP(n_channels=8)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device)

def predict(model, ds, device, needs_graph, batch_size=32):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds, trues, gidx = [], [], []
    g = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr) if needs_graph else model(batch.x)
            preds.append(out.cpu().numpy())
            trues.append(batch.y.cpu().numpy())
            gidx.append((batch.batch + g).cpu().numpy())
            g += int(batch.num_graphs)
    return np.concatenate(preds), np.concatenate(trues), np.concatenate(gidx)

def fig_perturbation_panel(pred, true, labels, out_path):
    families = ["channel_blockade", "gj_blockade", "spatial_gradient", "exogenous_expression"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for i, fam in enumerate(families):
        ax = axes[i // 2][i % 2]
        mask = (labels == fam)
        if mask.sum() == 0: 
            ax.set_title(fam + " (no data)")
            continue
        p, t = pred[mask], true[mask]
        ax.scatter(t, p, s=5, alpha=0.4)
        lo, hi = min(p.min(), t.min()), max(p.max(), t.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title("%s\nMAE %.3f mV, R2 %.4f" % (fam, compute_mae(p, t), compute_r_squared(p, t)))
        ax.set_xlabel("True Vmem (mV)")
        ax.set_ylabel("Predicted Vmem (mV)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def fig_learning_curves(run_dirs, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in run_dirs:
        path = os.path.join(d, "summary.json")
        if not os.path.exists(path): 
            continue
        summary = json.load(open(path))
        epochs = range(len(summary["history"]["val_loss"]))
        ax.plot(epochs, summary["history"]["val_loss"], label=os.path.basename(d))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation MAE (mV)")
    ax.set_title("Learning curves"); ax.legend(fontsize=8); ax.set_yscale("log")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

def fig_speed(speed_path, out_path):
    if not os.path.exists(speed_path):
        print("skip fig4, no", speed_path, flush=True)
        return
    rep = json.load(open(speed_path))
    names = ["BETSE"] + ["model " + d for d in rep["model_seconds"]]
    meds = [rep["betse_seconds"]["median"]] + [rep["model_seconds"][d]["median"] for d in rep["model_seconds"]]
    lo = [rep["betse_seconds"]["median"] - rep["betse_seconds"]["q1"]] + [rep["model_seconds"][d]["median"] - rep["model_seconds"][d]["q1"] for d in rep["model_seconds"]]
    hi = [rep["betse_seconds"]["q3"] - rep["betse_seconds"]["median"]] + [rep["model_seconds"][d]["q3"] - rep["model_seconds"][d]["median"] for d in rep["model_seconds"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(names, meds, yerr=[lo, hi], capsize=5, color=["#c44", "#48a", "#4a8"][:len(names)])
    ax.set_yscale("log")
    ax.set_ylabel("Seconds per tissue (log scale)")
    ax.set_title("Time to obtain Vmem for one tissue\nmedian with interquartile range, n=%d" % rep["n_completed"])
    for i, v in enumerate(meds):
        ax.text(i, v, "  %.4g s" % v, ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def fig_ablation_table(abl_path, out_path):
    if not os.path.exists(abl_path):
        print("skip fig5, no", abl_path, flush=True)
        return
    rows = json.load(open(abl_path))
    header = ["tag", "arch", "K", "n_train", "norm", "params", "test_id MAE", "test_ood MAE", "epochs"]
    cells = []
    for r in rows:
        cells.append([str(r["tag"]), str(r["arch"]), str(r["n_layers"]), str(r["train_size"]),
                      str(r["normalized"]), "%d" % r["n_parameters"],
                      "%.4f" % r["test_id_mae"], "%.4f" % r["test_ood_mae"], "%d" % r["epochs_run"]])
    fig, ax = plt.subplots(figsize=(13, 0.45 * (len(cells) + 2)))
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.4)
    ax.set_title("Ablations (seed 42)", pad=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/synthetic")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--arch", type=str, default="mpnn", choices=["mpnn","mlp"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--outdir", type=str, default="outputs/figures")
    parser.add_argument("--runs", type=str, default="outputs")
    args = parser.parse_args()
    
    device = args.device if not (args.device.startswith("cuda") and not torch.cuda.is_available()) else "cpu"
    device = torch.device(device)
    os.makedirs(args.outdir, exist_ok=True)
    needs_graph = args.arch == "mpnn"
    model = load_model(args.arch, args.checkpoint, device)

    # FIGURE 1
    ds = BioelectricDataset(root=args.data, split="test_id")
    pred, true, _ = predict(model, ds, device, needs_graph)
    generate_scatter_plot(pred, true, os.path.join(args.outdir, "fig1_scatter_test_id.png"))
    print(f"MAE: {compute_mae(pred, true):.3f} mV, R2: {compute_r_squared(pred, true):.4f}")

    # FIGURE 2
    ood = BioelectricDataset(root=args.data, split="test_ood")
    opred, otrue, ograph = predict(model, ood, device, needs_graph)
    files = sorted(glob.glob(os.path.join(args.data, "test_ood", "*.npz")))
    labels = np.empty(opred.shape[0], dtype=object)
    for i in range(len(ood)):
        labels[ograph == i] = str(np.load(files[i], allow_pickle=True)["perturbation_type"])
    fig_perturbation_panel(opred, otrue, labels, os.path.join(args.outdir, "fig2_perturbation_panel.png"))

    # FIGURE 3
    for j in range(2):
        d = ds[j]
        p, t, _ = predict(model, ds[j:j+1], device, needs_graph)
        generate_spatial_error_map(d.pos.numpy(), p, t, os.path.join(args.outdir, "fig3_spatial_id_%d.png" % j))
    for j in range(4):
        d = ood[j * 500]
        p, t, _ = predict(model, ood[j*500:j*500+1], device, needs_graph)
        generate_spatial_error_map(d.pos.numpy(), p, t, os.path.join(args.outdir, "fig3_spatial_ood_%d.png" % j))

    # FIGURE 6
    run_dirs = sorted(glob.glob(os.path.join(args.runs, "*seed42")))
    fig_learning_curves(run_dirs, os.path.join(args.outdir, "fig6_learning_curves.png"))

    fig_speed(os.path.join(args.runs, "speed_benchmark.json"), os.path.join(args.outdir, "fig4_speed.png"))
    fig_ablation_table(os.path.join(args.runs, "ablations.json"), os.path.join(args.outdir, "fig5_ablations.png"))

    print("\n".join(sorted(os.listdir(args.outdir))))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
