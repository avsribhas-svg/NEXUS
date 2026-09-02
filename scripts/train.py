import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHANNEL_MAXES = torch.tensor([50.0, 30.0, 20.0, 10.0, 15.0, 30.0, 10.0, 10.0], dtype=torch.float32)

from nexus.data.dataset import BioelectricDataset
from nexus.model.mpnn import MPNN
from nexus.model.baseline import BaselineMLP
from nexus.training.trainer import Trainer
from nexus.training.config import TrainingConfig
from nexus.evaluation.metrics import compute_mae, compute_r_squared

def denormalize(data):
    data.x = data.x * CHANNEL_MAXES
    data.edge_attr = data.edge_attr * 50.0
    return data

def evaluate(model, loader, device) -> tuple:
    model.eval()
    needs_graph = len(inspect.signature(model.forward).parameters) >= 3
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr) if needs_graph else model(batch.x)
            preds.append(out.detach().cpu().numpy())
            trues.append(batch.y.detach().cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mae = compute_mae(preds, trues)
    r2 = compute_r_squared(preds, trues)
    return (mae, r2, preds, trues)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/synthetic")
    parser.add_argument("--arch", type=str, default="mpnn", choices=["mpnn", "mlp"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--train-size", type=int, default=0)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--physics-weight", type=float, default=0.0)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print("device:", device, flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tf = denormalize if args.no_normalize else None
    train_ds = BioelectricDataset(root=args.data, split="train", transform=tf)
    val_ds = BioelectricDataset(root=args.data, split="val", transform=tf)
    test_id_ds = BioelectricDataset(root=args.data, split="test_id", transform=tf)
    test_ood_ds = BioelectricDataset(root=args.data, split="test_ood", transform=tf)

    if args.train_size > 0 and args.train_size < len(train_ds):
        train_ds = train_ds[:args.train_size]

    print("splits: train %d val %d test_id %d test_ood %d"
          % (len(train_ds), len(val_ds), len(test_id_ds), len(test_ood_ds)), flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_id_loader = DataLoader(test_id_ds, batch_size=args.batch_size)
    test_ood_loader = DataLoader(test_ood_ds, batch_size=args.batch_size)

    if args.arch == "mpnn":
        model = MPNN(n_channels=8, n_layers=args.n_layers)
    else:
        model = BaselineMLP(n_channels=8)
    n_params = sum(p.numel() for p in model.parameters())
    print("arch %s, %d parameters" % (args.arch, n_params), flush=True)

    run_name = args.tag if args.tag else "%s_seed%d" % (args.arch, args.seed)
    run_dir = os.path.join(args.out, run_name)
    os.makedirs(run_dir, exist_ok=True)
    config = TrainingConfig(lr=args.lr, max_epochs=args.epochs, patience=args.patience,
                            checkpoint_dir=run_dir, device=device,
                            physics_loss_weight=args.physics_weight)

    t0 = time.time()
    trainer = Trainer(model=model, config=config)
    history = trainer.train(train_loader, val_loader)
    train_seconds = time.time() - t0
    print("trained %d epochs in %.1f s" % (len(history["train_loss"]), train_seconds), flush=True)

    results = {}
    for name, loader in (("train", train_loader), ("val", val_loader),
                         ("test_id", test_id_loader), ("test_ood", test_ood_loader)):
        mae, r2, _, _ = evaluate(model, loader, torch.device(device))
        results[name] = {"mae": float(mae), "r2": float(r2)}
        print("  %-8s MAE %7.3f mV   R2 %7.4f" % (name, mae, r2), flush=True)

    trainer.save_checkpoint(os.path.join(run_dir, "final.pt"))
    summary = {
        "arch": args.arch,
        "seed": args.seed,
        "device": device,
        "tag": run_name,
        "n_layers": args.n_layers if args.arch == "mpnn" else None,
        "train_size": len(train_ds),
        "normalized": not args.no_normalize,
        "physics_weight": args.physics_weight,
        "n_parameters": int(n_params),
        "epochs_run": len(history["train_loss"]),
        "train_seconds": float(train_seconds),
        "results": results,
        "history": {k: [float(x) for x in v] for k, v in history.items()},
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", os.path.join(run_dir, "summary.json"), flush=True)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
