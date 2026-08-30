import os
import inspect
import torch
from nexus.model.losses import mae_loss

class Trainer:
    def __init__(self, model, config):
        self.device = torch.device(getattr(config, "device", "cpu"))
        self.model = model.to(self.device)
        self.config = config
        self.history = {"train_loss": [], "val_loss": [], "lr": []}
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        params = inspect.signature(self.model.forward).parameters
        self.needs_graph = len(params) >= 3

    def _predict(self, batch):
        batch = batch.to(self.device)
        if self.needs_graph:
            return self.model(batch.x, batch.edge_index, batch.edge_attr)
        else:
            return self.model(batch.x)

    def train(self, train_loader, val_loader):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, self.config.max_epochs), eta_min=self.config.min_lr)
        best_val = float("inf")
        best_state = None
        epochs_without_improvement = 0
        self.history = {"train_loss": [], "val_loss": [], "lr": []}

        for epoch in range(self.config.max_epochs):
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            self.model.train()
            train_total = 0.0
            n_train = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                pred = self._predict(batch)
                loss = mae_loss(pred, batch.y.to(self.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.optimizer.step()
                train_total += loss.item()
                n_train += 1
            self.history["train_loss"].append(train_total / max(1, n_train))

            self.model.eval()
            val_total = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    pred = self._predict(batch)
                    loss = mae_loss(pred, batch.y.to(self.device))
                    val_total += loss.item()
                    n_val += 1
            val_loss = val_total / max(1, n_val)
            self.history["val_loss"].append(val_loss)

            scheduler.step()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
                if self.config.checkpoint_dir is not None:
                    os.makedirs(self.config.checkpoint_dir, exist_ok=True)
                    torch.save({"model_state_dict": best_state, "epoch": epoch, "val_loss": val_loss},
                               os.path.join(self.config.checkpoint_dir, "best.pt"))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history

    def save_checkpoint(self, path):
        torch.save({"model_state_dict": self.model.state_dict(),
                      "optimizer_state_dict": self.optimizer.state_dict(),
                      "history": self.history}, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        return checkpoint
