from dataclasses import dataclass

@dataclass
class TrainingConfig:
    lr: float
    max_epochs: int
    patience: int
    checkpoint_dir: str = None
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-4
    min_lr: float = 1e-5
    device: str = "cpu"
    physics_loss_weight: float = 0.0
