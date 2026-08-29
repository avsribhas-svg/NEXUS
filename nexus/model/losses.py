import torch

def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()

def physics_auxiliary_loss(pred: torch.Tensor, edge_index: torch.Tensor, conductances: torch.Tensor) -> torch.Tensor:
    if edge_index.shape[1] == 0:
        return torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
    
    n_nodes = pred.shape[0]
    current = conductances * (pred[edge_index[0]] - pred[edge_index[1]])
    net = torch.zeros(n_nodes, dtype=pred.dtype, device=pred.device)
    net.index_add_(0, edge_index[0], current)
    return (net ** 2).mean()
