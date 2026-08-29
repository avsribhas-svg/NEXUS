import torch
import torch.nn as nn

class MPNN(nn.Module):
    def __init__(self, n_channels: int, n_layers: int = 6):
        super().__init__()
        self.n_layers = n_layers
        self.hidden = 128
        self.out_scale = 30.0
        self.out_bias = -50.0

        self.encoder = nn.Sequential(
            nn.Linear(n_channels, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(1, 32), nn.LayerNorm(32), nn.ReLU(),
            nn.Linear(32, 64), nn.LayerNorm(64), nn.ReLU(),
        )

        self.message_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(320, 128), nn.ReLU(), nn.Linear(128, 128))
            for _ in range(n_layers)
        ])

        self.update_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 128))
            for _ in range(n_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(128) for _ in range(n_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        n_nodes = x.shape[0]
        n_edges = edge_index.shape[1]

        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        h = self.encoder(x)
        e = self.edge_encoder(edge_attr)
        src, dst = edge_index

        deg = torch.zeros(n_nodes, dtype=h.dtype, device=h.device)
        if n_edges > 0:
            deg.index_add_(0, dst, torch.ones(n_edges, dtype=h.dtype, device=h.device))
        deg = deg.clamp(min=1.0).unsqueeze(-1)

        for k in range(self.n_layers):
            agg = torch.zeros(n_nodes, 128, dtype=h.dtype, device=h.device)
            if n_edges > 0:
                msg = self.message_mlps[k](torch.cat([h[dst], h[src], e], dim=-1))
                agg = agg.index_add(0, dst, msg)
            agg = agg / deg

            upd = self.update_mlps[k](torch.cat([h, agg], dim=-1))
            h = self.layer_norms[k](h + upd)

        out = self.decoder(h).squeeze(-1)
        return out * self.out_scale + self.out_bias
