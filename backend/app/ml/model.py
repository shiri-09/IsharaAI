import torch
from torch import nn


class SignTransformer(nn.Module):
    """Transformer encoder over a keypoint sequence -> sentence class (CLS-token pooling)."""

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        seq_len: int = 64,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, in_dim)
        h = self.proj(x)
        h = torch.cat([self.cls.expand(h.size(0), -1, -1), h], dim=1) + self.pos[:, : h.size(1) + 1]
        h = self.encoder(h)
        return self.head(self.norm(h[:, 0]))
