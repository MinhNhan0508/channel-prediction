"""GRU baseline with parallel cross-attention prediction head."""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        hidden_size: int = 224,
        num_layers: int = 2,
        p_frames: int = 30,
        q_frames: int = 5,
        dropout: float = 0.1,
        n_pred_heads: int = 4,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.queries = nn.Parameter(torch.randn(q_frames, hidden_size) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=n_pred_heads, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, input_dim)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h, _ = self.gru(x)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(query=q, key=h, value=h)
        q = self.norm(q + attn_out)
        return self.output_proj(q)


if __name__ == "__main__":
    model = GRUBaseline()
    x = torch.randn(2, 30, 64)
    y = model(x)
    assert y.shape == (2, 5, 64), f"got {y.shape}"
    print(f"GRU OK | params: {sum(p.numel() for p in model.parameters()):,}")
