"""RNN baseline with parallel cross-attention prediction head.

Same parallel prediction mechanism as ST-Mamba (Q learned queries attending to
encoded history) so the comparison isolates the temporal backbone choice.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RNNBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        hidden_size: int = 256,
        num_layers: int = 3,
        p_frames: int = 30,
        q_frames: int = 5,
        dropout: float = 0.1,
        n_pred_heads: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.p_frames = p_frames
        self.q_frames = q_frames

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity="tanh",
        )

        self.queries = nn.Parameter(torch.randn(q_frames, hidden_size) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_pred_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, input_dim)
        # Small init so initial output magnitude matches the normalized channel
        # range (~0.1) instead of ~sqrt(hidden_size). This makes overfit converge
        # in tens of steps instead of hundreds.
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, input_dim)
        B = x.shape[0]
        h, _ = self.rnn(x)  # (B, P, hidden_size)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(query=q, key=h, value=h)
        q = self.norm(q + attn_out)
        return self.output_proj(q)  # (B, Q, input_dim)


if __name__ == "__main__":
    model = RNNBaseline()
    x = torch.randn(2, 30, 64)
    y = model(x)
    assert y.shape == (2, 5, 64), f"got {y.shape}"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"RNN OK | params: {n_params:,}")
