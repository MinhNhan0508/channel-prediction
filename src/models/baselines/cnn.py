"""CNN baseline -- coi chuỗi CSI như ảnh (Conv2D không-gian + Conv1D thời-gian).

Có trong CPMamba/ChannelMamba. Input giữ cấu trúc không gian (KHÔNG flatten):
    (B, P, N_RF, N_M, 2) -> Conv2D mỗi khung -> đặc trưng (B,P,d)
                         -> Conv1D theo thời gian -> map P->Q -> giải mã
    -> (B, Q, N_RF, N_M, 2)
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    def __init__(
        self,
        n_rf: int = 8,
        n_m: int = 4,
        p_frames: int = 30,
        q_frames: int = 5,
        d_model: int = 128,
        spatial_channels: Sequence[int] = (16, 32),
        temporal_channels: int = 128,
        dropout: float = 0.1,
        **_unused,
    ):
        super().__init__()
        self.n_rf, self.n_m = n_rf, n_m
        self.p_frames, self.q_frames = p_frames, q_frames
        self.d_model = d_model

        # --- Spatial Conv2D (per-frame) ---
        layers = []
        in_ch = 2
        for out_ch in spatial_channels:
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1),
                       nn.GroupNorm(min(8, out_ch), out_ch), nn.ReLU(inplace=True)]
            in_ch = out_ch
        self.spatial = nn.Sequential(*layers)
        self.spatial_fc = nn.Linear(spatial_channels[-1] * n_rf * n_m, d_model)

        # --- Temporal Conv1D ---
        self.temporal = nn.Sequential(
            nn.Conv1d(d_model, temporal_channels, 3, padding=1),
            nn.GroupNorm(min(8, temporal_channels), temporal_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(temporal_channels, d_model, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # map độ dài thời gian P -> Q
        self.time_proj = nn.Linear(p_frames, q_frames)
        # giải mã đặc trưng -> kênh
        self.out_proj = nn.Linear(d_model, n_rf * n_m * 2)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, N_RF, N_M, 2)
        B, P, nrf, nm, _ = x.shape
        h = x.permute(0, 1, 4, 2, 3).reshape(B * P, 2, nrf, nm)
        h = self.spatial(h).flatten(1)                      # (B*P, c*nrf*nm)
        h = self.spatial_fc(h).view(B, P, self.d_model)     # (B,P,d)
        h = self.temporal(h.transpose(1, 2))                # (B,d,P)
        h = self.time_proj(h).transpose(1, 2)               # (B,Q,d)
        out = self.out_proj(h).view(B, self.q_frames, nrf, nm, 2)
        return out


if __name__ == "__main__":
    m = CNNBaseline()
    x = torch.randn(2, 30, 8, 4, 2)
    y = m(x)
    assert y.shape == (2, 5, 8, 4, 2), f"got {y.shape}"
    print(f"CNN OK | params: {sum(p.numel() for p in m.parameters() if p.requires_grad):,}")
