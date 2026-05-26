"""Spatial encoder: 2D CNN that compresses one effective-channel slice into a
spatial feature vector, applied independently per timestep.

Input  : (B, P, N_RF, N_M, 2)  -- real/imag as 2 channels of an (N_RF, N_M) image
Output : (B, P, d_spatial)
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        n_rf: int = 8,
        n_m: int = 4,
        d_spatial: int = 64,
        cnn_channels: Sequence[int] = (16, 32, 64),
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.n_rf = n_rf
        self.n_m = n_m
        self.d_spatial = d_spatial

        layers = []
        in_ch = 2  # real, imag
        for out_ch in cnn_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.conv_net = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cnn_channels[-1], d_spatial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, N_RF, N_M, 2)
        B, P, N_RF, N_M, C = x.shape
        assert C == 2, f"expected last dim 2 (real,imag), got {C}"

        x = x.permute(0, 1, 4, 2, 3).contiguous()       # (B, P, 2, N_RF, N_M)
        x = x.view(B * P, 2, N_RF, N_M)
        x = self.conv_net(x)
        x = self.pool(x).flatten(1)                     # (B*P, last_ch)
        x = self.fc(x)                                  # (B*P, d_spatial)
        return x.view(B, P, self.d_spatial)


if __name__ == "__main__":
    enc = SpatialEncoder()
    x = torch.randn(2, 30, 8, 4, 2)
    y = enc(x)
    assert y.shape == (2, 30, 64), f"got {y.shape}"
    n_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print(f"SpatialEncoder OK | params: {n_params:,}")
