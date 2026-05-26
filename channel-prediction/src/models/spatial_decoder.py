"""Spatial decoder: reconstruct (N_RF, N_M, 2) channel from the d_spatial feature.

Input  : (B, Q, d_spatial)
Output : (B, Q, N_RF, N_M, 2)
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class SpatialDecoder(nn.Module):
    def __init__(
        self,
        n_rf: int = 8,
        n_m: int = 4,
        d_spatial: int = 64,
        hidden_channels: Sequence[int] = (64, 32, 16),
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.n_rf = n_rf
        self.n_m = n_m
        self.d_spatial = d_spatial

        self.fc = nn.Linear(d_spatial, hidden_channels[0] * n_rf * n_m)

        layers = []
        in_ch = hidden_channels[0]
        for out_ch in hidden_channels[1:]:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        final_conv = nn.Conv2d(in_ch, 2, kernel_size=3, padding=1)
        # Small init on final conv: output starts near zero so the network does
        # not waste hundreds of optimization steps fighting an oversized output
        # magnitude before fitting the (~0.1 std) channel targets.
        nn.init.normal_(final_conv.weight, mean=0.0, std=0.02)
        nn.init.zeros_(final_conv.bias)
        layers.append(final_conv)
        self.conv_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Q, d_spatial)
        B, Q, D = x.shape
        x = x.view(B * Q, D)
        x = self.fc(x).view(B * Q, -1, self.n_rf, self.n_m)
        x = self.conv_net(x)                            # (B*Q, 2, N_RF, N_M)
        x = x.view(B, Q, 2, self.n_rf, self.n_m).permute(0, 1, 3, 4, 2).contiguous()
        return x


if __name__ == "__main__":
    dec = SpatialDecoder()
    x = torch.randn(2, 5, 64)
    y = dec(x)
    assert y.shape == (2, 5, 8, 4, 2), f"got {y.shape}"
    n_params = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    print(f"SpatialDecoder OK | params: {n_params:,}")
