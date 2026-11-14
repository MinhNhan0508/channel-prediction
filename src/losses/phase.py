"""Magnitude-weighted phase loss for complex channels stored as (..., 2)."""
from __future__ import annotations

import torch
import torch.nn as nn


class PhaseLoss(nn.Module):
    """Circular phase error weighted by ground-truth magnitude.

    Phase has physical meaning for beamforming (precoders depend on phase),
    so penalize phase error -- but weight by magnitude so that near-zero
    entries (where phase is numerically undefined) do not dominate the loss.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred, y_true: (..., 2) real/imag
        phase_pred = torch.atan2(y_pred[..., 1], y_pred[..., 0])
        phase_true = torch.atan2(y_true[..., 1], y_true[..., 0])

        diff = phase_pred - phase_true
        # wrap to (-pi, pi]
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))

        magnitude = torch.sqrt(
            y_true[..., 0] ** 2 + y_true[..., 1] ** 2 + self.eps
        )
        # Normalize so the loss is roughly scale-invariant w.r.t. channel magnitude
        weighted = (diff.abs() * magnitude).mean() / (magnitude.mean() + self.eps)
        return weighted


if __name__ == "__main__":
    loss_fn = PhaseLoss()
    a = torch.randn(2, 5, 8, 4, 2)
    b = a + 0.05 * torch.randn_like(a)
    v = loss_fn(b, a)
    print(f"phase OK | value={v.item():.6f}")
