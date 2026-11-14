"""NMSE loss for channel prediction."""
from __future__ import annotations

import torch
import torch.nn as nn


class NMSELoss(nn.Module):
    """E[||y_pred - y_true||^2] / E[||y_true||^2], averaged over batch & Q frames."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        B, Q = y_pred.shape[0], y_pred.shape[1]
        y_pred_flat = y_pred.reshape(B, Q, -1)
        y_true_flat = y_true.reshape(B, Q, -1)
        numerator = torch.sum((y_pred_flat - y_true_flat) ** 2, dim=-1)
        denominator = torch.sum(y_true_flat ** 2, dim=-1) + self.eps
        return (numerator / denominator).mean()


def nmse_db(nmse: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(nmse.clamp(min=1e-12))


if __name__ == "__main__":
    loss_fn = NMSELoss()
    a = torch.randn(2, 5, 8, 4, 2)
    b = a + 0.1 * torch.randn_like(a)
    v = loss_fn(b, a)
    print(f"nmse OK | value={v.item():.6f} | dB={nmse_db(v).item():.2f}")
