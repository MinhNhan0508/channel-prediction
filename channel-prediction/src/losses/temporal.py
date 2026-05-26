"""Temporal consistency loss: predicted channels should evolve at the correct rate.

If the true channel changes by Delta_true between t and t+1, the predicted channel
should change by approximately the same Delta. This encodes Doppler physics
implicitly: at higher velocity the channel changes faster, and the model must learn
this rate to keep this loss low.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape[1] < 2:
            return torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)

        pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
        true_diff = y_true[:, 1:] - y_true[:, :-1]

        B, Qm1 = pred_diff.shape[0], pred_diff.shape[1]
        pred_rate = pred_diff.reshape(B, Qm1, -1).norm(dim=-1)
        true_rate = true_diff.reshape(B, Qm1, -1).norm(dim=-1)

        return (pred_rate - true_rate).abs().mean()


if __name__ == "__main__":
    loss_fn = TemporalConsistencyLoss()
    a = torch.randn(2, 5, 8, 4, 2)
    b = a + 0.05 * torch.randn_like(a)
    v = loss_fn(b, a)
    print(f"temporal OK | value={v.item():.6f}")
