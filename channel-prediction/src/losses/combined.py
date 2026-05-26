"""Physics-informed combined loss: L = alpha*NMSE + beta*Phase + gamma*Temporal."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from src.losses.nmse import NMSELoss
from src.losses.phase import PhaseLoss
from src.losses.temporal import TemporalConsistencyLoss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.nmse_loss = NMSELoss(eps=eps)
        self.phase_loss = PhaseLoss(eps=eps)
        self.temporal_loss = TemporalConsistencyLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        l_nmse = self.nmse_loss(y_pred, y_true)

        if y_pred.dim() >= 4 and y_pred.shape[-1] == 2:
            l_phase = self.phase_loss(y_pred, y_true)
        else:
            l_phase = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)

        l_temporal = self.temporal_loss(y_pred, y_true)

        total = self.alpha * l_nmse + self.beta * l_phase + self.gamma * l_temporal
        return {
            "total": total,
            "nmse": l_nmse.detach(),
            "phase": l_phase.detach(),
            "temporal": l_temporal.detach(),
        }


if __name__ == "__main__":
    loss_fn = CombinedLoss()
    a = torch.randn(2, 5, 8, 4, 2)
    b = a + 0.05 * torch.randn_like(a)
    d = loss_fn(b, a)
    print(f"combined OK | total={d['total'].item():.6f} | "
          f"nmse={d['nmse'].item():.6f} | phase={d['phase'].item():.6f} | "
          f"temporal={d['temporal'].item():.6f}")
