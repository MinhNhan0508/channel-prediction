"""Combined / pure-NMSE losses for channel prediction.

CombinedLoss = alpha*NMSE + beta*Phase + gamma*Temporal -- physics-informed.

NMSEOnlyLoss = the loss used by ChannelMamba (TWC 2026) and CPMamba (ArXiv
Dec 2025). When `pred` is near zero, atan2-based phase and `|norm(diff)|`
temporal losses inject gradient noise that traps the model in the pred=0
basin. Using NMSE alone removes those traps; phase and temporal can be
reintroduced for ablation later.
"""
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
        warmup_epochs: int = 0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.nmse_loss = NMSELoss(eps=eps)
        self.phase_loss = PhaseLoss(eps=eps)
        self.temporal_loss = TemporalConsistencyLoss()
        self.mse_loss = nn.MSELoss()
        # MSE trong `warmup_epochs` đầu để PHÁ đối xứng pred=0 (NMSE/phase bẫy
        # pred~0), rồi mới chuyển sang NMSE + phase + temporal. Trainer gọi
        # set_epoch ở mỗi epoch.
        self.warmup_epochs = int(warmup_epochs)
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        l_nmse = self.nmse_loss(y_pred, y_true)

        if y_pred.dim() >= 4 and y_pred.shape[-1] == 2:
            l_phase = self.phase_loss(y_pred, y_true)
        else:
            l_phase = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)

        l_temporal = self.temporal_loss(y_pred, y_true)

        if self.current_epoch < self.warmup_epochs:
            total = self.mse_loss(y_pred, y_true)             # warmup: MSE thoát pred=0
        else:
            total = self.alpha * l_nmse + self.beta * l_phase + self.gamma * l_temporal
        return {
            "total": total,
            "nmse": l_nmse.detach(),
            "phase": l_phase.detach(),
            "temporal": l_temporal.detach(),
        }


class NMSEOnlyLoss(nn.Module):
    """Pure NMSE -- matches ChannelMamba & CPMamba training recipes.

    Returns the same dict shape as CombinedLoss so the trainer's logging code
    works unchanged. `phase` and `temporal` are reported as zero.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.nmse_loss = NMSELoss(eps=eps)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        l = self.nmse_loss(y_pred, y_true)
        zero = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
        return {
            "total": l,
            "nmse": l.detach(),
            "phase": zero,
            "temporal": zero,
        }


class MSEWarmupNMSELoss(nn.Module):
    """MSE for the first `warmup_epochs`, then NMSE.

    Rationale (FIX.md exp1): NMSE has a degenerate basin at pred=0 with very
    weak gradient out. MSE has no such basin -- it pushes pred toward true
    even when pred is small. Warming up on MSE breaks the zero-symmetry, then
    switching to NMSE matches the evaluation metric and scales correctly across
    samples with different |H|.
    """

    def __init__(self, warmup_epochs: int = 10, eps: float = 1e-8):
        super().__init__()
        self.warmup_epochs = int(warmup_epochs)
        self.nmse_loss = NMSELoss(eps=eps)
        self.mse_loss = nn.MSELoss()
        # Mutable epoch counter set by the trainer at epoch boundaries.
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.current_epoch < self.warmup_epochs:
            l_total = self.mse_loss(y_pred, y_true)
        else:
            l_total = self.nmse_loss(y_pred, y_true)
        # Always report NMSE for monitoring (so val curves stay comparable).
        l_nmse = self.nmse_loss(y_pred, y_true).detach()
        zero = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
        return {
            "total": l_total,
            "nmse": l_nmse,
            "phase": zero,
            "temporal": zero,
        }


def build_loss(cfg: dict) -> nn.Module:
    """Factory: read `loss_type` (default 'nmse_only') and weight knobs from cfg."""
    loss_type = str(cfg.get("loss_type", "nmse_only")).lower()
    if loss_type == "nmse_only":
        return NMSEOnlyLoss()
    if loss_type == "mse_warmup":
        return MSEWarmupNMSELoss(warmup_epochs=int(cfg.get("mse_warmup_epochs", 10)))
    if loss_type == "combined":
        return CombinedLoss(
            alpha=float(cfg.get("alpha", 1.0)),
            beta=float(cfg.get("beta", 0.1)),
            gamma=float(cfg.get("gamma", 0.05)),
            warmup_epochs=int(cfg.get("phase_warmup_epochs", 0)),
        )
    raise ValueError(f"Unknown loss_type: {loss_type!r}")


if __name__ == "__main__":
    a = torch.randn(2, 5, 8, 4, 2)
    b = a + 0.05 * torch.randn_like(a)
    for cfg in ({"loss_type": "nmse_only"}, {"loss_type": "combined", "beta": 0.1, "gamma": 0.05}):
        ln = build_loss(cfg)
        d = ln(b, a)
        print(f"{cfg['loss_type']:11s} | total={d['total'].item():.6f} | "
              f"nmse={d['nmse'].item():.6f} | phase={d['phase'].item():.6f} | "
              f"temporal={d['temporal'].item():.6f}")
