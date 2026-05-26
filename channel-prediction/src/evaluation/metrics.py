"""Evaluation metrics: per-frame NMSE and zero-forcing sum-rate."""
from __future__ import annotations

from typing import Dict

import torch


def compute_nmse_per_frame(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Args:
        pred, true: (B, Q, ...)  reshapable to (B, Q, D)
    Returns:
        (Q,) NMSE per future frame, averaged over the batch.
    """
    B, Q = pred.shape[0], pred.shape[1]
    pred_flat = pred.reshape(B, Q, -1)
    true_flat = true.reshape(B, Q, -1)
    num = (pred_flat - true_flat).pow(2).sum(dim=-1)
    den = true_flat.pow(2).sum(dim=-1).clamp(min=1e-12)
    return (num / den).mean(dim=0)


def nmse_to_db(nmse: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(nmse.clamp(min=1e-12))


def compute_sum_rate(
    He_pred: torch.Tensor,
    He_true: torch.Tensor,
    snr_db: float = 20.0,
) -> Dict[str, torch.Tensor]:
    """Achievable sum-rate with ZF precoder using predicted CSI vs ideal CSI.

    Args:
        He_pred, He_true: (B, Q, N_RF, N_M, 2) real/imag effective channels.

    Returns:
        dict with keys 'sum_rate_predicted' and 'sum_rate_ideal' (mean scalars).
    """
    assert He_pred.shape == He_true.shape and He_pred.shape[-1] == 2

    He_pred_c = torch.complex(He_pred[..., 0], He_pred[..., 1])
    He_true_c = torch.complex(He_true[..., 0], He_true[..., 1])

    B, Q, N_RF, N_M = He_pred_c.shape
    sigma2 = 10.0 ** (-snr_db / 10.0)
    eye_rf = torch.eye(N_RF, dtype=He_pred_c.dtype, device=He_pred.device)

    rates_pred = []
    rates_ideal = []

    for q in range(Q):
        Hp_q = He_pred_c[:, q]  # (B, N_RF, N_M)
        Ht_q = He_true_c[:, q]
        for b in range(B):
            Hp = Hp_q[b]   # (N_RF, N_M)
            Ht = Ht_q[b]

            # ZF precoder from predicted CSI: A_p = Hp^H (Hp Hp^H)^-1   (right inverse)
            HpHp = Hp @ Hp.conj().transpose(-2, -1)
            try:
                A_p = Hp.conj().transpose(-2, -1) @ torch.linalg.inv(
                    HpHp + 1e-6 * eye_rf
                )
            except Exception:
                continue
            A_p = A_p / (torch.linalg.matrix_norm(A_p) + 1e-12)

            H_eff_p = Ht @ A_p  # (N_RF, N_RF)
            arg_p = eye_rf + (1.0 / max(N_M * sigma2, 1e-12)) * (
                H_eff_p @ H_eff_p.conj().transpose(-2, -1)
            )
            sr_p = torch.log2(torch.linalg.det(arg_p).real.clamp(min=1e-12))
            rates_pred.append(sr_p)

            # Ideal CSI ZF precoder
            HtHt = Ht @ Ht.conj().transpose(-2, -1)
            try:
                A_t = Ht.conj().transpose(-2, -1) @ torch.linalg.inv(
                    HtHt + 1e-6 * eye_rf
                )
            except Exception:
                continue
            A_t = A_t / (torch.linalg.matrix_norm(A_t) + 1e-12)
            H_eff_t = Ht @ A_t
            arg_t = eye_rf + (1.0 / max(N_M * sigma2, 1e-12)) * (
                H_eff_t @ H_eff_t.conj().transpose(-2, -1)
            )
            sr_t = torch.log2(torch.linalg.det(arg_t).real.clamp(min=1e-12))
            rates_ideal.append(sr_t)

    return {
        "sum_rate_predicted": torch.stack(rates_pred).mean() if rates_pred else torch.zeros(()),
        "sum_rate_ideal": torch.stack(rates_ideal).mean() if rates_ideal else torch.zeros(()),
    }
