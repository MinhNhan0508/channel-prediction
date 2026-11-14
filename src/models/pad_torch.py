"""PAD prior trong torch -- port khả vi của src/models/pad.py (AR-LS extrapolation).

Dùng làm classical prior parameter-free bên trong DAST-hybrid (gated residual):
    pred = g(Doppler) * PAD(history) + (1-g) * NN(history)
PAD chạy dưới no_grad (prior là hàm thuần của input, không cần gradient). PAD
scale-equivariant nên chạy thẳng trên input đã chuẩn hoá = prior đã chuẩn hoá.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def pad_extrapolate(x: torch.Tensor, q_frames: int = 5, order: int = 4,
                    ridge: float = 1e-3, clamp_scale: float = 3.0) -> torch.Tensor:
    """x: (B,P,n_rf,n_m,2) real -> (B,Q,n_rf,n_m,2). AR(p) least-squares trên từng
    chuỗi hệ số phức, ngoại suy đệ quy Q bước. clamp_scale chặn nghiệm AR bùng nổ."""
    B, P, R, M, _ = x.shape
    z = torch.complex(x[..., 0], x[..., 1])                  # (B,P,R,M)
    z = z.reshape(B, P, R * M).permute(0, 2, 1).reshape(B * R * M, P)  # (S,P)
    p = int(min(order, P - 1))
    if p < 1:
        return x[:, -1:].expand(B, q_frames, R, M, 2).contiguous()

    Mn = P - p
    A = torch.stack([z[:, (p - 1 - k):(p - 1 - k) + Mn] for k in range(p)], dim=-1)  # (S,Mn,p)
    y = z[:, p:p + Mn]                                       # (S,Mn)
    Ah = A.conj().transpose(1, 2)                            # (S,p,Mn)
    AhA = Ah @ A
    Ahy = Ah @ y.unsqueeze(-1)
    eye = torch.eye(p, dtype=AhA.dtype, device=x.device).unsqueeze(0)
    reg = ridge * AhA.abs().mean(dim=(1, 2), keepdim=True) * eye
    coeff = torch.linalg.solve(AhA + reg, Ahy)[..., 0]       # (S,p)
    bad = ~torch.isfinite(coeff).all(dim=1)
    if bad.any():
        coeff[bad] = 0.0
        coeff[bad, 0] = 1.0

    lim = clamp_scale * z.abs().amax(dim=1)                  # (S,)
    hist = z[:, -p:].clone()                                 # (S,p)
    preds = []
    for _ in range(q_frames):
        nxt = (coeff * hist.flip(1)).sum(dim=1)              # a1*x[t-1]+...+ap*x[t-p]
        nxt = torch.where(torch.isfinite(nxt), nxt, hist[:, -1])
        mag = nxt.abs()
        scale = torch.where(mag > lim, lim / (mag + 1e-12), torch.ones_like(mag))
        nxt = nxt * scale
        preds.append(nxt)
        hist = torch.cat([hist[:, 1:], nxt[:, None]], dim=1)
    pr = torch.stack(preds, dim=1)                           # (S,Q)
    pr = pr.reshape(B, R * M, q_frames).permute(0, 2, 1).reshape(B, q_frames, R, M)
    return torch.stack([pr.real, pr.imag], dim=-1).to(x.dtype)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.models.pad import pad_predict
    x = np.random.default_rng(0).standard_normal((4, 10, 8, 4, 2)).astype(np.float32)
    a = pad_predict(x, q_frames=5)
    b = pad_extrapolate(torch.from_numpy(x), q_frames=5, clamp_scale=1e9).numpy()
    print("max |numpy - torch| (no clamp) =", float(np.abs(a - b).max()))
