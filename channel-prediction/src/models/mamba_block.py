"""Mamba block wrapper.

Tries the official `mamba-ssm` CUDA kernels first; on platforms where it cannot
build (notably Windows + GTX 1650), falls back to a clean pure-PyTorch
implementation of the Selective SSM (S6) from Gu & Dao 2024.

The fallback is functionally equivalent to the official implementation (same
state-space recurrence and selective parameterization) but ~5-10x slower because
the time-axis scan runs in Python. For P=30 timesteps this is still fast enough
for the GTX 1650 budget.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from mamba_ssm import Mamba as _OfficialMamba  # type: ignore
    MAMBA_BACKEND = "official"
except Exception:  # ImportError on Linux without CUDA, build errors on Windows
    _OfficialMamba = None
    MAMBA_BACKEND = "fallback"


class MambaBlockFallback(nn.Module):
    """Pure-PyTorch Selective SSM (S6) block.

    Same I/O contract as `mamba_ssm.Mamba`:
        forward(x: (B, L, d_model)) -> (B, L, d_model)

    Internal structure:
        in_proj    : Linear(d_model -> 2 * d_inner)         # produces (x, z)
        conv1d     : Causal Conv1d(d_inner, d_inner, k=d_conv, depthwise)
        x_proj     : Linear(d_inner -> dt_rank + 2*d_state) # produces (dt_raw, B, C)
        dt_proj    : Linear(dt_rank -> d_inner)
        A_log,D    : learnable SSM state matrix + skip
        out_proj   : Linear(d_inner -> d_model)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank if dt_rank is not None else max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # produces (dt_raw, B_param, C_param) from x_inner
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        # low-rank delta projection: dt_rank -> d_inner, with positive bias init
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        with torch.no_grad():
            # init dt_proj.bias so softplus(bias) is in a "reasonable" range
            dt_init_std = self.dt_rank ** -0.5
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            )
            inv_softplus_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_softplus_dt)

        # A: (d_inner, d_state), negative real values via -exp(A_log)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        # D: skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        B, L, _ = x.shape

        xz = self.in_proj(x)                            # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                   # each (B, L, d_inner)

        # Causal depthwise conv on time axis
        x_in_t = x_in.transpose(1, 2)                   # (B, d_inner, L)
        x_in_t = self.conv1d(x_in_t)[..., :L]           # crop right padding
        x_in = x_in_t.transpose(1, 2)                   # (B, L, d_inner)
        x_in = F.silu(x_in)

        # Compute selective parameters
        x_dbl = self.x_proj(x_in)                       # (B, L, dt_rank + 2*d_state)
        dt_raw, B_param, C_param = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt_raw))        # (B, L, d_inner) -- positive

        # A: (d_inner, d_state) -- negative
        A = -torch.exp(self.A_log.float())              # keep fp32 for stability
        # Discretize: A_disc = exp(delta * A), B_disc = delta * B
        # Shapes: delta (B,L,d_inner), A (d_inner, d_state)
        # We want A_disc (B, L, d_inner, d_state)
        delta_A = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)   # (B,L,d_inner,d_state)
        A_disc = torch.exp(delta_A)
        # B_disc: (B, L, d_inner, d_state)
        B_disc = delta.unsqueeze(-1) * B_param.unsqueeze(2)

        # Recurrent scan
        # h_t = A_disc_t * h_{t-1} + B_disc_t * x_t
        # y_t = sum over state of (C_t * h_t) + D * x_t
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=delta.dtype)
        ys = []
        for t in range(L):
            x_t = x_in[:, t]                            # (B, d_inner)
            h = A_disc[:, t] * h + B_disc[:, t] * x_t.unsqueeze(-1)   # (B, d_inner, d_state)
            # C_t: (B, d_state) -> (B, 1, d_state)
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(dim=-1)        # (B, d_inner)
            y_t = y_t + self.D * x_t
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                      # (B, L, d_inner)

        # Gating + output projection
        y = y * F.silu(z)
        return self.out_proj(y)


def make_mamba_block(
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
) -> nn.Module:
    """Return official Mamba if available, else the pure-PyTorch fallback."""
    if _OfficialMamba is not None:
        return _OfficialMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    return MambaBlockFallback(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


class BidirectionalMamba(nn.Module):
    """Bi-directional wrapper: forward + reversed Mamba, residual + LayerNorm.

    Inspired by ViM (Vision Mamba). Helps with non-causal temporal modeling --
    channels at intermediate frames can use information from both adjacent past
    and future frames.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.forward_mamba = make_mamba_block(d_model, d_state, d_conv, expand)
        self.backward_mamba = make_mamba_block(d_model, d_state, d_conv, expand)
        self.merge = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        x_fwd = self.forward_mamba(x)
        x_bwd = self.backward_mamba(torch.flip(x, dims=[1]))
        x_bwd = torch.flip(x_bwd, dims=[1])
        merged = self.merge(torch.cat([x_fwd, x_bwd], dim=-1))
        return self.norm(x + merged)


if __name__ == "__main__":
    print(f"[mamba_block] backend = {MAMBA_BACKEND}")
    block = BidirectionalMamba(d_model=64, d_state=16)
    x = torch.randn(2, 30, 64)
    y = block(x)
    assert y.shape == (2, 30, 64), f"got {y.shape}"
    n_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    print(f"BidirectionalMamba OK | params: {n_params:,}")
