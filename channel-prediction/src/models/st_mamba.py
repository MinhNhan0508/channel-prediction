"""Spatial-Temporal Mamba (ST-Mamba) for MIMO Channel Prediction.

Architecture (PLAN section 6.5):
    (B, P, N_RF, N_M, 2)
        -> SpatialEncoder (CNN2D per timestep)
    (B, P, d_spatial)
        -> stack of BidirectionalMamba blocks
    (B, P, d_spatial)
        -> ParallelPredictionHead (Q learned queries, cross-attn over history)
    (B, Q, d_spatial)
        -> SpatialDecoder (CNN2D per timestep)
    (B, Q, N_RF, N_M, 2)
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src.models.mamba_block import BidirectionalMamba
from src.models.spatial_decoder import SpatialDecoder
from src.models.spatial_encoder import SpatialEncoder


class ParallelPredictionHead(nn.Module):
    """Predict Q future frames in one shot via cross-attention.

    Q learnable query tokens attend to the Mamba-encoded P history features.
    Same mechanism the IEICE-Transformer baseline uses, so the comparison
    isolates the temporal backbone (Mamba vs RNN/LSTM/GRU/Transformer).
    """

    def __init__(
        self,
        d_spatial: int,
        p_frames: int,
        q_frames: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_spatial = d_spatial
        self.p_frames = p_frames
        self.q_frames = q_frames

        self.queries = nn.Parameter(torch.randn(q_frames, d_spatial) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_spatial,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_spatial)
        self.norm2 = nn.LayerNorm(d_spatial)
        self.ffn = nn.Sequential(
            nn.Linear(d_spatial, d_spatial * 2),
            nn.GELU(),
            nn.Linear(d_spatial * 2, d_spatial),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        # encoded: (B, P, d_spatial)
        B = encoded.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(query=q, key=encoded, value=encoded)
        q = self.norm1(q + attn_out)
        q = self.norm2(q + self.ffn(q))
        return q


class STMamba(nn.Module):
    """ST-Mamba: Spatial-Temporal decoupled channel predictor."""

    def __init__(
        self,
        n_rf: int = 8,
        n_m: int = 4,
        p_frames: int = 30,
        q_frames: int = 5,
        d_spatial: int = 64,
        n_mamba_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        cnn_encoder_channels: Sequence[int] = (16, 32, 64),
        cnn_decoder_channels: Sequence[int] = (64, 32, 16),
        n_pred_heads: int = 4,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.config = {
            "n_rf": n_rf, "n_m": n_m,
            "p_frames": p_frames, "q_frames": q_frames,
            "d_spatial": d_spatial,
            "n_mamba_layers": n_mamba_layers,
        }

        self.spatial_encoder = SpatialEncoder(
            n_rf=n_rf, n_m=n_m, d_spatial=d_spatial,
            cnn_channels=tuple(cnn_encoder_channels),
            use_batchnorm=use_batchnorm,
        )

        self.mamba_blocks = nn.ModuleList(
            [
                BidirectionalMamba(
                    d_model=d_spatial,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_mamba_layers)
            ]
        )

        self.pred_head = ParallelPredictionHead(
            d_spatial=d_spatial,
            p_frames=p_frames,
            q_frames=q_frames,
            n_heads=n_pred_heads,
        )

        self.spatial_decoder = SpatialDecoder(
            n_rf=n_rf, n_m=n_m, d_spatial=d_spatial,
            hidden_channels=tuple(cnn_decoder_channels),
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, N_RF, N_M, 2)
        h = self.spatial_encoder(x)                 # (B, P, d_spatial)
        for block in self.mamba_blocks:
            h = block(h)
        future = self.pred_head(h)                  # (B, Q, d_spatial)
        return self.spatial_decoder(future)         # (B, Q, N_RF, N_M, 2)

    def count_params(self) -> dict:
        return {
            "spatial_encoder": sum(p.numel() for p in self.spatial_encoder.parameters()),
            "mamba_blocks":    sum(p.numel() for p in self.mamba_blocks.parameters()),
            "pred_head":       sum(p.numel() for p in self.pred_head.parameters()),
            "spatial_decoder": sum(p.numel() for p in self.spatial_decoder.parameters()),
            "total":           sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


if __name__ == "__main__":
    model = STMamba()
    x = torch.randn(2, 30, 8, 4, 2)
    y = model(x)
    assert y.shape == (2, 5, 8, 4, 2), f"got {y.shape}"
    counts = model.count_params()
    print("STMamba OK")
    for k, v in counts.items():
        print(f"  {k:18s} : {v:,}")
