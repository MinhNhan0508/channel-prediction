"""Transformer baseline modeled after IEICE 2026 "Hybrid-Attention Parallel
Synchronous Transformer" for channel prediction.

This is the encoder-decoder baseline. Q learned query tokens go through the
decoder and cross-attend to the encoded P history. No autoregression.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        p_frames: int = 30,
        q_frames: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.p_frames = p_frames
        self.q_frames = q_frames

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc_encoder = nn.Parameter(torch.randn(p_frames, d_model) * 0.02)
        self.pos_enc_decoder = nn.Parameter(torch.randn(q_frames, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.queries = nn.Parameter(torch.randn(q_frames, d_model) * 0.02)
        self.output_proj = nn.Linear(d_model, input_dim)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, input_dim)
        B = x.shape[0]
        x = self.input_proj(x) + self.pos_enc_encoder.unsqueeze(0)
        memory = self.encoder(x)
        q = self.queries.unsqueeze(0).expand(B, -1, -1) + self.pos_enc_decoder.unsqueeze(0)
        out = self.decoder(tgt=q, memory=memory)
        return self.output_proj(out)


if __name__ == "__main__":
    model = TransformerBaseline()
    x = torch.randn(2, 30, 64)
    y = model(x)
    assert y.shape == (2, 5, 64), f"got {y.shape}"
    print(f"Transformer OK | params: {sum(p.numel() for p in model.parameters()):,}")
