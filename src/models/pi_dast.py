"""Model ĐỀ XUẤT. Tên hiển thị: **DAST** (Doppler-Adaptive Sequence Transformer).
Khoá nội bộ / checkpoint / registry vẫn là `pi_dast` (giữ nguyên để không vỡ
pipeline). Tên cũ (lỗi thời, spatial+physics đã tắt): PI-DAST / Doppler-Aware
Spatial-Temporal.

    (B,P,N_RF,N_M,2)
      -> flatten + Linear (beam×antenna -> d_model)
      -> DopplerEncoder  (FiLM theo phổ Doppler ước lượng)    [use_doppler_enc]
      -> Transformer encoder-decoder (temporal, parallel P->Q)
      -> Linear -> (B,Q,N_RF,N_M,2)  [NN_pred]
      -> HYBRID: fuse PAD prior cổ điển qua cổng Doppler      [use_pad_prior/use_gate]

3 toggle (use_doppler_enc / use_pad_prior / use_gate) cho ablation chứng minh từng trụ.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.attention import sinusoidal_pe
from src.models.pad_torch import pad_extrapolate


class DopplerEncoder(nn.Module):
    """Ước lượng đặc trưng Doppler per-sample từ phổ thời gian của chuỗi đặc trưng,
    rồi FiLM-modulate (scale+shift). Đánh trúng lỗ hổng 'Transformer không phân
    biệt tốc độ user'. h: (B,P,d) -> (B,P,d)."""

    def __init__(self, d_model: int, p_frames: int):
        super().__init__()
        n_freq = p_frames // 2 + 1 - 1                    # bỏ DC (bin 0)
        self.mlp = nn.Sequential(
            nn.Linear(n_freq, d_model), nn.GELU(),
            nn.Linear(d_model, 2 * d_model))
        # init nhỏ -> ban đầu gần identity (gamma≈0, beta≈0)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        spec = torch.fft.rfft(h, dim=1).abs()             # (B, P//2+1, d)
        prof = spec[:, 1:].mean(dim=-1)                   # (B, n_freq) -- bỏ DC, gộp feature
        gamma, beta = self.mlp(prof).chunk(2, dim=-1)     # mỗi (B,d)
        return (1.0 + gamma).unsqueeze(1) * h + beta.unsqueeze(1)


class PIDAST(nn.Module):
    def __init__(
        self,
        n_rf: int = 8,
        n_m: int = 4,
        p_frames: int = 30,
        q_frames: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        use_doppler_enc: bool = True,    # toggle ablation: bật/tắt Doppler-aware encoding
        use_pad_prior: bool = False,     # HYBRID: fuse PAD cổ điển với NN (gated residual)
        use_gate: bool = True,           # True = cổng theo Doppler; False = blend cố định (ablation)
        **_unused,                       # nuốt key cũ trong config (cnn_*_channels, use_spatial, use_batchnorm)
    ):
        super().__init__()
        self.n_rf, self.n_m = n_rf, n_m
        self.p_frames, self.q_frames = p_frames, q_frames
        self.d_model = d_model
        self.use_doppler_enc = use_doppler_enc
        self.use_pad_prior = use_pad_prior
        self.use_gate = use_gate
        in_dim = n_rf * n_m * 2

        # --- Front/back: flatten beam×antenna <-> d_model (spatial-CNN đã loại) ---
        self.encoder_spatial = nn.Linear(in_dim, d_model)
        self.decoder_spatial = nn.Linear(d_model, in_dim)

        # --- Doppler-aware encoding ---
        if use_doppler_enc:
            self.doppler = DopplerEncoder(d_model, p_frames)

        # --- Temporal Transformer (chuẩn, full-attention) ---
        self.register_buffer("pe_enc", sinusoidal_pe(p_frames, d_model))
        self.register_buffer("pe_dec", sinusoidal_pe(q_frames, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward, dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward, dropout, activation="gelu", batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)
        self.queries = nn.Parameter(torch.randn(q_frames, d_model) * 0.02)

        # --- Doppler-aware gate cho PAD prior (HYBRID) ---
        if use_pad_prior:
            n_freq = p_frames // 2 + 1 - 1                 # bỏ DC, khớp DopplerEncoder
            if use_gate:
                self.gate = nn.Sequential(
                    nn.Linear(n_freq, d_model // 2), nn.GELU(),
                    nn.Linear(d_model // 2, q_frames))
                nn.init.zeros_(self.gate[-1].weight)       # khởi tạo g=0.5 (blend trung tính)
                nn.init.zeros_(self.gate[-1].bias)
            else:
                self.gate_logit = nn.Parameter(torch.zeros(q_frames))  # blend cố định học được

        self.config = {"n_rf": n_rf, "n_m": n_m, "p_frames": p_frames, "q_frames": q_frames,
                       "use_doppler_enc": use_doppler_enc,
                       "use_pad_prior": use_pad_prior, "use_gate": use_gate}

    def _encode_spatial(self, x):
        B, P = x.shape[0], x.shape[1]
        return self.encoder_spatial(x.reshape(B, P, -1))             # (B,P,d)

    def _decode_spatial(self, h):
        B, Q = h.shape[0], h.shape[1]
        return self.decoder_spatial(h).view(B, Q, self.n_rf, self.n_m, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,P,N_RF,N_M,2)
        h = self._encode_spatial(x)                       # (B,P,d)
        if self.use_pad_prior and self.use_gate:
            # phổ Doppler từ đặc trưng (trước FiLM) -> điều khiển cổng
            prof = torch.fft.rfft(h, dim=1).abs()[:, 1:].mean(dim=-1)   # (B, n_freq)
        if self.use_doppler_enc:
            h = self.doppler(h)
        h = h + self.pe_enc.unsqueeze(0)
        memory = self.encoder(h)
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1) + self.pe_dec.unsqueeze(0)
        out = self.decoder(tgt=q, memory=memory)          # (B,Q,d)
        nn_pred = self._decode_spatial(out)               # (B,Q,N_RF,N_M,2)
        if not self.use_pad_prior:
            return nn_pred
        # HYBRID: fuse PAD prior (cổ điển) với NN qua cổng Doppler
        prior = pad_extrapolate(x, q_frames=self.q_frames)            # (B,Q,N_RF,N_M,2)
        if self.use_gate:
            g = torch.sigmoid(self.gate(prof))                        # (B,Q) phụ thuộc Doppler
        else:
            g = torch.sigmoid(self.gate_logit).unsqueeze(0).expand(x.shape[0], -1)
        g = g[:, :, None, None, None]                                 # broadcast (B,Q,1,1,1)
        return g * prior + (1.0 - g) * nn_pred

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfgs = [
        dict(use_doppler_enc=True, use_pad_prior=False),                 # pure NN DAST
        dict(use_doppler_enc=True, use_pad_prior=True, use_gate=True),    # HYBRID (đề xuất)
        dict(use_doppler_enc=True, use_pad_prior=True, use_gate=False),   # fixed-blend ablation
        dict(use_doppler_enc=False, use_pad_prior=False),                 # w/o doppler
    ]
    for c in cfgs:
        m = PIDAST(p_frames=10, **c)
        x = torch.randn(2, 10, 8, 4, 2)
        y = m(x)
        assert y.shape == (2, 5, 8, 4, 2), f"got {y.shape}"
        print(f"PIDAST({c}) OK | params: {m.count_params():,}")
