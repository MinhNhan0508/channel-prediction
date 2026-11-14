"""Khối attention dùng chung: ProbSparse (Informer) + PE hình sin + encoder/decoder
layer. Dùng cho baseline `informer` và `hybrid_attention` (bản tái hiện IEICE).

ProbSparse: với chuỗi dài L, chọn top-u = ceil(factor·ln L) query "giàu thông tin
nhất" (metric thưa M(q)=max_j q·k_j − mean_j q·k_j) để attention đầy đủ; các query
còn lại lấy context = mean(V) (Informer/IEICE Eq 21-24). Ở L nhỏ (P=30/Q=5) ta tính
đủ score rồi mask cho đúng ngữ nghĩa.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_pe(length: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(length).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.factor = factor
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dk).transpose(1, 2)   # (B,h,L,dk)

    def forward(self, query, key, value):
        B, Lq, _ = query.shape
        Lk = key.shape[1]
        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)      # (B,h,Lq,Lk)

        # metric thưa + chọn top-u query
        u = max(1, min(Lq, int(math.ceil(self.factor * math.log(max(Lk, 2))))))
        M = scores.max(dim=-1).values - scores.mean(dim=-1)          # (B,h,Lq)
        topu = M.topk(u, dim=-1).indices                            # (B,h,u)
        sel = torch.zeros(B, self.h, Lq, dtype=torch.bool, device=query.device)
        sel.scatter_(-1, topu, True)

        attn = self.drop(F.softmax(scores, dim=-1))
        full_ctx = attn @ v                                         # (B,h,Lq,dk)
        mean_ctx = v.mean(dim=2, keepdim=True).expand(-1, -1, Lq, -1)
        ctx = torch.where(sel.unsqueeze(-1), full_ctx, mean_ctx)
        out = ctx.transpose(1, 2).reshape(B, Lq, self.h * self.dk)
        return self.out_proj(out)


class ProbEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout=0.1, factor=5):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads, factor, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(dim_ff, d_model))
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.n1(x + self.drop(self.attn(x, x, x)))
        return self.n2(x + self.drop(self.ff(x)))


class ProbDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout=0.1, factor=5):
        super().__init__()
        self.self_attn = ProbSparseAttention(d_model, n_heads, factor, dropout)
        self.cross_attn = ProbSparseAttention(d_model, n_heads, factor, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(dim_ff, d_model))
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d_model), nn.LayerNorm(d_model),
                                     nn.LayerNorm(d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory):
        x = self.n1(x + self.drop(self.self_attn(x, x, x)))
        x = self.n2(x + self.drop(self.cross_attn(x, memory, memory)))
        return self.n3(x + self.drop(self.ff(x)))
