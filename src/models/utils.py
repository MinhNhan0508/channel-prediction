"""Model factory + parameter counting helpers."""
from __future__ import annotations

import torch.nn as nn

from src.models.baselines.cnn import CNNBaseline
from src.models.baselines.gru import GRUBaseline
from src.models.baselines.lstm import LSTMBaseline
from src.models.baselines.rnn import RNNBaseline
from src.models.baselines.transformer import TransformerBaseline
from src.models.pi_dast import PIDAST


_MODEL_REGISTRY = {
    "pi_dast": PIDAST,            # ĐỀ XUẤT
    "rnn": RNNBaseline,
    "lstm": LSTMBaseline,
    "gru": GRUBaseline,
    "cnn": CNNBaseline,
    "transformer": TransformerBaseline,
}

# Model nhận input GIỮ chiều không gian (B,P,N_RF,N_M,2) -- KHÔNG flatten.
# Các model còn lại nhận (B,P,64). Dùng để set flatten_input trong train/eval.
SPATIAL_MODELS = {"pi_dast", "cnn"}


def is_spatial_model(name: str) -> bool:
    return name.lower() in SPATIAL_MODELS


def build_model(model_name: str, config: dict) -> nn.Module:
    name = model_name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(_MODEL_REGISTRY)}")
    cls = _MODEL_REGISTRY[name]
    kwargs = {k: v for k, v in config.items() if k != "model_name"}
    return cls(**kwargs)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
