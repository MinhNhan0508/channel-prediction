"""Model factory + parameter counting helpers."""
from __future__ import annotations

import torch.nn as nn

from src.models.baselines.gru import GRUBaseline
from src.models.baselines.lstm import LSTMBaseline
from src.models.baselines.rnn import RNNBaseline
from src.models.baselines.transformer import TransformerBaseline
from src.models.st_mamba import STMamba


_MODEL_REGISTRY = {
    "st_mamba": STMamba,
    "rnn": RNNBaseline,
    "lstm": LSTMBaseline,
    "gru": GRUBaseline,
    "transformer": TransformerBaseline,
}


def build_model(model_name: str, config: dict) -> nn.Module:
    name = model_name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(_MODEL_REGISTRY)}")
    cls = _MODEL_REGISTRY[name]
    kwargs = {k: v for k, v in config.items() if k != "model_name"}
    return cls(**kwargs)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
