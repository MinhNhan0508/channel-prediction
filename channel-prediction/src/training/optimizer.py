"""AdamW + linear warmup + cosine annealing."""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: dict,
    steps_per_epoch: int,
) -> Tuple[AdamW, LambdaLR]:
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        betas=(0.9, 0.95),
    )

    total_steps = max(1, int(config["n_epochs"]) * steps_per_epoch)
    warmup_steps = max(1, int(float(config.get("warmup_ratio", 0.05)) * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler
