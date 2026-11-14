"""AdamW with two LR schedules:
  - 'cosine'    : linear warmup -> cosine annealing to 0 (default)
  - 'two_stage' : warmup -> hold lr for stage1_ratio of training -> drop to
                  stage2_lr_factor * lr for the rest (matches CPMamba recipe)
"""
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

    sched_type = str(config.get("lr_schedule", "cosine")).lower()

    if sched_type == "two_stage":
        stage1_ratio = float(config.get("stage1_ratio", 2.0 / 3.0))
        stage2_lr_factor = float(config.get("stage2_lr_factor", 0.1))
        stage1_end = max(warmup_steps, int(total_steps * stage1_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            if step < stage1_end:
                return 1.0
            return stage2_lr_factor

    else:  # cosine
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler
