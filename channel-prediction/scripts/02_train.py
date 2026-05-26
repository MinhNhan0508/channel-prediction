"""Train one model.

Usage:
    python scripts/02_train.py --model st_mamba --train_config full
    python scripts/02_train.py --model lstm     --train_config full
    python scripts/02_train.py --model st_mamba --train_config curriculum
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ChannelDataset  # noqa: E402
from src.data.normalize import ChannelNormalizer  # noqa: E402
from src.losses.combined import CombinedLoss  # noqa: E402
from src.models.utils import build_model, count_trainable_params  # noqa: E402
from src.training.curriculum import CurriculumScheduler  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.config import load_config, merge_configs  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


PARAM_BUDGET_MIN = 280_000
PARAM_BUDGET_MAX = 420_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["st_mamba", "rnn", "lstm", "gru", "transformer"],
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="full",
        choices=["full", "curriculum", "debug"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    set_seed(int(args.seed))

    model_cfg = load_config(f"configs/model/{args.model}.yaml")
    train_cfg = load_config(f"configs/train/{args.train_config}.yaml")
    cfg = merge_configs(model_cfg, train_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}", flush=True)

    data_dir = Path(args.data_dir)
    normalizer = ChannelNormalizer.load(data_dir / "normalize_stats.npz")

    flatten = args.model != "st_mamba"

    train_ds = ChannelDataset(data_dir / "train", normalizer=normalizer, flatten_spatial=flatten)
    val_ds = ChannelDataset(data_dir / "val", normalizer=normalizer, flatten_spatial=flatten)

    # Optional curriculum
    if args.train_config == "curriculum":
        stages = [(int(end), tuple(rng)) for end, rng in train_cfg["curriculum_stages"]]
        # Use the full stage that contains all velocities for the loader; the actual
        # curriculum subsetting is left as a per-epoch hook (omitted in this minimal
        # path -- training runs on full train set with the warmup/cosine schedule).
        scheduler = CurriculumScheduler(train_ds, stages)
        train_subset = scheduler.get_subset(int(train_cfg["n_epochs"]))
        train_loader = DataLoader(
            train_subset,
            batch_size=int(cfg["batch_size"]),
            shuffle=True,
            num_workers=int(cfg.get("num_workers", 2)),
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["batch_size"]),
            shuffle=True,
            num_workers=int(cfg.get("num_workers", 2)),
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
        pin_memory=True,
    )

    model = build_model(args.model, model_cfg)
    n_params = count_trainable_params(model)
    print(f"[train] model={args.model} params={n_params:,}", flush=True)
    if not (PARAM_BUDGET_MIN <= n_params <= PARAM_BUDGET_MAX):
        print(
            f"[WARN] params {n_params:,} outside fair budget "
            f"[{PARAM_BUDGET_MIN:,}, {PARAM_BUDGET_MAX:,}] -- "
            f"adjust configs/model/{args.model}.yaml",
            flush=True,
        )

    loss_fn = CombinedLoss(
        alpha=float(cfg.get("alpha", 1.0)),
        beta=float(cfg.get("beta", 0.1)),
        gamma=float(cfg.get("gamma", 0.05)),
    )

    exp_name = args.exp_name or f"{args.model}_{args.train_config}_seed{args.seed}"
    log_dir = Path("results/logs") / exp_name
    ckpt_dir = Path("results/checkpoints") / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        config=cfg,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        flatten_input=flatten,
    )
    best_val = trainer.train()
    print(f"[train] done. best val total_loss = {best_val:.6f}", flush=True)


if __name__ == "__main__":
    main()
