"""Ablation studies for ST-Mamba.

Each ablation modifies ONE component:
    A1 (no_spatial)   : replace SpatialEncoder with a plain Linear (preserve params budget)
    A2 (no_mamba)     : replace Mamba with LSTM at same hidden size (= d_spatial)
    A3 (no_phys_loss) : keep arch, train with beta=gamma=0
    A4 (no_curriculum): train without curriculum schedule (already the default)

A3/A4 reuse the existing checkpoint pipeline by passing override configs.
A1/A2 require minor model variants -- defined inline here.

Usage:
    python scripts/05_ablation.py --run a1     # train + eval ablation A1
    python scripts/05_ablation.py --run all    # all 4 ablations
    python scripts/05_ablation.py --report     # compile summary table from JSON files
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baselines.lstm import LSTMBaseline  # noqa: E402
from src.models.spatial_decoder import SpatialDecoder  # noqa: E402
from src.models.spatial_encoder import SpatialEncoder  # noqa: E402
from src.models.st_mamba import ParallelPredictionHead, STMamba  # noqa: E402


ABLATIONS = {
    "a1": {
        "name": "st_mamba_no_spatial",
        "description": "Replace 2D-CNN spatial encoder/decoder with linear projections",
        "kind": "model_variant",
    },
    "a2": {
        "name": "st_mamba_no_mamba",
        "description": "Replace Mamba temporal backbone with LSTM (same hidden)",
        "kind": "model_variant",
    },
    "a3": {
        "name": "st_mamba_no_phys_loss",
        "description": "NMSE-only loss (beta=gamma=0)",
        "kind": "train_only",
        "loss": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
    },
    "a4": {
        "name": "st_mamba_no_curriculum",
        "description": "Train ST-Mamba without curriculum (equivalent to --train_config full)",
        "kind": "train_only",
        "train_config": "full",
    },
}


class STMambaNoSpatial(STMamba):
    """A1: replace 2D-CNN encoder/decoder with linear projections."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        in_dim = self.config["n_rf"] * self.config["n_m"] * 2
        d_spatial = self.config["d_spatial"]
        # Override encoder/decoder with linear-only equivalents
        self.spatial_encoder = _LinearEncoder(in_dim, d_spatial)
        self.spatial_decoder = _LinearDecoder(d_spatial, self.config["n_rf"], self.config["n_m"])


class _LinearEncoder(nn.Module):
    def __init__(self, in_dim: int, d_spatial: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_spatial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, P = x.shape[0], x.shape[1]
        return self.proj(x.reshape(B, P, -1))


class _LinearDecoder(nn.Module):
    def __init__(self, d_spatial: int, n_rf: int, n_m: int):
        super().__init__()
        self.n_rf = n_rf
        self.n_m = n_m
        self.proj = nn.Linear(d_spatial, n_rf * n_m * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Q = x.shape[0], x.shape[1]
        out = self.proj(x).view(B, Q, self.n_rf, self.n_m, 2)
        return out


class STMambaNoMamba(nn.Module):
    """A2: replace Mamba blocks with LSTM at d_spatial.

    Keeps SpatialEncoder/Decoder/PredHead identical to ST-Mamba.
    """

    def __init__(
        self,
        n_rf: int = 8,
        n_m: int = 4,
        p_frames: int = 30,
        q_frames: int = 5,
        d_spatial: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        cnn_encoder_channels=(16, 32, 64),
        cnn_decoder_channels=(48, 32, 16),
        n_pred_heads: int = 4,
        use_batchnorm: bool = True,
        # accept unused mamba kwargs for config compatibility
        **_unused,
    ):
        super().__init__()
        self.config = {"n_rf": n_rf, "n_m": n_m, "p_frames": p_frames, "q_frames": q_frames}
        self.spatial_encoder = SpatialEncoder(
            n_rf=n_rf, n_m=n_m, d_spatial=d_spatial,
            cnn_channels=tuple(cnn_encoder_channels), use_batchnorm=use_batchnorm,
        )
        self.lstm = nn.LSTM(
            input_size=d_spatial, hidden_size=d_spatial,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.pred_head = ParallelPredictionHead(
            d_spatial=d_spatial, p_frames=p_frames, q_frames=q_frames, n_heads=n_pred_heads
        )
        self.spatial_decoder = SpatialDecoder(
            n_rf=n_rf, n_m=n_m, d_spatial=d_spatial,
            hidden_channels=tuple(cnn_decoder_channels), use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.spatial_encoder(x)
        h, _ = self.lstm(h)
        f = self.pred_head(h)
        return self.spatial_decoder(f)


def _run_subprocess(cmd: list[str]) -> int:
    print(f"[ablation] running: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def run_ablation(key: str):
    cfg = ABLATIONS[key]
    name = cfg["name"]
    print(f"[ablation] === {key}: {cfg['description']} ===", flush=True)

    if cfg["kind"] == "train_only":
        # A3: NMSE-only loss
        if key == "a3":
            # Patch train config in-place: we re-run with --train_config full but
            # override beta/gamma via a temporary yaml in the exp name (PLAN style
            # is to edit the YAML; for ablation we just rely on the user adjusting
            # configs/train/full.yaml or run with phys-loss zeros baked in).
            # Simplest: invoke 02_train with a one-shot env variable approach is
            # out of scope; instead user copies configs/train/full.yaml to
            # configs/train/ablation_a3.yaml with beta=gamma=0 then trains:
            train_cfg = "ablation_a3"
            print(
                f"[ablation] please ensure configs/train/{train_cfg}.yaml exists "
                f"with alpha=1.0, beta=0.0, gamma=0.0 before running.",
                flush=True,
            )
            _run_subprocess([
                sys.executable, "scripts/02_train.py",
                "--model", "st_mamba",
                "--train_config", train_cfg,
                "--exp_name", name,
            ])
        elif key == "a4":
            _run_subprocess([
                sys.executable, "scripts/02_train.py",
                "--model", "st_mamba",
                "--train_config", "full",
                "--exp_name", name,
            ])

        # eval on both scenarios
        for sc in ("v30", "v50"):
            _run_subprocess([
                sys.executable, "scripts/03_evaluate.py",
                "--model", "st_mamba",
                "--exp", name,
                "--test_scenario", sc,
            ])

    elif cfg["kind"] == "model_variant":
        # A1, A2: requires a tweaked model. We side-load it by monkey-patching the
        # model factory: register the variant under the same name 'st_mamba' for
        # the run, then call the standard training script.
        # Simpler: build, train, and evaluate inline here.
        print(
            f"[ablation] {key} ({name}) requires a custom training run via the "
            f"inline classes (STMambaNoSpatial / STMambaNoMamba) -- see code.",
            flush=True,
        )
        # We expose the classes via module attributes; full inline training is
        # left as a thin wrapper around Trainer below.
        _train_eval_variant(key, name)


def _train_eval_variant(key: str, name: str):
    """Inline train+eval for A1/A2 variants."""
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import ChannelDataset
    from src.data.normalize import ChannelNormalizer
    from src.losses.combined import CombinedLoss
    from src.training.trainer import Trainer
    from src.utils.config import load_config, merge_configs
    from src.utils.seed import set_seed

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = load_config("configs/model/st_mamba.yaml")
    train_cfg = load_config("configs/train/full.yaml")
    cfg = merge_configs(model_cfg, train_cfg)

    if key == "a1":
        model = STMambaNoSpatial(
            **{k: v for k, v in model_cfg.items() if k != "model_name"}
        )
    else:  # a2
        kwargs = {k: v for k, v in model_cfg.items() if k != "model_name"}
        kwargs["n_layers"] = int(kwargs.pop("n_mamba_layers", 3))
        # strip mamba-only kwargs
        for k in ("d_state", "d_conv", "expand"):
            kwargs.pop(k, None)
        model = STMambaNoMamba(**kwargs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ablation] variant params = {n_params:,}", flush=True)

    data_dir = Path("data")
    normalizer = ChannelNormalizer.load(data_dir / "normalize_stats.npz")
    train_ds = ChannelDataset(data_dir / "train", normalizer=normalizer, flatten_spatial=False)
    val_ds = ChannelDataset(data_dir / "val", normalizer=normalizer, flatten_spatial=False)

    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True,
                              num_workers=int(cfg.get("num_workers", 2)), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False,
                            num_workers=int(cfg.get("num_workers", 2)), pin_memory=True)

    loss_fn = CombinedLoss(
        alpha=float(cfg.get("alpha", 1.0)),
        beta=float(cfg.get("beta", 0.1)),
        gamma=float(cfg.get("gamma", 0.05)),
    )

    log_dir = Path("results/logs") / name
    ckpt_dir = Path("results/checkpoints") / name
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
        flatten_input=False,
    )
    best_val = trainer.train()
    print(f"[ablation] {name} done. best val total = {best_val:.6f}", flush=True)


def report():
    print()
    print("=== Ablation Summary (NMSE avg dB, v=30 km/h) ===")
    rows = []
    baseline_path = Path("results/tables/st_mamba_full_seed42/results_v30.json")
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            rows.append(("st_mamba_full (baseline)", json.load(f)["nmse_avg_db"]))
    for key, cfg in ABLATIONS.items():
        f = Path(f"results/tables/{cfg['name']}/results_v30.json")
        if f.exists():
            with open(f, "r", encoding="utf-8") as fp:
                rows.append((f"{key}: {cfg['name']}", json.load(fp)["nmse_avg_db"]))
        else:
            rows.append((f"{key}: {cfg['name']}", float("nan")))

    print(f"{'Experiment':40s} | {'NMSE (dB)':>10s}")
    print("-" * 55)
    for label, v in rows:
        if isinstance(v, float):
            print(f"{label:40s} | {v:>10.2f}")
        else:
            print(f"{label:40s} | {'N/A':>10s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None,
                        choices=[*ABLATIONS.keys(), "all"])
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if args.run == "all":
        for k in ABLATIONS:
            run_ablation(k)
    elif args.run:
        run_ablation(args.run)

    if args.report:
        report()


if __name__ == "__main__":
    main()
