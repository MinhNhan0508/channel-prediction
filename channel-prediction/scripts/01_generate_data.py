"""Generate CDL-A channel dataset for training and testing.

Usage:
    python scripts/01_generate_data.py --config configs/data/cdl_a.yaml

Outputs:
    data/train/sample_*.npz
    data/val/sample_*.npz
    data/test_v30/sample_*.npz
    data/test_v50/sample_*.npz
    data/normalize_stats.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.channel_generator import ChannelConfig, generate_one_sample  # noqa: E402
from src.data.normalize import ChannelNormalizer  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _build_channel_cfg(cfg: dict) -> ChannelConfig:
    return ChannelConfig(
        n_bs=int(cfg["n_bs"]),
        n_rf=int(cfg["n_rf"]),
        n_m=int(cfg["n_m"]),
        f_carrier_hz=float(cfg["f_carrier_ghz"]) * 1e9,
        t_sample_s=float(cfg["t_sample_ms"]) * 1e-3,
        p_frames=int(cfg["p_frames"]),
        q_frames=int(cfg["q_frames"]),
    )


def _generate_split(
    out_dir: Path,
    n_samples: int,
    velocity_range,
    delay_spread_range,
    snr_range,
    base_seed: int,
    channel_cfg: ChannelConfig,
    desc: str,
    collect_for_norm: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(base_seed)
    collected = []
    for i in tqdm(range(n_samples), desc=desc):
        v = float(rng.uniform(*velocity_range))
        ds = float(rng.uniform(*delay_spread_range))
        snr = float(rng.uniform(*snr_range))
        sample = generate_one_sample(
            velocity_kmh=v,
            delay_spread_ns=ds,
            snr_db=snr,
            seed=base_seed + i,
            cfg=channel_cfg,
        )
        np.savez(out_dir / f"sample_{i:05d}.npz", **sample)
        if collect_for_norm:
            collected.append(
                np.concatenate([sample["He_history"], sample["He_future"]], axis=0)
            )
    if collect_for_norm:
        return np.stack(collected)
    return None


def _generate_fixed_split(
    out_dir: Path,
    n_samples: int,
    velocity_kmh: float,
    delay_spread_ns: float,
    snr_db: float,
    base_seed: int,
    channel_cfg: ChannelConfig,
    desc: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(n_samples), desc=desc):
        sample = generate_one_sample(
            velocity_kmh=velocity_kmh,
            delay_spread_ns=delay_spread_ns,
            snr_db=snr_db,
            seed=base_seed + i,
            cfg=channel_cfg,
        )
        np.savez(out_dir / f"sample_{i:05d}.npz", **sample)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data/cdl_a.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    channel_cfg = _build_channel_cfg(cfg)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # TRAIN (also collect for normalizer fit)
    train_data = _generate_split(
        out_dir=output_dir / "train",
        n_samples=int(cfg["n_train"]),
        velocity_range=cfg["velocity_range_train"],
        delay_spread_range=cfg["delay_spread_range"],
        snr_range=cfg["snr_range_train"],
        base_seed=int(cfg["seed"]),
        channel_cfg=channel_cfg,
        desc="train",
        collect_for_norm=True,
    )

    normalizer = ChannelNormalizer.fit(train_data)
    normalizer.save(output_dir / "normalize_stats.npz")
    print(f"[normalize] scale = {normalizer.scale:.6f}")

    # VAL
    _generate_split(
        out_dir=output_dir / "val",
        n_samples=int(cfg["n_val"]),
        velocity_range=cfg["velocity_range_train"],
        delay_spread_range=cfg["delay_spread_range"],
        snr_range=cfg["snr_range_train"],
        base_seed=int(cfg["seed"]) + 100_000,
        channel_cfg=channel_cfg,
        desc="val",
    )

    # TEST v=30 km/h
    _generate_fixed_split(
        out_dir=output_dir / "test_v30",
        n_samples=int(cfg["n_test_per_scenario"]),
        velocity_kmh=30.0,
        delay_spread_ns=50.0,
        snr_db=20.0,
        base_seed=int(cfg["seed"]) + 200_000,
        channel_cfg=channel_cfg,
        desc="test_v30",
    )

    # TEST v=50 km/h
    _generate_fixed_split(
        out_dir=output_dir / "test_v50",
        n_samples=int(cfg["n_test_per_scenario"]),
        velocity_kmh=50.0,
        delay_spread_ns=50.0,
        snr_db=20.0,
        base_seed=int(cfg["seed"]) + 300_000,
        channel_cfg=channel_cfg,
        desc="test_v50",
    )

    print("Dataset generation complete.")


if __name__ == "__main__":
    main()
