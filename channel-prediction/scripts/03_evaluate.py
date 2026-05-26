"""Evaluate one trained model on a test scenario.

Usage:
    python scripts/03_evaluate.py --model st_mamba --exp st_mamba_full_seed42 \
        --test_scenario v30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ChannelDataset  # noqa: E402
from src.data.normalize import ChannelNormalizer  # noqa: E402
from src.evaluation.evaluator import Evaluator  # noqa: E402
from src.models.utils import build_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--snr_list", type=str, default="10,12,14,16,18,20")
    parser.add_argument("--test_scenario", type=str, default="v30", choices=["v30", "v50"])
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snr_list = [float(s) for s in args.snr_list.split(",")]

    model_cfg = load_config(f"configs/model/{args.model}.yaml")
    model = build_model(args.model, model_cfg)
    ckpt_path = Path(f"results/checkpoints/{args.exp}/checkpoint_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    data_dir = Path(args.data_dir)
    normalizer = ChannelNormalizer.load(data_dir / "normalize_stats.npz")
    flatten = args.model != "st_mamba"
    test_ds = ChannelDataset(
        data_dir / f"test_{args.test_scenario}",
        normalizer=normalizer,
        flatten_spatial=flatten,
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    evaluator = Evaluator(model, device, normalizer, flatten_input=flatten)
    results = evaluator.evaluate(test_loader, snr_db_list=snr_list)

    out_dir = Path(f"results/tables/{args.exp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_{args.test_scenario}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[evaluate] NMSE avg = {results['nmse_avg_db']:.2f} dB", flush=True)
    print(f"[evaluate] NMSE per frame = {results['nmse_per_frame_db']}", flush=True)
    print(f"[evaluate] saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
