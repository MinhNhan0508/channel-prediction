"""Sanity check: each model should overfit a tiny dataset.

If a model cannot reach < -30 dB NMSE on 8 training samples in ~200 steps, there
is a bug. Stop and debug before full training.

Usage:
    python scripts/06_sanity_check.py st_mamba
    python scripts/06_sanity_check.py rnn
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ChannelDataset  # noqa: E402
from src.data.normalize import ChannelNormalizer  # noqa: E402
from src.losses.nmse import NMSELoss  # noqa: E402
from src.models.utils import build_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main(model_name: str, n_samples: int = 8, n_steps: int = 2000):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sanity] model={model_name} device={device}", flush=True)

    normalizer = ChannelNormalizer.load(Path("data/normalize_stats.npz"))
    flatten = model_name != "st_mamba"
    full_ds = ChannelDataset(Path("data/train"), normalizer=normalizer, flatten_spatial=flatten)
    tiny_ds = Subset(full_ds, list(range(n_samples)))
    loader = DataLoader(tiny_ds, batch_size=n_samples, shuffle=True)

    model_cfg = load_config(f"configs/model/{model_name}.yaml")
    # Force-disable dropout for sanity: we WANT the model to memorize 8 samples.
    for key in ("dropout",):
        if key in model_cfg:
            model_cfg[key] = 0.0
    model = build_model(model_name, model_cfg).to(device)
    # Belt-and-braces: zero every Dropout module inside the built model.
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=1e-5
    )
    loss_fn = NMSELoss()

    model.train()
    last_loss = float("nan")
    for step in range(n_steps):
        for batch in loader:
            x = batch["history"].to(device)
            y = batch["future"].to(device)
            if flatten and x.dim() == 5:
                B = x.shape[0]
                x = x.view(B, x.shape[1], -1)
                y_target = y.view(B, y.shape[1], -1)
            else:
                y_target = y
            pred = model(x)
            loss = loss_fn(pred, y_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            last_loss = float(loss.item())
        if step % 20 == 0:
            nmse_db = 10.0 * torch.log10(torch.tensor(max(last_loss, 1e-12))).item()
            print(f"  step {step:3d} | NMSE = {last_loss:.6f} | dB = {nmse_db:.2f}", flush=True)

    final_db = 10.0 * torch.log10(torch.tensor(max(last_loss, 1e-12))).item()
    verdict = "PASS" if final_db < -30.0 else "FAIL"
    print(f"[{verdict}] {model_name} final NMSE = {final_db:.2f} dB", flush=True)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "st_mamba"
    main(name)
