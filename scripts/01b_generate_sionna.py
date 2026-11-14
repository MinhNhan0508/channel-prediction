"""Generate simulated CDL-A channels with history-only observations.

Use an empty output directory. Legacy data and checkpoints do not establish
results for this corrected policy. Data are simulated, not radio measurements.
"""
from __future__ import annotations

import argparse
import math
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sionna.phy.channel.tr38901 import CDL, PanelArray  # noqa: E402
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies  # noqa: E402
from sionna.phy import config as sionna_config  # noqa: E402

from src.data.observation import make_observation, OBSERVATION_POLICY  # noqa: E402
from src.data.normalize import ChannelNormalizer  # noqa: E402
from src.data.provenance import data_content_fingerprint  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _ula(n_elem, fc):
    return PanelArray(num_rows_per_panel=1, num_cols_per_panel=n_elem, num_rows=1, num_cols=1,
                      polarization="single", polarization_type="V", antenna_pattern="omni",
                      carrier_frequency=fc)


def _gen_batch(v_kmh, ds_ns, snr_db, B, seed, *, n_bs, n_m, n_rf, fc, ts, P, Q, bs_arr, ut_arr):
    """Sinh B sample tại (v, ds, snr) cố định. Trả list dict .npz-ready."""
    torch.manual_seed(seed)
    sionna_config.seed = int(seed)
    T = P + Q
    cdl = CDL("A", delay_spread=ds_ns * 1e-9, carrier_frequency=fc, ut_array=ut_arr,
              bs_array=bs_arr, direction="downlink", min_speed=v_kmh / 3.6, max_speed=v_kmh / 3.6)
    a, tau = cdl(batch_size=B, num_time_steps=T, sampling_frequency=1.0 / ts)
    f = subcarrier_frequencies(1, 30e3)                      # narrowband 1 subcarrier
    h = cir_to_ofdm_channel(f, a, tau)                       # (B,1,N_M,1,N_BS,T,1)
    H = h[:, 0, :, 0, :, :, 0].permute(0, 3, 2, 1).cpu().numpy()  # (B,T,N_BS,N_M)

    rng = np.random.default_rng(seed)
    out = []
    for b in range(B):
        Hb = H[b]                                            # (T,N_BS,N_M)
        sample = make_observation(Hb, P, Q, n_rf, snr_db, rng)
        sample.update(velocity=np.float32(v_kmh), delay_spread=np.float32(ds_ns), snr=np.float32(snr_db))
        out.append(sample)
    return out


def _write_split(out_dir, n_samples, v_fn, ds_fn, snr_fn, base_seed, batch, kw, desc, collect=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(base_seed)
    collected, idx = [], 0
    n_batches = math.ceil(n_samples / batch)
    for bi in tqdm(range(n_batches), desc=desc):
        B = min(batch, n_samples - idx)
        v, ds, snr = v_fn(rng), ds_fn(rng), snr_fn(rng)      # mỗi batch 1 (v,ds,snr)
        samples = _gen_batch(v, ds, snr, B, base_seed + bi, **kw)
        for s in samples:
            np.savez(out_dir / f"sample_{idx:05d}.npz", **s)
            if collect:
                collected.append(np.concatenate([s["He_history"], s["He_future"]], axis=0))
            idx += 1
    return np.stack(collected) if collect else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna_cdl.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fc = float(cfg["f_carrier_ghz"]) * 1e9
    ts = float(cfg["t_sample_ms"]) * 1e-3
    n_bs, n_m, n_rf = int(cfg["n_bs"]), int(cfg["n_m"]), int(cfg["n_rf"])
    P, Q = int(cfg["p_frames"]), int(cfg["q_frames"])
    batch = int(cfg.get("gen_batch", 256))
    out = Path(cfg["output_dir"])
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to mix datasets: output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    manifest = {
        "schema_version": 1,
        "observation_policy": OBSERVATION_POLICY,
        "status": "generating",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
                          for name in ["scripts/01b_generate_sionna.py", "src/data/observation.py",
                                       "src/data/channel_generator.py", "src/data/normalize.py", "src/data/provenance.py"]},
        "packages": {name: importlib.metadata.version(name) for name in ["sionna", "torch", "numpy"]},
        "assumptions": ["simulated CDL-A, one subcarrier", "clean historical pilots for beam selection",
                        "fixed combiner over P+Q frames", "history-only noise-power calibration"],
    }
    manifest_path = out / "metadata.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    kw = dict(n_bs=n_bs, n_m=n_m, n_rf=n_rf, fc=fc, ts=ts, P=P, Q=Q,
              bs_arr=_ula(n_bs, fc), ut_arr=_ula(n_m, fc))

    vmin, vmax = cfg["velocity_range_train"]
    dmin, dmax = cfg["delay_spread_range"]
    smin, smax = cfg["snr_range_train"]
    v_rand = lambda r: float(r.uniform(vmin, vmax))
    ds_rand = lambda r: float(r.uniform(dmin, dmax))
    snr_rand = lambda r: float(r.uniform(smin, smax))

    train = _write_split(out / "train", int(cfg["n_train"]), v_rand, ds_rand, snr_rand,
                         seed, batch, kw, "train", collect=True)
    norm = ChannelNormalizer.fit(train)
    norm.save(out / "normalize_stats.npz")
    print(f"[normalize] scale = {norm.scale:.6f}", flush=True)

    _write_split(out / "val", int(cfg["n_val"]), v_rand, ds_rand, snr_rand,
                 seed + 100_000, batch, kw, "val")

    td = float(cfg.get("test_delay_spread", 50.0))
    tsnr = float(cfg.get("test_snr", 20.0))
    for i, v in enumerate([float(x) for x in cfg["test_velocities"]]):
        _write_split(out / f"test_v{int(round(v))}", int(cfg["n_test_per_scenario"]),
                     (lambda vv: (lambda r: vv))(v), (lambda r: td), (lambda r: tsnr),
                     seed + 200_000 + i * 10_000, batch, kw, f"test_v{int(round(v))}")
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["normalizer_scale"] = norm.scale
    manifest["data_content_sha256"] = data_content_fingerprint(out)
    manifest["split_counts"] = {p.name: len(list(p.glob("sample_*.npz")))
                                for p in out.iterdir() if p.is_dir()}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Sionna dataset generation complete.", flush=True)


if __name__ == "__main__":
    main()
