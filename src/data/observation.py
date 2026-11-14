"""Build history/target pairs without targets configuring observations."""
from __future__ import annotations
import math
import numpy as np
from src.data.channel_generator import _beam_training_combiner, _to_real_imag

OBSERVATION_POLICY = "history_only_v1"


def make_observation(channel: np.ndarray, history_frames: int, future_frames: int,
                     n_rf: int, snr_db: float, rng: np.random.Generator) -> dict:
    """Select beams from clean historical pilots; keep combiner fixed over P+Q.

    Clean historical physical-channel pilots are an idealized simulation
    assumption. Neither beam selection nor noise power uses future values.
    """
    channel = np.asarray(channel)
    if channel.ndim != 3 or not np.iscomplexobj(channel):
        raise ValueError("channel must be complex with shape (time, BS antennas, UE antennas)")
    if history_frames < 1 or future_frames < 1:
        raise ValueError("history_frames and future_frames must be positive")
    if channel.shape[0] != history_frames + future_frames:
        raise ValueError("channel time dimension must equal history_frames + future_frames")
    if not 1 <= n_rf <= channel.shape[1] or channel.shape[2] < 1:
        raise ValueError("invalid RF-chain or UE-antenna count")
    if not np.isfinite(snr_db) or not np.isfinite(channel).all():
        raise ValueError("channel and snr_db must be finite")
    combiner = _beam_training_combiner(channel[:history_frames].mean(axis=0), n_rf, rng)
    clean = np.einsum("rb,tbm->trm", combiner, channel)
    history = clean[:history_frames]
    power = float(np.mean(np.abs(history) ** 2))
    noise_std = math.sqrt(power * 10.0 ** (-float(snr_db) / 10.0) / 2.0)
    noise = noise_std * (rng.standard_normal(history.shape) + 1j * rng.standard_normal(history.shape))
    return {"He_history": _to_real_imag(history + noise),
            "He_history_clean": _to_real_imag(history),
            "He_future": _to_real_imag(clean[history_frames:]),
            "D": _to_real_imag(combiner)}
