"""3GPP CDL-A channel generator for MIMO channel-prediction dataset.

Implements the model from PLAN.md section 5:
    H(t) = sum_l xi_l * a_BS(phi_BS,l) * a_user^H(phi_user,l) * exp(-j 2 pi f_l t T_s)

The 23-cluster power-delay-angle profile is taken from 3GPP TR 38.901 Table 7.7.1-1
(CDL-A, NLOS). Hybrid analog combining is applied to reduce N_BS x N_M to N_RF x N_M
"effective" channels which is what the prediction model actually sees.

Outputs (per generated sample):
    He_history: (P, N_RF, N_M, 2) real/imag history
    He_future:  (Q, N_RF, N_M, 2) ground-truth future
    D:          (N_RF, N_BS, 2)   analog combiner (per-sample, fixed across time)
    velocity, delay_spread, snr: scalar metadata
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


# 3GPP TR 38.901 v17.0.0 Table 7.7.1-1 -- CDL-A (NLOS)
# Columns: (cluster_index, normalized_delay, power_dB, AoD_deg, AoA_deg, ZoD_deg, ZoA_deg)
CDL_A_PROFILE: Tuple[Tuple[float, float, float, float, float, float], ...] = (
    (0.0000, -13.4, -178.1,  51.3,  50.2, 125.4),
    (0.3819, -22.0, -178.1,  51.3,  50.2, 125.4),
    (0.4025, -15.2, -178.1,  51.3,  50.2, 125.4),
    (0.5868,  -8.2,  -93.2,  -39.1, 88.3, 117.5),
    (0.4610, -22.0,  -93.2,  -39.1, 88.3, 117.5),
    (0.5375, -22.0,  -93.2,  -39.1, 88.3, 117.5),
    (0.6708, -25.3,  -93.2,  -39.1, 88.3, 117.5),
    (0.5750, -22.0,  -93.2,  -39.1, 88.3, 117.5),
    (0.7618,  -7.5,   34.6,  -34.5, 91.3, 116.0),
    (1.5375, -25.5,  -64.5,  -32.9, 103.8, 113.1),
    (1.8978, -19.8,  -32.9,  -32.9,  93.0, 109.7),
    (2.2242, -27.7,   52.6,  -32.9,  91.4, 107.9),
    (2.1718, -18.0,  -132.1, -26.6,  101.1, 113.1),
    (2.4942, -27.6,   77.2,  -26.6,  103.8, 113.1),
    (2.5119, -22.0,  -64.5,  -26.6,  103.8, 113.1),
    (3.0582, -23.7,  -32.9,  -26.6,  103.8, 113.1),
    (4.0810, -16.4,   52.6,  -32.9,  91.4, 107.9),
    (4.4579, -24.0,   -1.6,    8.5,  108.1, 113.1),
    (4.5695, -16.0,   13.7,    8.5,  101.1, 113.1),
    (4.7966, -15.0,   77.2,   13.5,  113.1, 110.8),
    (5.0066, -18.0,  -103.0,   -1.6, 101.1, 113.1),
    (5.3043, -25.0,   77.2,    8.5,  113.1, 110.8),
    (9.6586, -25.0,  -178.1,   51.3, 116.7, 117.5),
)

assert len(CDL_A_PROFILE) == 23, "CDL-A must have 23 clusters per 3GPP TR 38.901"

SPEED_OF_LIGHT = 299_792_458.0  # m/s


@dataclass
class ChannelConfig:
    """All numbers come directly from PLAN.md sec. 2.2."""

    n_bs: int = 64
    n_rf: int = 8
    n_m: int = 4
    f_carrier_hz: float = 30e9
    t_sample_s: float = 0.625e-3
    p_frames: int = 30
    q_frames: int = 5
    # Đa đường giàu hơn: mỗi cluster CDL-A tách thành n_subrays sub-ray (3GPP có
    # ~20). n_subrays=1 = hành vi cũ (kênh ~tổng-sin -> PAD/AR gần tối ưu).
    # n_subrays lớn -> chuỗi/coef là tổng RẤT nhiều sin lệch nhẹ -> AR bậc thấp
    # với P=30 không giải nổi, NN khai thác cấu trúc không-gian chung để thắng.
    n_subrays: int = 1
    subray_spread_deg: float = 2.0


def _ula_steering(n_ant: int, angle_rad: float, spacing_wl: float = 0.5) -> np.ndarray:
    """Uniform Linear Array steering vector. Returns complex (n_ant,).

    a_k = (1/sqrt(N)) * exp(-j 2 pi k d sin(theta) / lambda)
    """
    k = np.arange(n_ant)
    phase = -2.0 * math.pi * spacing_wl * k * math.sin(angle_rad)
    return np.exp(1j * phase) / math.sqrt(n_ant)


def _build_paths(rng: np.random.Generator, delay_spread_ns: float,
                 n_subrays: int = 1, subray_spread_deg: float = 2.0) -> Dict[str, np.ndarray]:
    """Sample CDL-A path angles/gains/Doppler. Mỗi trong 23 cluster được tách thành
    `n_subrays` sub-ray (theo 3GPP TR 38.901), mỗi sub-ray lệch góc Laplacian quanh
    tâm cluster và chia đều công suất cluster. Nhiều sub-ray -> đa đường giàu ->
    phá khả năng giải của AR/Prony bậc thấp.
    """
    aod0 = np.array([p[2] for p in CDL_A_PROFILE], dtype=np.float64)
    aoa0 = np.array([p[3] for p in CDL_A_PROFILE], dtype=np.float64)
    power_db = np.array([p[1] for p in CDL_A_PROFILE], dtype=np.float64)
    norm_delay = np.array([p[0] for p in CDL_A_PROFILE], dtype=np.float64)

    n_sub = max(1, int(n_subrays))
    K = 23 * n_sub

    # Cluster power (chuẩn hoá tổng = 1), chia đều cho các sub-ray của cluster
    power_lin = np.power(10.0, power_db / 10.0)
    power_lin = power_lin / power_lin.sum()
    subray_power = np.repeat(power_lin / n_sub, n_sub)               # (K,)

    # Góc sub-ray = tâm cluster + nhiễu Laplacian (intra-cluster spread)
    spread = subray_spread_deg / math.sqrt(2.0)
    aod_deg = np.repeat(aod0, n_sub) + rng.laplace(0.0, spread, size=K)
    aoa_deg = np.repeat(aoa0, n_sub) + rng.laplace(0.0, spread, size=K)
    aod_rad = np.deg2rad(aod_deg)
    aoa_rad = np.deg2rad(aoa_deg)

    # Gain phức Rayleigh theo công suất sub-ray, pha ngẫu nhiên
    amp = np.sqrt(subray_power / 2.0)
    xi = (rng.standard_normal(K) + 1j * rng.standard_normal(K)) * amp

    delay_s = np.repeat(norm_delay, n_sub) * (delay_spread_ns * 1e-9)

    return {
        "aod_rad": aod_rad,
        "aoa_rad": aoa_rad,
        "xi": xi,
        "delay_s": delay_s,
    }


def _beam_training_combiner(
    H_avg: np.ndarray, n_rf: int, rng: np.random.Generator
) -> np.ndarray:
    """Pick the N_RF strongest beams from a DFT codebook to form analog combiner D.

    H_avg: (N_BS, N_M) average channel for beam selection
    Returns D: (N_RF, N_BS) complex.
    """
    n_bs = H_avg.shape[0]
    # DFT codebook of size N_BS
    k = np.arange(n_bs)
    n = np.arange(n_bs).reshape(-1, 1)
    F = np.exp(-1j * 2.0 * math.pi * k * n / n_bs) / math.sqrt(n_bs)  # (N_BS, N_BS)
    # Beam metric: ||F^H H||^2 averaged over user antennas
    Y = F.conj().T @ H_avg  # (N_BS, N_M)
    beam_power = np.sum(np.abs(Y) ** 2, axis=1)  # (N_BS,)
    top_idx = np.argpartition(-beam_power, n_rf)[:n_rf]
    top_idx = top_idx[np.argsort(-beam_power[top_idx])]
    D = F[:, top_idx].conj().T  # (N_RF, N_BS) -- combiner applied as D @ H
    # Add tiny random phase jitter so combiner is sample-specific
    jitter = np.exp(1j * rng.uniform(-0.05, 0.05, size=(n_rf, 1)))
    return D * jitter


def _to_real_imag(x: np.ndarray) -> np.ndarray:
    return np.stack([x.real, x.imag], axis=-1).astype(np.float32)


def generate_one_sample(
    velocity_kmh: float,
    delay_spread_ns: float,
    snr_db: float,
    seed: int,
    cfg: ChannelConfig = ChannelConfig(),
) -> Dict[str, np.ndarray]:
    """Generate one (history, future) channel sequence.

    The model in PLAN section 5.1 is a narrowband sum-of-sinusoids channel; we sample
    one realization of cluster gains/angles for the sample, then advance through
    P + Q frames using Doppler-induced phase rotation.
    """
    rng = np.random.default_rng(seed)
    paths = _build_paths(rng, delay_spread_ns, cfg.n_subrays, cfg.subray_spread_deg)
    aod_rad = paths["aod_rad"]
    aoa_rad = paths["aoa_rad"]
    xi = paths["xi"]

    # Doppler shift per path: f_l = v/lambda * cos(AoA) (project onto motion direction)
    velocity_mps = velocity_kmh / 3.6
    wavelength = SPEED_OF_LIGHT / cfg.f_carrier_hz
    f_doppler = velocity_mps / wavelength * np.cos(aoa_rad)  # (23,)

    # Precompute steering vectors for all paths
    a_bs = np.stack([_ula_steering(cfg.n_bs, ang) for ang in aod_rad], axis=0)   # (L, N_BS)
    a_user = np.stack([_ula_steering(cfg.n_m, ang) for ang in aoa_rad], axis=0)  # (L, N_M)

    total_frames = cfg.p_frames + cfg.q_frames
    t_grid = np.arange(total_frames) * cfg.t_sample_s  # (T,)

    # H(t) shape: (T, N_BS, N_M)
    # H(t) = sum_l xi_l * exp(-j 2 pi f_l t) * a_bs[l] outer a_user[l]^H
    phase = -2.0 * math.pi * np.outer(t_grid, f_doppler)  # (T, L)
    time_factor = np.exp(1j * phase) * xi[None, :]        # (T, L)

    # outer products: (L, N_BS, N_M)
    outer = a_bs[:, :, None] * np.conj(a_user)[:, None, :]
    # combine
    # H[t, b, m] = sum_l time_factor[t, l] * outer[l, b, m]
    H_full = np.tensordot(time_factor, outer, axes=([1], [0]))  # (T, N_BS, N_M)

    # Beam training on time-averaged channel
    H_avg = np.mean(H_full, axis=0)  # (N_BS, N_M)
    D = _beam_training_combiner(H_avg, cfg.n_rf, rng)  # (N_RF, N_BS)

    # Effective channel He(t) = D @ H(t)  -> (T, N_RF, N_M)
    He_full = np.einsum("rb,tbm->trm", D, H_full)

    # Add AWGN at the specified SNR (per-sample, complex Gaussian, mean noise power scaled to He)
    signal_power = float(np.mean(np.abs(He_full) ** 2))
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / max(snr_linear, 1e-12)
    noise_std = math.sqrt(noise_power / 2.0)
    noise = noise_std * (
        rng.standard_normal(He_full.shape) + 1j * rng.standard_normal(He_full.shape)
    )
    He_noisy = He_full + noise

    He_history = _to_real_imag(He_noisy[: cfg.p_frames])  # (P, N_RF, N_M, 2)
    He_future = _to_real_imag(He_noisy[cfg.p_frames :])   # (Q, N_RF, N_M, 2)
    D_arr = _to_real_imag(D)                              # (N_RF, N_BS, 2)

    return {
        "He_history": He_history,
        "He_future": He_future,
        "D": D_arr,
        "velocity": np.float32(velocity_kmh),
        "delay_spread": np.float32(delay_spread_ns),
        "snr": np.float32(snr_db),
    }


if __name__ == "__main__":
    sample = generate_one_sample(
        velocity_kmh=40.0, delay_spread_ns=100.0, snr_db=15.0, seed=42
    )
    print("He_history:", sample["He_history"].shape, sample["He_history"].dtype)
    print("He_future :", sample["He_future"].shape, sample["He_future"].dtype)
    print("D         :", sample["D"].shape)
    print("velocity  :", sample["velocity"], "snr:", sample["snr"])
    assert sample["He_history"].shape == (30, 8, 4, 2)
    assert sample["He_future"].shape == (5, 8, 4, 2)
    print("channel_generator OK")
