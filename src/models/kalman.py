"""Kalman channel predictor (baseline cổ điển, KHÔNG cần train).

Mô hình state-space AR(p) + lọc Kalman -- predictor cổ điển CHUẨN trong dự đoán
kênh, mạnh hơn PAD ở chỗ có mô hình nhiễu quan sát tường minh (khử nhiễu lịch sử
thay vì fit thẳng như AR-LS). Đánh giá trực tiếp như PAD/No-Prediction.

Trạng thái s[t] = [x[t], x[t-1], ..., x[t-p+1]]^T (phức), chuyển trạng thái bằng
ma trận companion từ hệ số AR (ước lượng LS như PAD). Nhiễu quá trình Q và nhiễu
quan sát R chia đều từ phương sai innovation của AR (không "nhìn trộm" SNR thật).

    kalman_predict(He_history, q_frames, order) -> He_future_pred (N,Q,nrf,nm,2)
"""
from __future__ import annotations

import numpy as np


def kalman_predict(He_history: np.ndarray, q_frames: int = 5, order: int = 4,
                   ridge: float = 1e-3, clamp_scale: float = 3.0) -> np.ndarray:
    He_history = np.asarray(He_history, dtype=np.complex64 if False else np.float32)
    N, P, n_rf, n_m, _ = He_history.shape
    x = He_history[..., 0] + 1j * He_history[..., 1]           # (N,P,nrf,nm)
    y = x.reshape(N, P, n_rf * n_m).transpose(0, 2, 1).reshape(N * n_rf * n_m, P)  # (S,P) complex
    S = y.shape[0]
    p = int(min(order, P - 1))
    if p < 1:
        last = He_history[:, -1:]
        return np.repeat(last, q_frames, axis=1)

    # --- 1. Ước lượng hệ số AR(p) bằng LS (giống PAD) ---
    M = P - p
    A = np.stack([y[:, (p - 1 - k):(p - 1 - k) + M] for k in range(p)], axis=-1)   # (S,M,p)
    yt = y[:, p:p + M]                                                              # (S,M)
    Ah = np.conj(np.transpose(A, (0, 2, 1)))
    AhA = Ah @ A
    Ahy = np.einsum("spm,sm->sp", Ah, yt)
    reg = ridge * np.mean(np.abs(AhA), axis=(1, 2), keepdims=True) * np.eye(p)[None]
    try:
        a = np.linalg.solve(AhA + reg, Ahy[..., None])[..., 0]                      # (S,p)
    except np.linalg.LinAlgError:
        a = np.zeros((S, p), dtype=complex); a[:, 0] = 1.0
    bad = ~np.isfinite(a).all(axis=1)
    a[bad] = 0.0; a[bad, 0] = 1.0

    # phương sai innovation AR -> chia đôi cho Q (quá trình) và R (quan sát)
    resid = yt - np.einsum("smp,sp->sm", A, a)                 # (S,M)
    rho = np.mean(np.abs(resid) ** 2, axis=1) + 1e-6           # (S,)
    Q_var = 0.5 * rho                                          # nhiễu quá trình
    R_var = 0.5 * rho                                          # nhiễu quan sát (khử nhiễu)

    # --- 2. Companion matrix F (S,p,p) ---
    F = np.zeros((S, p, p), dtype=complex)
    F[:, 0, :] = a
    if p > 1:
        idx = np.arange(p - 1)
        F[:, idx + 1, idx] = 1.0
    Hc = np.zeros((1, p)); Hc[0, 0] = 1.0                      # observation row
    eye_p = np.eye(p)[None]
    Qm = np.zeros((S, p, p)); Qm[:, 0, 0] = Q_var              # nhiễu vào thành phần x[t]

    # --- 3. Khởi tạo + chạy Kalman tiến qua P quan sát ---
    m = y[:, p - 1::-1][:, :p].reshape(S, p, 1).astype(complex)  # [y[p-1],...,y[0]]
    Pcov = (rho[:, None, None] * eye_p).astype(float)           # hiệp phương sai thực (nhiễu vòng)
    for t in range(p, P):
        m = F @ m                                              # predict state (S,p,1)
        Pcov = np.real(F @ Pcov @ np.conj(np.transpose(F, (0, 2, 1)))) + Qm
        innov = y[:, t][:, None, None] - (Hc[None] @ m)        # (S,1,1) complex
        Sinn = (Hc[None] @ Pcov @ Hc.T[None])[:, 0, 0] + R_var # (S,) real
        K = (Pcov @ Hc.T[None])[:, :, 0] / Sinn[:, None]       # (S,p) real gain
        m = m + K[:, :, None] * innov[:, 0, :][:, None, :]     # update (broadcast complex)
        Pcov = (eye_p - K[:, :, None] * Hc[None]) @ Pcov

    # --- 4. Ngoại suy Q bước (chỉ predict, không update) ---
    lim = clamp_scale * np.abs(y).max(axis=1)                  # (S,)
    preds = np.empty((S, q_frames), dtype=complex)
    for q in range(q_frames):
        m = F @ m
        nxt = m[:, 0, 0]
        nxt = np.where(np.isfinite(nxt), nxt, 0.0)
        mag = np.abs(nxt)
        scale = np.where(mag > lim, lim / (mag + 1e-12), 1.0)
        nxt = nxt * scale
        m[:, 0, 0] = nxt
        preds[:, q] = nxt

    preds = preds.reshape(N, n_rf * n_m, q_frames).transpose(0, 2, 1).reshape(N, q_frames, n_rf, n_m)
    return np.stack([preds.real, preds.imag], axis=-1).astype(np.float32)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import glob
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.models.pad import pad_predict
    files = sorted(glob.glob(str(Path(__file__).resolve().parents[2] / "data_sionna/test_v30/*.npz")))[:200]
    if files:
        h = np.stack([np.load(f)["He_history"] for f in files]).astype(np.float32)
        fut = np.stack([np.load(f)["He_future"] for f in files]).astype(np.float32)
        nmse = lambda pr: 10 * np.log10(((pr - fut) ** 2).sum() / (fut ** 2).sum())
        print(f"PAD    NMSE = {nmse(pad_predict(h)):.2f} dB  (lin {10**(nmse(pad_predict(h))/10):.4f})")
        k = kalman_predict(h)
        print(f"Kalman NMSE = {nmse(k):.2f} dB  (lin {10**(nmse(k)/10):.4f})  finite={np.isfinite(k).all()}")
