"""PAD -- classical model-based predictor (Prony / AR linear prediction).

Baseline kinh điển có trong cả 3 bài tham chiếu (PAD = Prony Angular-Delay). Ở
đây triển khai dạng **AR(p) linear predictor** trên từng hệ số CSI phức: ước
lượng hệ số tự hồi quy bằng least-squares từ chuỗi lịch sử rồi ngoại suy Q bước.
Đây là cách bắt động học Doppler (tổng các mũ phức) bằng mô hình tuyến tính, KHÔNG
cần huấn luyện -- đánh giá trực tiếp như No-Prediction/Perfect CSI trong evaluate.py.

Hàm chính:
    pad_predict(He_history, q_frames, order) -> He_future_pred
        He_history: (N, P, n_rf, n_m, 2) thang vật lý
        trả về    : (N, Q, n_rf, n_m, 2)
"""
from __future__ import annotations

import numpy as np


def pad_predict(He_history: np.ndarray, q_frames: int = 5, order: int = 4,
                ridge: float = 1e-3) -> np.ndarray:
    """Dự đoán Q khung tương lai bằng AR(p) least-squares trên từng hệ số phức.

    Vectorised trên toàn bộ N*n_rf*n_m chuỗi: dựng phương trình chuẩn
    (A^H A + λI) a = A^H y theo lô rồi giải 1 lần.
    """
    He_history = np.asarray(He_history, dtype=np.float32)
    N, P, n_rf, n_m, _ = He_history.shape
    x = He_history[..., 0] + 1j * He_history[..., 1]           # (N,P,n_rf,n_m)
    x = x.reshape(N, P, n_rf * n_m).transpose(0, 2, 1)         # (N, C, P)
    C = x.shape[1]
    series = x.reshape(N * C, P)                               # (S, P)
    S = series.shape[0]
    p = int(min(order, P - 1))
    if p < 1:                                                  # quá ngắn -> lặp frame cuối
        last = He_history[:, -1:]
        return np.repeat(last, q_frames, axis=1)

    M = P - p                                                  # số phương trình
    # features A[s, t, k] = series[s, (p-1)+t - k],  target y[s,t] = series[s, p+t]
    idx_t = np.arange(M)
    A = np.stack([series[:, (p - 1 - k):(p - 1 - k) + M] for k in range(p)], axis=-1)  # (S,M,p)
    y = series[:, p:p + M]                                     # (S,M)

    Ah = np.conj(np.transpose(A, (0, 2, 1)))                   # (S,p,M)
    AhA = Ah @ A                                               # (S,p,p)
    Ahy = np.einsum("spm,sm->sp", Ah, y)                       # (S,p)
    reg = ridge * np.mean(np.abs(AhA), axis=(1, 2), keepdims=True) * np.eye(p)[None]
    try:
        coeff = np.linalg.solve(AhA + reg, Ahy[..., None])[..., 0]   # (S,p) AR coeffs
    except np.linalg.LinAlgError:
        coeff = np.zeros((S, p), dtype=complex)
        coeff[:, 0] = 1.0                                      # fallback: giữ frame cuối

    # ổn định: nếu nghiệm bùng nổ -> rơi về giữ nguyên frame cuối cho chuỗi đó
    bad = ~np.isfinite(coeff).all(axis=1)
    if bad.any():
        coeff[bad] = 0.0
        coeff[bad, 0] = 1.0

    # ngoại suy đệ quy Q bước
    buf = list(series[:, -p:].T)                               # p phần tử cuối (mỗi (S,))
    preds = np.empty((S, q_frames), dtype=complex)
    hist = series[:, -p:].copy()                               # (S,p) [t-p..t-1]
    for q in range(q_frames):
        nxt = np.einsum("sk,sk->s", coeff, hist[:, ::-1])      # a_1 x[t-1]+...+a_p x[t-p]
        if not np.isfinite(nxt).all():
            nxt = np.where(np.isfinite(nxt), nxt, hist[:, -1])
        preds[:, q] = nxt
        hist = np.concatenate([hist[:, 1:], nxt[:, None]], axis=1)

    preds = preds.reshape(N, C, q_frames).transpose(0, 2, 1)  # (N,Q,C)
    preds = preds.reshape(N, q_frames, n_rf, n_m)
    out = np.stack([preds.real, preds.imag], axis=-1).astype(np.float32)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # chuỗi mũ phức (giả lập Doppler) để kiểm tra ngoại suy
    P, Q = 30, 5
    t = np.arange(P + Q)
    sig = np.exp(1j * 0.3 * t) * 0.5
    full = np.stack([sig.real, sig.imag], -1).astype(np.float32)
    full = np.broadcast_to(full[None, :, None, None, :], (2, P + Q, 8, 4, 2)).copy()
    pred = pad_predict(full[:, :P], q_frames=Q)
    err = np.mean((pred - full[:, P:]) ** 2)
    print(f"PAD OK | pred shape={pred.shape} | extrapolation MSE (complex exp)={err:.4e}")
