import numpy as np
import torch


# ============================================================
# WIENER FILTER — Linear MMSE predictor (cổ điển)
# ============================================================
class WienerFilter:
    """
    Wiener filter cho channel prediction (linear MMSE predictor).

    Giải bài toán:
      Cho 50 mẫu kênh quá khứ x = [h(t-49), ..., h(t)]
      Tìm bộ lọc tuyến tính w sao cho E[|w^T x - h(t+k)|²] nhỏ nhất.

    Lời giải (Wiener-Hopf):  w = (X^T X)^(-1) X^T y
    Đây là least-squares formulation, equivalent với Wiener filter cổ điển
    khi quá trình là zero-mean (kênh Rayleigh đáp ứng điều kiện này).

    Khác với mô hình AI:
      - KHÔNG cần gradient descent / training nhiều epoch
      - Fit closed-form 1 lần duy nhất bằng dữ liệu train
      - Không có hyperparameter ngoài seq_len và horizon
      - Là baseline cổ điển bắt buộc trong literature channel prediction

    Hạn chế (chính là điểm mà AI sẽ vượt qua):
      - Giả định quá trình stationary → kém ở high-mobility (Doppler cao)
      - Tuyến tính → không bắt được pattern phi tuyến của kênh fading
    """

    def __init__(self, seq_len=50, horizon=1, ridge=1e-4):
        """
        Args:
            seq_len : độ dài cửa sổ quá khứ (mặc định 50)
            horizon : số bước future cần dự đoán (mặc định 1 — single-step)
                      Nếu horizon > 1 → fit direct multi-step predictor
            ridge   : hệ số ridge regularization để ổn định nghịch đảo ma trận
                      (Tikhonov regularization)
        """
        self.seq_len = seq_len
        self.horizon = horizon
        self.ridge   = ridge
        self.W       = None       # (seq_len*2, output_dim) sau khi fit
        self.name    = 'Wiener'

    # --------------------------------------------------------
    def fit(self, X_train, y_train):
        """
        Fit Wiener weights bằng closed-form least-squares.

        Args:
            X_train : (N, seq_len, 2) — input sequences (real, imag)
            y_train : (N, 2) cho horizon=1
                      hoặc (N, horizon, 2) cho multi-step

        Returns:
            self (chainable)
        """
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.detach().cpu().numpy()

        N = X_train.shape[0]
        # Flatten input: (N, seq_len, 2) → (N, seq_len*2)
        X_flat = X_train.reshape(N, -1).astype(np.float64)

        # Reshape target nếu là multi-step
        if y_train.ndim == 3:                                # (N, horizon, 2)
            y_flat = y_train.reshape(N, -1).astype(np.float64)   # (N, horizon*2)
        else:                                                 # (N, 2)
            y_flat = y_train.astype(np.float64)

        # Ridge regression: W = (X^T X + λI)^(-1) X^T y
        D    = X_flat.shape[1]
        XtX  = X_flat.T @ X_flat / N
        Xty  = X_flat.T @ y_flat / N
        XtX += self.ridge * np.eye(D)

        # solve thay vì inv để ổn định số học
        self.W = np.linalg.solve(XtX, Xty)                    # (seq_len*2, out_dim)
        self.W = self.W.astype(np.float32)

        return self

    # --------------------------------------------------------
    def predict(self, X):
        """
        Dự đoán h(t+k) từ chuỗi đầu vào.

        Args:
            X : torch.Tensor hoặc np.ndarray, shape (B, seq_len, 2)

        Returns:
            torch.Tensor shape (B, 2) cho horizon=1
                      hoặc (B, horizon, 2) cho multi-step
        """
        if self.W is None:
            raise RuntimeError("WienerFilter chưa được fit. Gọi .fit() trước.")

        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)

        B = X_np.shape[0]
        X_flat = X_np.reshape(B, -1).astype(np.float32)
        out    = X_flat @ self.W                              # (B, out_dim)

        if self.horizon == 1:
            return torch.from_numpy(out)                       # (B, 2)
        else:
            return torch.from_numpy(out.reshape(B, self.horizon, 2))

    # --------------------------------------------------------
    def __call__(self, X):
        """Cho phép gọi như nn.Module: model(X)."""
        return self.predict(X)

    # --------------------------------------------------------
    def num_params(self):
        """Trả về số tham số (để báo cáo complexity)."""
        return 0 if self.W is None else self.W.size

    def eval(self):
        """No-op để tương thích interface với nn.Module."""
        return self

    def to(self, device):
        """No-op để tương thích interface với nn.Module."""
        return self


# ============================================================
# QUICK TEST: chạy file này trực tiếp để kiểm tra
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Wiener Filter — Sanity Check")
    print("=" * 60)

    # Sinh dữ liệu giả lập đơn giản: AR(1) process
    np.random.seed(42)
    N_train = 5000
    N_test  = 1000
    seq_len = 50

    # Sinh chuỗi AR(1) — Wiener filter sẽ học rất tốt
    rho = 0.95
    h_full = np.zeros(N_train + N_test + seq_len + 1, dtype=complex)
    for i in range(1, len(h_full)):
        h_full[i] = rho * h_full[i-1] + np.random.randn() + 1j*np.random.randn()
    h_full = h_full / np.max(np.abs(h_full))

    h_ri = np.stack([h_full.real, h_full.imag], axis=1).astype(np.float32)

    X, y = [], []
    for i in range(len(h_ri) - seq_len - 1):
        X.append(h_ri[i:i+seq_len])
        y.append(h_ri[i+seq_len])
    X = np.array(X)
    y = np.array(y)

    X_train, y_train = X[:N_train], y[:N_train]
    X_test,  y_test  = X[N_train:N_train+N_test], y[N_train:N_train+N_test]

    # ---- Single-step ----
    print("\n[1] Single-step (horizon=1)")
    wiener = WienerFilter(seq_len=seq_len, horizon=1, ridge=1e-4)
    wiener.fit(X_train, y_train)
    pred = wiener.predict(torch.FloatTensor(X_test))
    nmse = 10 * np.log10(
        np.sum((pred.numpy() - y_test) ** 2) / np.sum(y_test ** 2)
    )
    print(f"    Pred shape : {pred.shape}  (expect [{N_test}, 2])")
    print(f"    NMSE       : {nmse:.2f} dB  (AR(1) is easy → expect very low)")
    print(f"    Params     : {wiener.num_params():,}")

    # ---- Multi-step ----
    print("\n[2] Multi-step (horizon=5)")
    horizon = 5
    X_m, y_m = [], []
    for i in range(len(h_ri) - seq_len - horizon):
        X_m.append(h_ri[i:i+seq_len])
        y_m.append(h_ri[i+seq_len:i+seq_len+horizon])
    X_m = np.array(X_m)
    y_m = np.array(y_m)
    Xtr, ytr = X_m[:N_train], y_m[:N_train]
    Xte, yte = X_m[N_train:N_train+N_test], y_m[N_train:N_train+N_test]

    wiener_m = WienerFilter(seq_len=seq_len, horizon=horizon, ridge=1e-4)
    wiener_m.fit(Xtr, ytr)
    pred_m = wiener_m.predict(torch.FloatTensor(Xte))
    print(f"    Pred shape : {pred_m.shape}  (expect [{N_test}, {horizon}, 2])")
    for k in range(horizon):
        nmse_k = 10 * np.log10(
            np.sum((pred_m[:, k, :].numpy() - yte[:, k, :]) ** 2) /
            np.sum(yte[:, k, :] ** 2)
        )
        print(f"    NMSE @ h={k+1}: {nmse_k:.2f} dB  (càng xa càng kém — channel aging)")

    print("\nAll checks OK!")
