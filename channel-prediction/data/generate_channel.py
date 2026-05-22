import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# THAM SỐ HỆ THỐNG MASSIVE MIMO
# ============================================================
FC          = 2.4e9     # Tần số sóng mang: 2.4 GHz
C           = 3e8       # Tốc độ ánh sáng (m/s)
FS          = 1000      # Tần số lấy mẫu (Hz)
N           = 50000     # Tổng số mẫu thời gian
SEQ_LEN     = 50        # Nhìn 50 bước quá khứ
TRAIN_RATIO = 0.8

# ── Thông số Massive MIMO ──────────────────────────────────
N_TX     = 64   # Số anten BS (Base Station) — đây là "Massive"
N_RX     = 2    # Số anten UE (User Equipment)
N_PATHS  = 10   # Số đường truyền đa đường (multipath)
D_LAMBDA = 0.5  # Khoảng cách anten / bước sóng (half-wavelength)

# Kết quả với N_TX=64, N_RX=2:
#   Kênh H(t) ∈ ℂ^(N_RX × N_TX) = ℂ^(2 × 64)
#   Mỗi bước thời gian có 2×64 = 128 số phức
#   = 256 features thực (real + imag) = 2 × N_RX × N_TX
#   Input model : (seq_len, 256)
#   Output model: (256,)


# ============================================================
# HÀM 1: Tính Doppler frequency
# ============================================================
def compute_doppler(velocity_kmh):
    """
    Tính tần số Doppler cực đại từ tốc độ di chuyển.

    Công thức: fd = v * fc / c

    Args:
        velocity_kmh: tốc độ di chuyển (km/h)

    Returns:
        fd: tần số Doppler cực đại (Hz)
    """
    v_ms = velocity_kmh / 3.6
    return v_ms * FC / C


# ============================================================
# HÀM 2: Tính steering vector của ULA
# ============================================================
def steering_vector(theta, n_antennas, d_lambda=D_LAMBDA):
    """
    Tính array steering vector cho ULA (Uniform Linear Array).

    Khi tín hiệu đến từ góc theta, mỗi anten nhận tín hiệu
    với pha khác nhau tùy vị trí trong mảng:

        a(theta) = [1, e^(j*2π*d*sin(θ)), ..., e^(j*2π*(N-1)*d*sin(θ))]

    Đây là đặc trưng quan trọng nhất của Massive MIMO so với SISO:
    - SISO: không có khái niệm steering vector
    - MIMO: mỗi đường sóng có steering vector riêng
    - Model AI phải học cả thông tin không gian (spatial) lẫn thời gian

    Args:
        theta      : góc đến (radian)
        n_antennas : số phần tử anten
        d_lambda   : khoảng cách anten / bước sóng

    Returns:
        a : steering vector shape (n_antennas,) dtype complex
    """
    n = np.arange(n_antennas)
    phase = 2 * np.pi * d_lambda * np.sin(theta) * n
    return np.exp(1j * phase)


# ============================================================
# HÀM 3: Sinh kênh Massive MIMO theo mô hình GSCM
# ============================================================
def generate_mimo_channel(n_samples, fd, n_tx=N_TX, n_rx=N_RX,
                           n_paths=N_PATHS, fs=FS, seed=42):
    """
    Sinh kênh Massive MIMO time-varying dùng mô hình GSCM
    (Geometry-based Stochastic Channel Model).

    Mô hình:
        H(t) = (1/sqrt(L)) * Σ_l  alpha_l(t) * a_r(phi_l) * a_t(theta_l)^H

    Trong đó:
        - L          : số đường truyền (paths)
        - alpha_l(t) : hệ số fading phức của đường thứ l (thay đổi theo t)
        - theta_l    : góc phát (AoD) từ BS
        - phi_l      : góc thu (AoA) tại UE
        - a_t, a_r   : steering vector tại TX và RX

    Khác với SISO:
        - SISO : h(t) là 1 số phức
        - MIMO : H(t) là ma trận (n_rx × n_tx) số phức
        - Phần không gian (steering vectors) được cố định
        - Phần thời gian (alpha_l) thay đổi theo Doppler

    Args:
        n_samples : số mẫu thời gian cần sinh
        fd        : tần số Doppler cực đại (Hz)
        n_tx      : số anten BS
        n_rx      : số anten UE
        n_paths   : số đường truyền
        fs        : tần số lấy mẫu
        seed      : random seed

    Returns:
        H : kênh MIMO shape (n_samples, n_rx, n_tx) dtype complex
    """
    rng = np.random.default_rng(seed)
    t   = np.arange(n_samples) / fs

    # Sinh tham số không gian cố định cho mỗi đường
    theta_l = rng.uniform(-np.pi/2, np.pi/2, n_paths)  # AoD tại BS
    phi_l   = rng.uniform(-np.pi/2, np.pi/2, n_paths)  # AoA tại UE
    phase0  = rng.uniform(0, 2*np.pi, n_paths)          # pha ban đầu
    doppler_angle = rng.uniform(0, 2*np.pi, n_paths)    # góc Doppler

    # Tính steering vectors (cố định theo thời gian)
    a_tx = np.array([steering_vector(th, n_tx) for th in theta_l])
    # a_tx shape: (n_paths, n_tx)
    a_rx = np.array([steering_vector(ph, n_rx) for ph in phi_l])
    # a_rx shape: (n_paths, n_rx)

    # Khởi tạo ma trận kênh
    H = np.zeros((n_samples, n_rx, n_tx), dtype=complex)

    # Tổng hợp đóng góp của từng đường
    for l in range(n_paths):
        # Hệ số fading thay đổi theo thời gian (Doppler)
        f_doppler_l = fd * np.cos(doppler_angle[l])
        alpha_l     = np.exp(1j * (2*np.pi * f_doppler_l * t + phase0[l]))
        # alpha_l shape: (n_samples,)

        # Ma trận kênh đóng góp của đường l
        # outer product: a_rx[l] × a_tx[l]^H
        G_l = np.outer(a_rx[l], np.conj(a_tx[l]))
        # G_l shape: (n_rx, n_tx)

        # Tích lũy: H(t) += alpha_l(t) * G_l
        H += alpha_l[:, np.newaxis, np.newaxis] * G_l[np.newaxis, :, :]

    # Chuẩn hóa công suất
    H /= np.sqrt(n_paths)
    return H


# ============================================================
# HÀM 4: Chuyển kênh MIMO thành chuỗi sliding window
# ============================================================
def create_sequences(H, seq_len=SEQ_LEN):
    """
    Chuyển chuỗi kênh MIMO H(t) thành cặp (input, output) cho model.

    SISO: mỗi bước có 2 features (real + imag của 1 số phức)
    MIMO: mỗi bước có 2 * n_rx * n_tx features

    Với N_TX=64, N_RX=2:
        Mỗi bước có 2 × 64 = 128 số phức
        → flatten thành 128 chiều phức
        → tách real + imag → 256 features
        Feature size = 2 × N_RX × N_TX = 256

    Cách flatten H(t):
        H(t) shape (n_rx, n_tx) → reshape (n_rx * n_tx,)
                                → cat([real, imag]) → (2 * n_rx * n_tx,)

    X shape: (n_samples, seq_len, 256)   ← input chuỗi
    y shape: (n_samples, 256)            ← output bước tiếp theo

    Args:
        H       : kênh MIMO shape (N, n_rx, n_tx) dtype complex
        seq_len : độ dài cửa sổ nhìn về quá khứ

    Returns:
        X, y : numpy arrays float32
    """
    n_samples, n_rx, n_tx = H.shape
    feature_size = 2 * n_rx * n_tx  # real + imag

    # Flatten và tách real/imag
    # H_flat: (N, n_rx*n_tx)
    # H_ri  : (N, 2*n_rx*n_tx) — half đầu là real, half sau là imag
    H_flat = H.reshape(n_samples, -1)
    H_ri   = np.concatenate([
        np.real(H_flat),
        np.imag(H_flat)
    ], axis=1)

    # Sliding window: cửa sổ trượt từng bước 1
    X, y = [], []
    for i in range(n_samples - seq_len):
        X.append(H_ri[i : i + seq_len])         # (seq_len, feature_size)
        y.append(H_ri[i + seq_len])             # (feature_size,)

    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32))


# ============================================================
# HÀM 5: Pipeline sinh và lưu dataset
# ============================================================
def generate_and_save(velocity_kmh, seed=42, save_dir='data'):
    """
    Pipeline đầy đủ: sinh kênh → tạo sequences → chia train/test → lưu.

    Args:
        velocity_kmh : tốc độ di chuyển (km/h)
        seed         : random seed (để tái lập)
        save_dir     : thư mục lưu file .npz
    """
    os.makedirs(save_dir, exist_ok=True)

    fd = compute_doppler(velocity_kmh)
    print(f"\n[v={velocity_kmh} km/h | seed={seed} | "
          f"N_TX={N_TX} | N_RX={N_RX}]  fd={fd:.1f} Hz")

    # Sinh kênh MIMO
    H = generate_mimo_channel(N, fd, N_TX, N_RX, N_PATHS, FS, seed)
    print(f"  H shape: {H.shape}  "
          f"(samples × n_rx × n_tx)")

    # Tạo sequences
    X, y = create_sequences(H, SEQ_LEN)
    print(f"  X shape: {X.shape}  "
          f"({SEQ_LEN} bước × {X.shape[2]} features)")

    # Chia train/test (không shuffle vì là time series)
    split   = int(len(X) * TRAIN_RATIO)
    X_train = X[:split]
    y_train = y[:split]
    X_test  = X[split:]
    y_test  = y[split:]

    # Chuẩn hóa theo max abs trên train set (giữ scale để inverse khi cần)
    scale   = np.max(np.abs(X_train))
    X_train = X_train / scale
    y_train = y_train / scale
    X_test  = X_test  / scale
    y_test  = y_test  / scale

    # Lưu file
    path = os.path.join(
        save_dir,
        f'mimo_v{velocity_kmh}kmh_s{seed}.npz'
    )
    np.savez(path,
             X_train   = X_train,
             y_train   = y_train,
             X_test    = X_test,
             y_test    = y_test,
             H_raw     = H,
             scale     = np.array([scale]),
             velocity  = np.array([velocity_kmh]),
             fd        = np.array([fd]),
             seed      = np.array([seed]),
             n_tx      = np.array([N_TX]),
             n_rx      = np.array([N_RX]))

    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Saved: {path}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test' : X_test,  'y_test' : y_test,
        'H_raw'  : H,       'scale'  : scale,
        'fd'     : fd
    }


# ============================================================
# HÀM 6: Vẽ biên độ kênh (anten 0)
# ============================================================
def plot_channels(data_list, velocities, n_show=500):
    """
    Vẽ biên độ kênh |h(t)| để kiểm tra trực quan dữ liệu sinh ra.

    Hiển thị 4 anten (0, 1, 5, 10) để thấy sự khác nhau giữa các anten
    trong cùng một thời điểm — minh hoạ spatial correlation của ULA.
    """
    os.makedirs('results', exist_ok=True)
    fig, axes = plt.subplots(len(velocities), 1,
                              figsize=(12, 3*len(velocities)),
                              sharex=True)
    if len(velocities) == 1:
        axes = [axes]

    colors = ['steelblue', 'darkorange', 'crimson']
    t_ms   = np.arange(n_show) / FS * 1000

    for ax, data, v, color in zip(axes, data_list, velocities, colors):
        # Lấy kênh tại (rx=0, tx=0)
        h0 = data['H_raw'][:n_show, 0, 0]
        fd = data['fd']

        ax.plot(t_ms, np.abs(h0),
                color=color, linewidth=0.8, alpha=0.9,
                label='Anten TX 0')

        # Vẽ thêm vài anten TX để thấy sự khác nhau
        for ant in [1, 5, 10]:
            if ant < N_TX:
                h_ant = data['H_raw'][:n_show, 0, ant]
                ax.plot(t_ms, np.abs(h_ant),
                        linewidth=0.5, alpha=0.4)

        ax.set_ylabel('|h(t)|')
        ax.set_title(
            f'v={v} km/h | fd={fd:.0f} Hz | '
            f'{N_TX} anten BS (hiển thị anten 0, 1, 5, 10 tại RX 0)',
            fontsize=10
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel('Thời gian (ms)')
    plt.suptitle(
        f'Kênh Massive MIMO Rayleigh Fading '
        f'({N_TX}×{N_RX}, {N_PATHS} paths)',
        fontsize=13
    )
    plt.tight_layout()
    out = 'results/mimo_channel_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out}")


# ============================================================
# CHẠY CHÍNH
# ============================================================
if __name__ == "__main__":
    feature_size = 2 * N_TX * N_RX
    print("=" * 60)
    print("  Massive MIMO Channel Data Generator")
    print(f"  N_TX={N_TX} (BS antennas) | N_RX={N_RX} (UE antennas)")
    print(f"  N={N} samples | SEQ_LEN={SEQ_LEN} | N_PATHS={N_PATHS}")
    print(f"  Feature size per step = {feature_size} "
          f"(2 × N_RX × N_TX = 2 × {N_RX} × {N_TX})")
    print("=" * 60)

    velocities = [30, 120, 300]
    seeds      = [42, 123, 456]

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Seed = {seed}")
        print(f"{'='*60}")
        for v in velocities:
            generate_and_save(v, seed=seed)

    # Vẽ đồ thị seed=42
    print("\nVẽ đồ thị kênh (seed=42)...")
    data_list = []
    for v in velocities:
        d = np.load(f'data/mimo_v{v}kmh_s42.npz', allow_pickle=True)
        data_list.append({
            'H_raw': d['H_raw'],
            'fd'   : float(d['fd'][0])
        })
    plot_channels(data_list, velocities)

    print("\nXong! Kiểm tra thư mục data/")