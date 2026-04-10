import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# CÁC THAM SỐ HỆ THỐNG
# ============================================================
FC = 2.4e9      # Tần số sóng mang: 2.4 GHz
C  = 3e8        # Tốc độ ánh sáng (m/s)
FS = 1000       # Tần số lấy mẫu: 1000 Hz
N  = 50000      # Tổng số mẫu kênh sinh ra
SEQ_LEN = 50    # Độ dài chuỗi đầu vào LSTM (nhìn 50 bước quá khứ)
TRAIN_RATIO = 0.8  # 80% train, 20% test
N_PATHS = 20    # Số đường truyền trong mô hình Jakes


# ============================================================
# HÀM 1: Tính Doppler frequency từ tốc độ di chuyển
# ============================================================
def compute_doppler(velocity_kmh):
    """
    Tính tần số Doppler fd từ tốc độ xe/tàu.

    Công thức: fd = v * fc / c
      - v  : tốc độ (m/s)
      - fc : tần số sóng mang (Hz)
      - c  : tốc độ ánh sáng (m/s)

    Ví dụ: 300 km/h, fc = 2.4 GHz -> fd ≈ 667 Hz
    fd càng lớn -> kênh thay đổi càng nhanh -> khó dự đoán hơn
    """
    v_ms = velocity_kmh / 3.6       # đổi km/h sang m/s
    fd   = v_ms * FC / C
    return fd


# ============================================================
# HÀM 2: Sinh kênh Rayleigh dùng mô hình Jakes
# ============================================================
def generate_rayleigh_jakes(N, fd, fs=FS, n_paths=N_PATHS, seed=42):
    """
    Sinh chuỗi kênh truyền Rayleigh time-varying dùng mô hình Jakes.

    Mô hình Jakes mô phỏng môi trường tán xạ đa đường (multipath):
    - Có n_paths đường truyền độc lập
    - Mỗi đường có góc đến (AoA) ngẫu nhiên -> Doppler shift khác nhau
    - Tổng hợp lại thành kênh phức h(t)

    Args:
        N       : số mẫu cần sinh
        fd      : Doppler frequency (Hz)
        fs      : tần số lấy mẫu (Hz)
        n_paths : số đường truyền
        seed    : random seed để tái lập kết quả

    Returns:
        h : mảng complex shape (N,) - hệ số kênh truyền
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(N) / fs           # trục thời gian (giây)
    h = np.zeros(N, dtype=complex)

    for _ in range(n_paths):
        alpha = np.random.uniform(0, 2 * np.pi)   # góc đến ngẫu nhiên
        phi   = np.random.uniform(0, 2 * np.pi)   # pha ngẫu nhiên

        # Doppler shift của đường này: fd * cos(alpha)
        f_d_n = fd * np.cos(alpha)

        # Cộng dồn đóng góp của đường này vào kênh tổng
        h += np.exp(1j * (2 * np.pi * f_d_n * t + phi))

    # Chuẩn hóa công suất trung bình = 1
    h /= np.sqrt(n_paths)
    return h


# ============================================================
# HÀM 3: Tạo cặp (X, y) dạng chuỗi trượt cho LSTM
# ============================================================
def create_sequences(h, seq_len=SEQ_LEN):
    """
    Chuyển chuỗi kênh h(t) thành các cặp (input, output) cho LSTM.

    Sliding window: dùng seq_len bước quá khứ để dự đoán bước tiếp theo.

    Ví dụ seq_len = 3:
      Input:  [h(0), h(1), h(2)]  -> Output: h(3)
      Input:  [h(1), h(2), h(3)]  -> Output: h(4)
      ...

    Vì kênh là số phức, tách thành 2 kênh real và imaginary:
      -> X shape: (num_samples, seq_len, 2)
      -> y shape: (num_samples, 2)

    Args:
        h       : chuỗi kênh phức shape (N,)
        seq_len : độ dài cửa sổ nhìn về quá khứ

    Returns:
        X : input sequences
        y : target values
    """
    # Tách real và imaginary
    h_ri = np.stack([np.real(h), np.imag(h)], axis=1)  # (N, 2)

    X, y = [], []
    for i in range(len(h_ri) - seq_len):
        X.append(h_ri[i : i + seq_len])    # cửa sổ seq_len bước
        y.append(h_ri[i + seq_len])        # bước tiếp theo cần dự đoán

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ============================================================
# HÀM 4: Chuẩn hóa data về [-1, 1]
# ============================================================
def normalize(X, y):
    """
    Chuẩn hóa data để training ổn định hơn.
    Dùng max-scaling: chia cho giá trị tuyệt đối lớn nhất.

    Returns:
        X_norm, y_norm : data đã chuẩn hóa
        scale          : hệ số scale, dùng để khôi phục lại giá trị gốc
    """
    scale  = np.max(np.abs(X))
    X_norm = X / scale
    y_norm = y / scale
    return X_norm, y_norm, scale


# ============================================================
# HÀM 5: Pipeline đầy đủ - sinh, xử lý và lưu data
# ============================================================
def generate_and_save(velocity_kmh, save_dir='data'):
    """
    Pipeline đầy đủ cho 1 tốc độ:
      1. Tính Doppler
      2. Sinh kênh Rayleigh
      3. Tạo sequences
      4. Chuẩn hóa
      5. Chia train/test
      6. Lưu file .npz

    Args:
        velocity_kmh : tốc độ di chuyển (km/h)
        save_dir     : thư mục lưu data

    Returns:
        dict chứa X_train, y_train, X_test, y_test, h_raw
    """
    os.makedirs(save_dir, exist_ok=True)

    # Bước 1: Tính Doppler
    fd = compute_doppler(velocity_kmh)
    print(f"\n[v = {velocity_kmh:>3} km/h]  fd = {fd:>7.1f} Hz")

    # Bước 2: Sinh kênh
    h = generate_rayleigh_jakes(N, fd)

    # Bước 3: Tạo sequences
    X, y = create_sequences(h, SEQ_LEN)

    # Bước 4: Chia train/test (LÀM TRƯỚC)
    split   = int(len(X) * TRAIN_RATIO)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test   = X[split:], y[split:]

    # Bước 5: Chuẩn hóa 
    scale = np.max(np.abs(X_train))
    X_train = X_train / scale
    y_train = y_train / scale
    
    X_test  = X_test / scale
    y_test  = y_test / scale

    # Bước 6: Lưu file
    path = os.path.join(save_dir, f'channel_v{velocity_kmh}kmh.npz')
    np.savez(path,
             X_train=X_train, y_train=y_train,
             X_test=X_test,   y_test=y_test,
             h_raw=h,         scale=np.array([scale]),
             velocity=np.array([velocity_kmh]),
             fd=np.array([fd]))

    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Saved: {path}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test':  X_test,  'y_test':  y_test,
        'h_raw':   h,       'scale':   scale,
        'fd':      fd
    }


# ============================================================
# HÀM 6: Vẽ đồ thị kênh truyền
# ============================================================
def plot_channels(data_dict_list, velocities, n_show=500):
    """
    Vẽ biên độ kênh của các tốc độ khác nhau trên cùng 1 hình.
    Thấy rõ kênh nhanh hơn khi tốc độ tăng.
    """
    os.makedirs('results', exist_ok=True)

    fig, axes = plt.subplots(len(velocities), 1,
                              figsize=(12, 3 * len(velocities)),
                              sharex=True)
    if len(velocities) == 1:
        axes = [axes]

    colors = ['steelblue', 'darkorange', 'crimson']

    for ax, data, v, color in zip(axes, data_dict_list, velocities, colors):
        h    = data['h_raw']
        fd   = data['fd']
        t_ms = np.arange(n_show) / FS * 1000   # đổi sang ms

        ax.plot(t_ms, np.abs(h[:n_show]),
                color=color, linewidth=0.8, alpha=0.9)
        ax.set_ylabel('|h(t)|')
        ax.set_title(f'v = {v} km/h  |  fd = {fd:.0f} Hz',
                     fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel('Thời gian (ms)')
    plt.suptitle('Kênh Rayleigh Fading - So sánh các tốc độ', fontsize=13)
    plt.tight_layout()

    out = 'results/channel_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved: {out}")


# ============================================================
# CHẠY CHÍNH
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  Rayleigh Fading Channel Data Generator")
    print(f"  N={N} samples | seq_len={SEQ_LEN} | fs={FS} Hz")
    print("=" * 55)

    velocities   = [30, 120, 300]   # km/h: đô thị, cao tốc, tàu nhanh
    results_list = []

    for v in velocities:
        data = generate_and_save(v)
        results_list.append(data)

    # Vẽ đồ thị so sánh 3 tốc độ
    plot_channels(results_list, velocities)

    print("\nXong! Kiểm tra thư mục data/ và results/")
