import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.baselines import get_baseline
from models.proposed  import (
    CNNLSTMAttention,
    CNNLSTMNoAttention,
    LSTMAttention,
)
from models.classical import WienerFilter
import time as _time

# ============================================================
# CẤU HÌNH
# ============================================================
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VELOCITIES  = [30, 120, 300]
SEEDS       = [42, 123, 456]
SNR_RANGE   = list(range(0, 31, 5))

# --- Phân nhóm model cho từng loại biểu đồ ---
# Tất cả model có trong evaluation
ALL_MODELS       = ['Wiener', 'RNN', 'LSTM', 'GRU', 'Transformer',
                    'CNN-LSTM', 'LSTM-Attention', 'CNN-LSTM-Attention']

# Model cần load checkpoint (loại trừ Wiener — fit closed-form)
TRAINED_MODELS   = ['RNN', 'LSTM', 'GRU', 'Transformer',
                    'CNN-LSTM', 'LSTM-Attention', 'CNN-LSTM-Attention']

# Model trong biểu đồ NMSE chính (gọn gàng, không quá nhiều đường)
# Wiener (cổ điển) + 3 baseline AI mạnh nhất + Proposed
MAIN_PLOT_MODELS = ['Wiener', 'LSTM', 'GRU', 'Transformer', 'CNN-LSTM-Attention']

# Model trong Ablation Study (4 phiên bản đề chứng minh vai trò CNN/Attention)
ABLATION_MODELS  = ['LSTM', 'CNN-LSTM', 'LSTM-Attention', 'CNN-LSTM-Attention']

# Để tương thích ngược với code cũ (in summary table dùng full list)
MODEL_NAMES      = ALL_MODELS

COLORS = {
    'Wiener'            : '#7F8C8D',  # xám — baseline cổ điển
    'RNN'               : '#E74C3C',
    'LSTM'              : '#3498DB',
    'GRU'               : '#2ECC71',
    'Transformer'       : '#F39C12',
    'CNN-LSTM'          : '#1ABC9C',  # teal — ablation
    'LSTM-Attention'    : '#E67E22',  # cam — ablation
    'CNN-LSTM-Attention': '#9B59B6',  # tím — proposed
}
MARKERS = {
    'Wiener'            : 'x',
    'RNN'               : 'o',
    'LSTM'              : 's',
    'GRU'               : '^',
    'Transformer'       : 'D',
    'CNN-LSTM'          : 'v',
    'LSTM-Attention'    : 'p',
    'CNN-LSTM-Attention': '*',
}

CONFIG = {
    'hidden_size': 64,
    'num_layers' : 2,
    'data_dir'   : 'data',
    'ckpt_dir'   : 'results/checkpoints',
    'out_dir'    : 'results',
    'batch_size' : 256,
}


# ============================================================
# HÀM 1: Tạo model
# ============================================================
def build_model(name):
    """Tạo model theo tên (chưa load weights).
    Wiener không cần build ở đây — xử lý riêng vì fit closed-form.
    """
    h          = CONFIG['hidden_size']
    num_layers = CONFIG['num_layers']

    if name == 'CNN-LSTM-Attention':
        return CNNLSTMAttention(
            hidden_size=h, num_layers=num_layers,
            cnn_channels=32, num_heads=2
        )
    if name == 'CNN-LSTM':
        return CNNLSTMNoAttention(
            hidden_size=h, num_layers=num_layers, cnn_channels=32
        )
    if name == 'LSTM-Attention':
        return LSTMAttention(
            hidden_size=h, num_layers=num_layers, num_heads=2
        )
    if name == 'Wiener':
        # Trả về instance chưa fit — caller sẽ gọi .fit() với data tương ứng
        return WienerFilter(seq_len=50, horizon=1, ridge=1e-4)
    return get_baseline(name, h, num_layers)


# ============================================================
# HÀM 1b: Build + fit Wiener cho 1 (velocity, seed)
# ============================================================
def build_wiener(velocity, seed, horizon=1):
    """
    Fit Wiener filter trên train data của (velocity, seed).
    Mỗi (velocity, seed) cần 1 Wiener riêng vì autocorrelation thay đổi.
    """
    path = os.path.join(
        CONFIG['data_dir'],
        f'channel_v{velocity}kmh_s{seed}.npz'
    )
    d = np.load(path)
    w = WienerFilter(seq_len=50, horizon=horizon, ridge=1e-4)
    if horizon == 1:
        w.fit(d['X_train'], d['y_train'])
    else:
        # Multi-step: cần xây y_train với nhiều bước future từ h_raw
        h_raw = d['h_raw']
        scale = float(d['scale'][0])
        h_ri  = np.stack([h_raw.real, h_raw.imag], axis=1).astype(np.float32)
        seq_len = 50
        train_ratio = 0.8
        Xs, ys = [], []
        for i in range(len(h_ri) - seq_len - horizon + 1):
            Xs.append(h_ri[i:i+seq_len])
            ys.append(h_ri[i+seq_len:i+seq_len+horizon])
        Xs = np.array(Xs) / scale
        ys = np.array(ys) / scale
        split = int(len(Xs) * train_ratio)
        w.fit(Xs[:split], ys[:split])
    return w


# ============================================================
# HÀM 2: Load checkpoint
# ============================================================
def load_checkpoint(model, name, velocity, seed):
    path = os.path.join(
        CONFIG['ckpt_dir'],
        f'{name}_v{velocity}kmh_s{seed}.pth'
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy: {path}")
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# HÀM 3: Load test data
# ============================================================
def load_test_data(velocity, seed):
    path = os.path.join(
        CONFIG['data_dir'],
        f'channel_v{velocity}kmh_s{seed}.npz'
    )
    d = np.load(path)
    return torch.FloatTensor(d['X_test']), torch.FloatTensor(d['y_test'])


# ============================================================
# HÀM 4: Tính NMSE tại 1 mức SNR
# ============================================================
def compute_nmse_at_snr(model, X_test, y_test, snr_db):
    signal_power = torch.mean(X_test ** 2).item()
    noise_std    = np.sqrt(signal_power / (10 ** (snr_db / 10)))

    # Wiener filter — chạy thẳng trên CPU (numpy-based)
    if isinstance(model, WienerFilter):
        X_noisy = X_test + torch.randn_like(X_test) * noise_std
        pred    = model.predict(X_noisy).numpy()
        targets = y_test.numpy()
        error   = pred - targets
        nmse    = np.sum(error ** 2) / (np.sum(targets ** 2) + 1e-10)
        return 10 * np.log10(nmse + 1e-10)

    # Mô hình neural network — dùng DataLoader trên DEVICE
    loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=CONFIG['batch_size'], shuffle=False
    )

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            X_noisy = X_batch + torch.randn_like(X_batch) * noise_std
            if hasattr(model, 'predict'):
                pred = model.predict(X_noisy)
            else:
                pred = model(X_noisy)
            all_preds.append(pred.cpu())
            all_targets.append(y_batch)

    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    error   = preds - targets
    nmse    = np.sum(error**2) / (np.sum(targets**2) + 1e-10)
    return 10 * np.log10(nmse + 1e-10)


# ============================================================
# HÀM 5: Evaluate 1 model, 1 seed — toàn bộ SNR range
# ============================================================
def evaluate_one(name, velocity, seed):
    if name == 'Wiener':
        # Wiener: fit closed-form trên train data — không có checkpoint
        model = build_wiener(velocity, seed, horizon=1)
    else:
        model = build_model(name)
        model = load_checkpoint(model, name, velocity, seed)
    X, y  = load_test_data(velocity, seed)
    return [compute_nmse_at_snr(model, X, y, snr) for snr in SNR_RANGE]


# ============================================================
# HÀM 6: Mean ± std NMSE qua tất cả seeds
# ============================================================
def evaluate_mean_std(name, velocity):
    all_runs = [evaluate_one(name, velocity, s) for s in SEEDS]
    arr      = np.array(all_runs)   # (n_seeds, n_snr)
    return arr.mean(axis=0).tolist(), arr.std(axis=0).tolist()


# ============================================================
# HÀM 7: Vẽ NMSE vs SNR với error band (chart chính)
# ============================================================
def plot_nmse_vs_snr(results, model_list=None, suffix='main'):
    """
    Vẽ NMSE vs SNR cho 1 nhóm model.

    Args:
        results    : dict[velocity][model] = (mean_list, std_list)
        model_list : list model cần vẽ (mặc định MAIN_PLOT_MODELS)
        suffix     : suffix tên file output ('main', 'ablation', 'all')
    """
    if model_list is None:
        model_list = MAIN_PLOT_MODELS

    os.makedirs(CONFIG['out_dir'], exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles    = ['v = 30 km/h\n(Đô thị)',
                 'v = 120 km/h\n(Cao tốc)',
                 'v = 300 km/h\n(Tàu cao tốc)']

    for ax, velocity, title in zip(axes, VELOCITIES, titles):
        for name in model_list:
            if name not in results[velocity]:
                continue                                    # bỏ qua nếu chưa eval
            mean_v, std_v = results[velocity][name]
            mean_arr = np.array(mean_v)
            std_arr  = np.array(std_v)
            lw   = 2.5 if name == 'CNN-LSTM-Attention' else 1.5
            ls   = '-'  if name == 'CNN-LSTM-Attention' else '--'

            ax.plot(SNR_RANGE, mean_arr,
                    color=COLORS[name], marker=MARKERS[name],
                    linewidth=lw, linestyle=ls,
                    markersize=7, label=name)
            ax.fill_between(SNR_RANGE,
                            mean_arr - std_arr,
                            mean_arr + std_arr,
                            color=COLORS[name], alpha=0.12)

        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('NMSE (dB)', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.suptitle(
        f'NMSE vs SNR — Mean ± Std qua {len(SEEDS)} seeds\n'
        '(Số càng âm càng tốt)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    out = os.path.join(CONFIG['out_dir'], f'nmse_vs_snr_{suffix}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# HÀM 7b: Vẽ Ablation Study — 4 phiên bản LSTM/CNN-LSTM/LSTM-Attn/Full
# ============================================================
def plot_ablation_study(results):
    """
    Biểu đồ ablation: chứng minh từng thành phần (CNN, Attention)
    đóng góp gì vào NMSE so với mô hình đầy đủ.

    4 phiên bản:
      [1] LSTM            — baseline thuần
      [2] CNN-LSTM        — thêm CNN, bỏ Attention
      [3] LSTM-Attention  — thêm Attention, bỏ CNN
      [4] CNN-LSTM-Attention — đầy đủ (đề xuất)

    Câu chuyện kỳ vọng: [4] < [2], [3] < [1] tại high-mobility
    """
    os.makedirs(CONFIG['out_dir'], exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles    = ['v = 30 km/h', 'v = 120 km/h', 'v = 300 km/h']

    for ax, velocity, title in zip(axes, VELOCITIES, titles):
        for name in ABLATION_MODELS:
            if name not in results[velocity]:
                continue
            mean_v, std_v = results[velocity][name]
            mean_arr = np.array(mean_v)
            std_arr  = np.array(std_v)
            lw   = 2.5 if name == 'CNN-LSTM-Attention' else 1.6
            ls   = '-'  if name == 'CNN-LSTM-Attention' else '--'

            ax.plot(SNR_RANGE, mean_arr,
                    color=COLORS[name], marker=MARKERS[name],
                    linewidth=lw, linestyle=ls,
                    markersize=7, label=name)
            ax.fill_between(SNR_RANGE,
                            mean_arr - std_arr,
                            mean_arr + std_arr,
                            color=COLORS[name], alpha=0.10)

        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('NMSE (dB)', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.suptitle(
        'Ablation Study — Vai trò của CNN và Attention\n'
        '(So sánh 4 phiên bản với mean ± std qua 3 seeds)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    out = os.path.join(CONFIG['out_dir'], 'ablation_study.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# HÀM 7c: Vẽ NMSE Gain so với Wiener — chart "key insight"
# ============================================================
def plot_nmse_gain_over_wiener(results, snr_target=20):
    """
    Bar chart: NMSE_AI - NMSE_Wiener (dB) tại SNR=20dB cho từng velocity.

    Câu chuyện:
      - Số ÂM = AI tốt hơn Wiener (AI gain)
      - Số DƯƠNG = AI tệ hơn Wiener (rất hiếm, chỉ ở v=30 nếu AI overfit)
      - Tại v=30 km/h: gain nhỏ (Wiener đủ tốt với coherent channel)
      - Tại v=300 km/h: gain LỚN (AI vượt rõ vì Wiener stationary giả định fail)

    Đây là chart "đắt giá" trả lời trực quan câu:
      "Tại sao cần AI? Wiener không đủ à?"
    """
    if 'Wiener' not in results[VELOCITIES[0]]:
        print("[SKIP] plot_nmse_gain_over_wiener — chưa có kết quả Wiener.")
        return

    os.makedirs(CONFIG['out_dir'], exist_ok=True)
    idx = SNR_RANGE.index(snr_target)

    # Model AI cần so sánh (loại Wiener vì là baseline)
    ai_models = [m for m in MODEL_NAMES if m != 'Wiener']

    # Tính gain matrix: rows = velocity, cols = model
    gain_matrix = np.zeros((len(VELOCITIES), len(ai_models)))
    std_matrix  = np.zeros((len(VELOCITIES), len(ai_models)))
    for i, v in enumerate(VELOCITIES):
        wiener_mean = results[v]['Wiener'][0][idx]
        for j, name in enumerate(ai_models):
            if name not in results[v]:
                gain_matrix[i, j] = np.nan
                continue
            ai_mean = results[v][name][0][idx]
            ai_std  = results[v][name][1][idx]
            gain_matrix[i, j] = ai_mean - wiener_mean    # số âm = AI tốt hơn
            std_matrix[i, j]  = ai_std

    # ----------------------------------------------------------
    # Vẽ
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6))
    n_models = len(ai_models)
    n_vel    = len(VELOCITIES)
    bar_w    = 0.8 / n_models
    x        = np.arange(n_vel)

    for j, name in enumerate(ai_models):
        offset = (j - (n_models - 1) / 2) * bar_w
        bars = ax.bar(x + offset, gain_matrix[:, j], bar_w,
                      yerr=std_matrix[:, j],
                      color=COLORS[name],
                      edgecolor='white', linewidth=0.8,
                      capsize=3,
                      label=name,
                      alpha=0.95 if name == 'CNN-LSTM-Attention' else 0.75)

        # Annotate giá trị lên đầu mỗi cột
        for i, b in enumerate(bars):
            h = b.get_height()
            if not np.isnan(h):
                va = 'top' if h < 0 else 'bottom'
                offset_text = -0.15 if h < 0 else 0.15
                ax.text(b.get_x() + b.get_width() / 2, h + offset_text,
                        f'{h:+.1f}',
                        ha='center', va=va, fontsize=7.5,
                        color=COLORS[name], fontweight='bold')

    # Đường zero — trên đường này = AI bằng Wiener
    ax.axhline(y=0, color='black', linewidth=1.2, linestyle='-', alpha=0.7)
    ax.text(n_vel - 0.5, 0.3,
            'Wiener baseline',
            fontsize=9, color='gray', style='italic')

    # Ghi chú vùng AI thắng / thua
    ax.text(-0.45, ax.get_ylim()[0] * 0.85 if ax.get_ylim()[0] < 0 else -0.5,
            '↓ AI tốt hơn Wiener\n   (gain âm)',
            fontsize=9, color='#27AE60', fontweight='bold',
            verticalalignment='bottom')

    ax.set_xticks(x)
    ax.set_xticklabels([f'v = {v} km/h' for v in VELOCITIES], fontsize=11)
    ax.set_ylabel(f'NMSE Gain so với Wiener (dB)\n'
                  f'(âm = AI tốt hơn) tại SNR = {snr_target} dB',
                  fontsize=11)
    ax.set_title(
        'Vai trò của AI so với Bộ lọc Wiener cổ điển\n'
        '(Càng âm càng chứng tỏ AI vượt Wiener; gap lớn nhất ở high-mobility)',
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=9, loc='best', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = os.path.join(CONFIG['out_dir'], 'nmse_gain_over_wiener.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# HÀM 8: Bảng tổng kết tại SNR = 20 dB
# ============================================================
def print_summary_table(results, snr_target=20):
    idx = SNR_RANGE.index(snr_target)
    print("\n" + "=" * 75)
    print(f"  NMSE (dB) tại SNR = {snr_target} dB — "
          f"Mean +/- Std qua {len(SEEDS)} seeds")
    print("=" * 75)
    print(f"  {'Model':<22} | {'30 km/h':>16} | "
          f"{'120 km/h':>16} | {'300 km/h':>16}")
    print("  " + "-" * 73)

    for name in MODEL_NAMES:
        if any(name not in results[v] for v in VELOCITIES):
            continue                                       # bỏ model chưa eval
        mk  = " *" if name == 'CNN-LSTM-Attention' else "  "
        row = f"  {name:<22}{mk}|"
        for v in VELOCITIES:
            m, s = results[v][name]
            row += f" {m[idx]:>6.2f}+/-{s[idx]:>4.2f} dB |"
        print(row)

    print("=" * 75)
    print("  * = Model đề xuất")


# ============================================================
# HÀM 8b: Lưu bảng kết quả ra CSV để paste vào báo cáo Word
# ============================================================
def save_summary_csv(results, complexity=None):
    """
    Xuất 2 file CSV để Nhân copy-paste thẳng vào báo cáo:
      [1] results/nmse_summary.csv  -- NMSE mean ± std × velocity × SNR
      [2] results/complexity_summary.csv  -- params/FLOPs/latency

    Định dạng CSV với delimiter ',' và quoting để mở thẳng trong Excel/Word.
    Số được format dạng "mean ± std" để Word hiển thị đẹp.
    """
    import csv
    os.makedirs(CONFIG['out_dir'], exist_ok=True)

    # ----------------------------------------------------------
    # [1] BẢNG NMSE — hàng = model, cột = (velocity, SNR)
    # ----------------------------------------------------------
    nmse_path = os.path.join(CONFIG['out_dir'], 'nmse_summary.csv')
    with open(nmse_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)

        # Header 1: nhóm theo velocity
        header1 = ['Model']
        for v in VELOCITIES:
            header1.extend([f'v={v} km/h'] * len(SNR_RANGE))
        w.writerow(header1)

        # Header 2: SNR cho từng velocity
        header2 = ['']
        for _ in VELOCITIES:
            header2.extend([f'SNR={s}dB' for s in SNR_RANGE])
        w.writerow(header2)

        # Data: mỗi hàng = 1 model
        for name in MODEL_NAMES:
            if any(name not in results[v] for v in VELOCITIES):
                continue                                   # bỏ model chưa eval
            row = [name + (' *' if name == 'CNN-LSTM-Attention' else '')]
            for v in VELOCITIES:
                m, s = results[v][name]
                for i in range(len(SNR_RANGE)):
                    row.append(f'{m[i]:.2f} ± {s[i]:.2f}')
            w.writerow(row)

    print(f"Saved: {nmse_path}")

    # ----------------------------------------------------------
    # [2] BẢNG NMSE TẠI SNR=20dB — gọn, dùng làm bảng chính báo cáo
    # ----------------------------------------------------------
    snr20_path = os.path.join(CONFIG['out_dir'], 'nmse_summary_snr20.csv')
    with open(snr20_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Model'] + [f'v={v} km/h (NMSE dB)' for v in VELOCITIES])
        idx = SNR_RANGE.index(20)
        for name in MODEL_NAMES:
            if any(name not in results[v] for v in VELOCITIES):
                continue
            row = [name + (' *' if name == 'CNN-LSTM-Attention' else '')]
            for v in VELOCITIES:
                m, s = results[v][name]
                row.append(f'{m[idx]:.2f} ± {s[idx]:.2f}')
            w.writerow(row)
    print(f"Saved: {snr20_path}")

    # ----------------------------------------------------------
    # [3] BẢNG COMPLEXITY (nếu có)
    # ----------------------------------------------------------
    if complexity is None:
        return

    comp_path = os.path.join(CONFIG['out_dir'], 'complexity_summary.csv')
    with open(comp_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Model', 'Parameters', 'FLOPs',
                    'CPU latency (ms)', 'GPU latency (ms)'])
        for name, info in complexity.items():
            lat_gpu = f"{info['lat_gpu']:.3f}" if info['lat_gpu'] is not None \
                      else 'N/A'
            w.writerow([
                name + (' *' if name == 'CNN-LSTM-Attention' else ''),
                f"{info['params']:,}",
                info['flops_str'],
                f"{info['lat_cpu']:.3f}",
                lat_gpu,
            ])
    print(f"Saved: {comp_path}")


# ============================================================
# HÀM 9: Time series (seed=42 đại diện)
# ============================================================
def plot_timeseries(velocity=300, snr_db=20, seed=42, n_show=200):
    os.makedirs(CONFIG['out_dir'], exist_ok=True)
    X_test, y_test = load_test_data(velocity, seed)
    signal_power   = torch.mean(X_test**2).item()
    noise_std      = np.sqrt(signal_power / (10**(snr_db/10)))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    y_true     = y_test[:n_show].numpy()
    h_true_amp = np.sqrt(y_true[:,0]**2 + y_true[:,1]**2)

    for ax, name, color in zip(
        axes,
        ['LSTM', 'CNN-LSTM-Attention'],
        ['#3498DB', '#9B59B6']
    ):
        model = build_model(name)
        model = load_checkpoint(model, name, velocity, seed)
        X_s   = X_test[:n_show].to(DEVICE)
        X_n   = X_s + torch.randn_like(X_s) * noise_std

        with torch.no_grad():
            pred = model.predict(X_n) if hasattr(model, 'predict') \
                   else model(X_n)

        pred_np    = pred.cpu().numpy()
        h_pred_amp = np.sqrt(pred_np[:,0]**2 + pred_np[:,1]**2)
        t          = np.arange(n_show)
        err        = pred_np - y_true
        nmse       = 10 * np.log10(
            np.sum(err**2) / (np.sum(y_true**2) + 1e-10))

        ax.plot(t, h_true_amp, color='#2ECC71', linewidth=1.2,
                label='Kênh thật |h(t)|', alpha=0.9)
        ax.plot(t, h_pred_amp, color=color, linewidth=1.2,
                linestyle='--', label=f'{name} (dự đoán)', alpha=0.9)
        ax.fill_between(t, h_true_amp, h_pred_amp,
                        alpha=0.15, color=color)
        ax.set_ylabel('Biên độ |h(t)|', fontsize=11)
        ax.set_title(
            f'{name} — v={velocity} km/h, SNR={snr_db} dB'
            f' | NMSE = {nmse:.2f} dB', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel('Bước thời gian', fontsize=11)
    plt.suptitle(
        f'Dự đoán kênh — v={velocity} km/h | SNR={snr_db} dB | '
        f'seed={seed} (đại diện)',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    out = os.path.join(CONFIG['out_dir'],
                       f'timeseries_v{velocity}kmh.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()                                           # tránh block trên Windows
    print(f"Saved: {out}")


# ============================================================
# HÀM 10: Attention weights
# ============================================================
def plot_attention_weights(velocity=300, seed=42, n_samples=3):
    os.makedirs(CONFIG['out_dir'], exist_ok=True)
    X_test, _ = load_test_data(velocity, seed)
    model     = build_model('CNN-LSTM-Attention')
    model     = load_checkpoint(model, 'CNN-LSTM-Attention', velocity, seed)

    with torch.no_grad():
        _, weights = model(X_test[:n_samples].to(DEVICE))
    weights = weights.cpu().numpy()

    fig, axes = plt.subplots(1, n_samples, figsize=(14, 3))
    for i, ax in enumerate(axes):
        ax.bar(range(len(weights[i])), weights[i],
               color='#9B59B6', alpha=0.7)
        ax.set_xlabel('Bước thời gian', fontsize=10)
        ax.set_ylabel('Attention weight', fontsize=10)
        ax.set_title(f'Mẫu {i+1} — v={velocity} km/h', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        'Attention Weights — CNN-LSTM-Attention\n'
        'Bước thời gian nào được model chú ý nhiều nhất',
        fontsize=12
    )
    plt.tight_layout()
    out = os.path.join(CONFIG['out_dir'], 'attention_weights.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()                                           # tránh block trên Windows
    print(f"Saved: {out}")

# ============================================================
# HÀM 11: Đo số Parameters và FLOPs
# ============================================================
def compute_model_complexity():
    """
    Đo complexity của tất cả model:
      - Số parameters
      - FLOPs (floating point operations) mỗi inference
      - Latency trên CPU và GPU
    Wiener filter được xử lý đặc biệt vì không phải nn.Module.
    """
    try:
        from thop import profile, clever_format
        HAS_THOP = True
    except ImportError:
        HAS_THOP = False
        print("  [WARN] thop chua cai. Chay: pip install thop")
        print("  [WARN] Se bo qua FLOPs, chi do params va latency")

    dummy_input = torch.randn(1, 50, 2)
    results     = {}

    print("\n" + "=" * 70)
    print("  COMPLEXITY ANALYSIS")
    print("=" * 70)
    print(f"  {'Model':<22} | {'Params':>10} | {'FLOPs':>12} | "
          f"{'CPU (ms)':>10} | {'GPU (ms)':>10}")
    print("  " + "-" * 68)

    for name in MODEL_NAMES:
        # ---------- Wiener: xử lý riêng ----------
        if name == 'Wiener':
            # Fit nhanh trên 1 (velocity, seed) đại diện để có .W
            w_eval = build_wiener(velocity=120, seed=42, horizon=1)
            n_params = w_eval.num_params()
            flops    = n_params * 2                          # 1 mul + 1 add mỗi tham số
            flops_str = f"{flops:,}"

            # Latency CPU (dùng numpy)
            x_cpu = dummy_input
            for _ in range(10):
                _ = w_eval.predict(x_cpu)
            N_RUNS = 100
            t0 = _time.perf_counter()
            for _ in range(N_RUNS):
                _ = w_eval.predict(x_cpu)
            lat_cpu     = ((_time.perf_counter() - t0) / N_RUNS) * 1000
            lat_gpu     = None
            lat_gpu_str = "N/A"

            results[name] = {
                'params'   : n_params,
                'flops'    : flops,
                'flops_str': flops_str,
                'lat_cpu'  : lat_cpu,
                'lat_gpu'  : lat_gpu,
            }
            print(f"  {name:<22}  | {n_params:>10,} | {flops_str:>12} | "
                  f"{lat_cpu:>9.3f} | {lat_gpu_str:>10}")
            continue

        # ---------- Neural network (nn.Module) ----------
        model = build_model(name).eval()
        n_params = sum(p.numel() for p in model.parameters())

        if HAS_THOP:
            try:
                flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
                flops_str, _ = clever_format([flops], "%.2f")
            except Exception:
                flops     = 0
                flops_str = "N/A"
        else:
            flops     = 0
            flops_str = "N/A"

        # Latency CPU
        model_cpu = model.cpu()
        x_cpu     = dummy_input.cpu()
        for _ in range(10):
            with torch.no_grad():
                _ = model_cpu.predict(x_cpu) if hasattr(model_cpu, 'predict') \
                    else model_cpu(x_cpu)
        N_RUNS = 100
        t0 = _time.perf_counter()
        for _ in range(N_RUNS):
            with torch.no_grad():
                _ = model_cpu.predict(x_cpu) if hasattr(model_cpu, 'predict') \
                    else model_cpu(x_cpu)
        lat_cpu = ((_time.perf_counter() - t0) / N_RUNS) * 1000

        # Latency GPU
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            x_gpu     = dummy_input.cuda()
            for _ in range(10):
                with torch.no_grad():
                    _ = model_gpu.predict(x_gpu) if hasattr(model_gpu, 'predict') \
                        else model_gpu(x_gpu)
            torch.cuda.synchronize()
            t0 = _time.perf_counter()
            for _ in range(N_RUNS):
                with torch.no_grad():
                    _ = model_gpu.predict(x_gpu) if hasattr(model_gpu, 'predict') \
                        else model_gpu(x_gpu)
            torch.cuda.synchronize()
            lat_gpu = ((_time.perf_counter() - t0) / N_RUNS) * 1000
            lat_gpu_str = f"{lat_gpu:.3f}"
        else:
            lat_gpu     = None
            lat_gpu_str = "N/A"

        results[name] = {
            'params'   : n_params,
            'flops'    : flops,
            'flops_str': flops_str,
            'lat_cpu'  : lat_cpu,
            'lat_gpu'  : lat_gpu,
        }

        star = " *" if name == 'CNN-LSTM-Attention' else "  "
        print(f"  {name:<22}{star}| {n_params:>10,} | {flops_str:>12} | "
              f"{lat_cpu:>9.3f} | {lat_gpu_str:>10}")

    print("=" * 70)
    print("  * = Model đề xuất")
    return results
 
 
# ============================================================
# HÀM 12: Vẽ NMSE vs Parameters
# ============================================================
def plot_nmse_vs_params(complexity_results, nmse_results,
                        velocity=120, snr_target=20):
    """
    Scatter plot: trục X = số params, trục Y = NMSE tại SNR=20dB.
    Chứng minh CNN-LSTM-Attention đạt accuracy tốt với complexity hợp lý
    (vị trí góc trên-trái = ít params nhưng NMSE tốt = lý tưởng nhất).
    """
    os.makedirs(CONFIG['out_dir'], exist_ok=True)
 
    snr_range = list(range(0, 31, 5))
    idx       = snr_range.index(snr_target)
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    # --- Plot 1: Scatter NMSE vs Params tại 1 velocity ---
    ax = axes[0]
    for name in MODEL_NAMES:
        params = complexity_results[name]['params']
        # Lấy mean NMSE qua 3 seeds
        mean_v, std_v = nmse_results[velocity][name]
        nmse_val = mean_v[idx]
        std_val  = std_v[idx]
 
        ms   = 180 if name == 'CNN-LSTM-Attention' else 100
        zord = 5   if name == 'CNN-LSTM-Attention' else 3
        mk   = '*' if name == 'CNN-LSTM-Attention' else 'o'
 
        ax.scatter(params / 1000, nmse_val,
                   c=COLORS[name], s=ms,
                   marker=mk, zorder=zord,
                   label=name, edgecolors='white', linewidths=0.8)
 
        ax.errorbar(params / 1000, nmse_val, yerr=std_val,
                    color=COLORS[name], fmt='none',
                    capsize=4, linewidth=1.2, zorder=zord - 1)
 
        # Annotate
        offset = (5, 5) if name != 'CNN-LSTM-Attention' else (5, -12)
        ax.annotate(name.replace('CNN-LSTM-Attention', 'Proposed ★'),
                    xy=(params / 1000, nmse_val),
                    xytext=offset, textcoords='offset points',
                    fontsize=8, color=COLORS[name])
 
    ax.set_xlabel('Số tham số (nghìn)', fontsize=12)
    ax.set_ylabel(f'NMSE (dB) tại SNR={snr_target} dB', fontsize=12)
    ax.set_title(f'NMSE vs Parameters  ·  v={velocity} km/h\n'
                 f'(Góc trên-trái = ít params, NMSE tốt = lý tưởng)',
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
 
    # --- Plot 2: Bar chart latency ---
    ax2 = axes[1]
    names    = list(complexity_results.keys())
    lat_cpus = [complexity_results[n]['lat_cpu'] for n in names]
    lat_gpus = [complexity_results[n]['lat_gpu'] or 0 for n in names]
    colors   = [COLORS[n] for n in names]
 
    x    = np.arange(len(names))
    w    = 0.35
    bars1 = ax2.bar(x - w/2, lat_cpus, w, label='CPU',
                    color=colors, alpha=0.7, edgecolor='white')
    ax2.bar(x + w/2, lat_gpus, w, label='GPU',
                    color=colors, alpha=0.4, edgecolor='white',
                    linestyle='--')
 
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.replace('CNN-LSTM-Attention', 'Proposed')
                         for n in names], rotation=15, ha='right', fontsize=9)
    ax2.set_ylabel('Latency (ms)', fontsize=12)
    ax2.set_title('Inference Latency\nCPU vs GPU (1 sample)', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
 
    for bar in bars1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                 f'{h:.2f}', ha='center', va='bottom', fontsize=8)
 
    plt.suptitle('Complexity Analysis — Parameters & Latency',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
 
    out = os.path.join(CONFIG['out_dir'], 'complexity_analysis.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()                                           # tránh block trên Windows
    print(f"Saved: {out}")
 
 
# ============================================================
# HÀM 13 (FIXED): Tính BER tại 1 mức SNR
# ============================================================
def compute_ber_at_snr(model, X_test, y_test, snr_db, n_symbols=5000):
    """
    Tính BER với QPSK — đúng về mặt channel prediction.
 
    Quy trình:
      1. Lấy n_symbols mẫu từ test set
      2. Model dự đoán kênh tương lai: h_pred = model(X_test[i])
      3. Kênh thật tương lai:          h_true = y_test[i]
      4. Sinh QPSK symbol ngẫu nhiên
      5. Truyền qua kênh THẬT h_true + nhiễu AWGN
      6. Equalize bằng h_pred (Predicted CSI)
      7. Equalize bằng h_true (Perfect CSI) — để so sánh
      8. Demodulate → đếm bit lỗi
 
    Args:
        model     : đã load weights, đã eval()
        X_test    : (N, seq_len, 2) — input sequences
        y_test    : (N, 2) — ground truth h(t+1), real+imag
        snr_db    : mức SNR (dB)
        n_symbols : số symbol test (mặc định 5000)
 
    Returns:
        ber_predicted : BER khi dùng model dự đoán kênh
        ber_perfect   : BER khi biết kênh hoàn hảo (lower bound)
    """
    n_symbols = min(n_symbols, len(X_test))
    np.random.seed(42)
 
    # ----------------------------------------------------------
    # Bước 1: Dự đoán kênh h(t+1) bằng model
    # ----------------------------------------------------------
    X_sample = X_test[:n_symbols]
 
    # Thêm nhiễu vào input (giống lúc evaluate NMSE)
    signal_power = torch.mean(X_sample ** 2).item()
    snr_linear   = 10 ** (snr_db / 10)
    noise_std_in = np.sqrt(signal_power / snr_linear)
    X_noisy      = X_sample + torch.randn_like(X_sample) * noise_std_in
 
    with torch.no_grad():
        X_gpu = X_noisy.to(DEVICE)
        if hasattr(model, 'predict'):
            pred = model.predict(X_gpu).cpu().numpy()   # (n_symbols, 2)
        else:
            pred = model(X_gpu).cpu().numpy()
 
    # Kênh dự đoán: h_pred = pred[:, 0] + j*pred[:, 1]
    h_pred_real = pred[:, 0]   # real part của h(t+1) dự đoán
    h_pred_imag = pred[:, 1]   # imag part của h(t+1) dự đoán
 
    # Kênh thật: h_true = y_test[:, 0] + j*y_test[:, 1]
    y_np         = y_test[:n_symbols].numpy()
    h_true_real  = y_np[:, 0]   # real part của h(t+1) thật
    h_true_imag  = y_np[:, 1]   # imag part của h(t+1) thật
 
    # ----------------------------------------------------------
    # Bước 2: Sinh QPSK symbols ngẫu nhiên
    # ----------------------------------------------------------
    # QPSK: mỗi symbol = 2 bits
    # Map: bit0=0→+1, bit0=1→-1 (trục I)
    #      bit1=0→+1, bit1=1→-1 (trục Q)
    # Normalize: chia sqrt(2) để công suất = 1
    bits = np.random.randint(0, 2, size=(n_symbols, 2))
    s_I  = (1 - 2 * bits[:, 0]) / np.sqrt(2)   # ±1/√2
    s_Q  = (1 - 2 * bits[:, 1]) / np.sqrt(2)   # ±1/√2
 
    # ----------------------------------------------------------
    # Bước 3: Truyền qua kênh THẬT h_true (fading + AWGN)
    # ----------------------------------------------------------
    # Mô hình: r = h * s + n
    # Với h phức: r_I = h_r*s_I - h_i*s_Q + n_I
    #             r_Q = h_r*s_Q + h_i*s_I + n_Q
    noise_std_ch = np.sqrt(1.0 / (2 * snr_linear))   # AWGN noise std
 
    n_I = np.random.randn(n_symbols) * noise_std_ch
    n_Q = np.random.randn(n_symbols) * noise_std_ch
 
    # Tín hiệu nhận được
    r_I = h_true_real * s_I - h_true_imag * s_Q + n_I
    r_Q = h_true_real * s_Q + h_true_imag * s_I + n_Q
 
    # ----------------------------------------------------------
    # Bước 4: Equalization — Zero-Forcing
    # ----------------------------------------------------------
    # ZF: ĥ* · r / |ĥ|²
    # Với ĥ = h_r + j*h_i:
    #   eq_I = (h_r*r_I + h_i*r_Q) / (h_r² + h_i²)
    #   eq_Q = (h_r*r_Q - h_i*r_I) / (h_r² + h_i²)
 
    def zf_equalize(r_i, r_q, h_r, h_i):
        h_pow = h_r**2 + h_i**2
        # Tránh chia cho 0 ở những mẫu kênh yếu
        h_pow = np.maximum(h_pow, 1e-8)
        eq_i  = (h_r * r_i + h_i * r_q) / h_pow
        eq_q  = (h_r * r_q - h_i * r_i) / h_pow
        return eq_i, eq_q
 
    # Predicted CSI equalization
    eq_I_pred, eq_Q_pred = zf_equalize(r_I, r_Q,
                                        h_pred_real, h_pred_imag)
 
    # Perfect CSI equalization (lower bound)
    eq_I_perf, eq_Q_perf = zf_equalize(r_I, r_Q,
                                        h_true_real, h_true_imag)
 
    # ----------------------------------------------------------
    # Bước 5: QPSK Demodulation — hard decision
    # ----------------------------------------------------------
    # QPSK: bit=0 nếu eq > 0, bit=1 nếu eq < 0
    def qpsk_demod(eq_i, eq_q):
        b0 = (eq_i < 0).astype(int)   # bit 0
        b1 = (eq_q < 0).astype(int)   # bit 1
        return np.stack([b0, b1], axis=1)   # (n_symbols, 2)
 
    bits_pred = qpsk_demod(eq_I_pred, eq_Q_pred)
    bits_perf = qpsk_demod(eq_I_perf, eq_Q_perf)
 
    # ----------------------------------------------------------
    # Bước 6: Tính BER
    # ----------------------------------------------------------
    ber_predicted = np.mean(bits_pred != bits)
    ber_perfect   = np.mean(bits_perf != bits)
 
    return ber_predicted, ber_perfect
 
 
# ============================================================
# HÀM 14 (FIXED): Tính BER qua toàn bộ SNR range
# ============================================================
def evaluate_ber(velocity, seed=42, model_list=None):
    """
    Tính BER của các model tại các mức SNR.
    Dùng seed=42 làm đại diện.

    Args:
        model_list : danh sách model cần eval (mặc định = MAIN_PLOT_MODELS
                     để chart không quá rối)

    Returns:
        ber_results : dict[model_name] = list BER theo SNR
        ber_perfect : list BER Perfect CSI theo SNR
        SNR_BER     : list SNR values
    """
    SNR_BER = list(range(0, 31, 5))   # 0, 5, 10, ..., 30 dB
    if model_list is None:
        model_list = MAIN_PLOT_MODELS

    X_test, y_test = load_test_data(velocity, seed)
    ber_results    = {}
    ber_perfect    = None

    print(f"\n--- BER Evaluation | v={velocity} km/h | seed={seed} ---")
    print(f"  {'Model':<22} | {'BER@0dB':>8} | {'BER@10dB':>9} | "
          f"{'BER@20dB':>9} | {'BER@30dB':>9}")
    print("  " + "-" * 62)

    for name in model_list:
        # Wiener filter — fit closed-form, không có checkpoint
        if name == 'Wiener':
            model = build_wiener(velocity, seed, horizon=1)
        else:
            model = build_model(name)
            try:
                model = load_checkpoint(model, name, velocity, seed)
            except FileNotFoundError:
                print(f"  [SKIP] {name}: chưa có checkpoint")
                continue

        ber_list  = []
        perf_list = []

        for snr in SNR_BER:
            ber_p, ber_perf = compute_ber_at_snr(
                model, X_test, y_test, snr, n_symbols=5000
            )
            ber_list.append(ber_p)
            perf_list.append(ber_perf)

        ber_results[name] = ber_list

        # Perfect CSI chỉ cần tính 1 lần (dùng y_test, không phụ thuộc model)
        if ber_perfect is None:
            ber_perfect = perf_list

        idxs = [SNR_BER.index(s) for s in [0, 10, 20, 30]]
        mk   = " *" if name == 'CNN-LSTM-Attention' else "  "
        print(f"  {name:<22}{mk}| "
              f"{ber_list[idxs[0]]:.4f} | "
              f"{ber_list[idxs[1]]:.6f} | "
              f"{ber_list[idxs[2]]:.6f} | "
              f"{ber_list[idxs[3]]:.6f}")
 
    print(f"  {'Perfect CSI':<24}| "
          f"{ber_perfect[0]:.4f} | "
          f"{ber_perfect[2]:.6f} | "
          f"{ber_perfect[4]:.6f} | "
          f"{ber_perfect[6]:.6f}")
    print("  " + "-" * 62)
 
    return ber_results, ber_perfect, SNR_BER
 
 
# ============================================================
# HÀM 15 (FIXED): Vẽ BER vs SNR
# ============================================================
def plot_ber_vs_snr(velocity=120, seed=42):
    """
    Vẽ BER vs SNR cho tất cả model.
 
    Đường Perfect CSI = lower bound (tốt nhất có thể đạt được).
    Model nào BER càng gần Perfect CSI càng tốt.
    """
    os.makedirs(CONFIG['out_dir'], exist_ok=True)
 
    ber_results, ber_perfect, SNR_BER = evaluate_ber(velocity, seed)
 
    fig, ax = plt.subplots(figsize=(10, 6))
 
    # --- Perfect CSI (lower bound) ---
    ax.semilogy(SNR_BER, ber_perfect,
                color='#95A5A6', linewidth=2.5,
                linestyle=':', marker='x', markersize=8,
                label='Perfect CSI (lower bound)', zorder=6)
 
    # --- Từng model (chỉ vẽ những model có trong ber_results) ---
    for name in ber_results.keys():
        lw   = 2.5 if name == 'CNN-LSTM-Attention' else 1.5
        ls   = '-'  if name == 'CNN-LSTM-Attention' else '--'
        zord = 5    if name == 'CNN-LSTM-Attention' else 3

        ax.semilogy(SNR_BER, ber_results[name],
                    color=COLORS[name], linewidth=lw,
                    linestyle=ls, marker=MARKERS[name],
                    markersize=7, label=name, zorder=zord)
 
    # Ngưỡng BER = 1%
    ax.axhline(y=0.01, color='#E74C3C', linewidth=1.2,
               linestyle='-.', alpha=0.7,
               label='BER = 1% threshold')
 
    # Tìm SNR để đạt BER < 1% cho model đề xuất (nếu có)
    if 'CNN-LSTM-Attention' in ber_results:
        ber_proposed = ber_results['CNN-LSTM-Attention']
        for i, (snr, ber) in enumerate(zip(SNR_BER, ber_proposed)):
            if ber < 0.01:
                ax.annotate(f'BER<1%\n@ {snr}dB',
                            xy=(snr, ber),
                            xytext=(snr + 1, ber * 3),
                            color=COLORS['CNN-LSTM-Attention'],
                            fontsize=9, fontweight='bold',
                            arrowprops=dict(
                                arrowstyle='->', lw=1.2,
                                color=COLORS['CNN-LSTM-Attention']
                            ))
                break

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title(
        f'Bit Error Rate vs SNR  -  QPSK  -  v={velocity} km/h\n'
        f'Channel Prediction + ZF Equalization  -  seed={seed}\n'
        f'(Đường càng thấp càng tốt — càng gần Perfect CSI càng tốt)',
        fontsize=11
    )
    ax.grid(True, alpha=0.35, which='both')
    ax.legend(fontsize=9, loc='lower left', framealpha=0.9)
    ax.set_ylim(bottom=1e-4, top=1.0)
    ax.set_xlim(left=-1, right=31)

    plt.tight_layout()

    out = os.path.join(CONFIG['out_dir'],
                       f'ber_vs_snr_v{velocity}kmh.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

    return ber_results, ber_perfect


# ============================================================
# HÀM 15b: Vẽ BER cho 3 velocity trên cùng 1 figure (cho báo cáo)
# ============================================================
def plot_ber_combined(velocities=None, seed=42):
    """
    Gộp BER 3 velocity vào 1 figure 1x3 subplots.
    Phù hợp paste vào báo cáo (tiết kiệm trang).

    Returns:
        all_ber : dict[velocity] = (ber_results, ber_perfect)
    """
    if velocities is None:
        velocities = VELOCITIES

    os.makedirs(CONFIG['out_dir'], exist_ok=True)
    fig, axes = plt.subplots(1, len(velocities), figsize=(16, 5))
    if len(velocities) == 1:
        axes = [axes]

    all_ber = {}

    for ax, velocity in zip(axes, velocities):
        ber_results, ber_perfect, SNR_BER = evaluate_ber(velocity, seed)
        all_ber[velocity] = (ber_results, ber_perfect)

        # Perfect CSI (lower bound)
        ax.semilogy(SNR_BER, ber_perfect,
                    color='#95A5A6', linewidth=2.2,
                    linestyle=':', marker='x', markersize=7,
                    label='Perfect CSI', zorder=6)

        # Từng model
        for name in ber_results.keys():
            lw = 2.3 if name == 'CNN-LSTM-Attention' else 1.4
            ls = '-'  if name == 'CNN-LSTM-Attention' else '--'
            zord = 5  if name == 'CNN-LSTM-Attention' else 3
            ax.semilogy(SNR_BER, ber_results[name],
                        color=COLORS[name], linewidth=lw,
                        linestyle=ls, marker=MARKERS[name],
                        markersize=6, label=name, zorder=zord)

        # Ngưỡng BER = 1%
        ax.axhline(y=0.01, color='#E74C3C', linewidth=1.0,
                   linestyle='-.', alpha=0.6)

        ax.set_xlabel('SNR (dB)', fontsize=11)
        ax.set_ylabel('BER', fontsize=11)
        ax.set_title(f'v = {velocity} km/h', fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(bottom=1e-4, top=1.0)
        ax.set_xlim(left=-1, right=31)
        ax.legend(fontsize=8, loc='lower left', framealpha=0.9)

    plt.suptitle(
        f'BER vs SNR — QPSK + ZF Equalization | seed={seed}\n'
        '(Đường càng thấp càng tốt; Perfect CSI = lower bound)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    out = os.path.join(CONFIG['out_dir'], 'ber_vs_snr_combined.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

    return all_ber


# ============================================================
# HÀM 16: Iterative multi-step prediction
# ============================================================
def predict_multistep_iterative(model, X, horizon):
    """
    Dự đoán nhiều bước future bằng cách feed lại output làm input.

    h_pred(t+1) = model([h(t-49), ..., h(t)])
    h_pred(t+2) = model([h(t-48), ..., h(t), h_pred(t+1)])
    ...
    h_pred(t+K) = model([..., h_pred(t+K-1)])

    Đây là cách chuẩn để demo channel aging — error tích lũy theo K.

    Args:
        model   : nn.Module hoặc WienerFilter
        X       : (B, seq_len, 2) torch.Tensor
        horizon : số bước cần predict

    Returns:
        preds : (B, horizon, 2) torch.Tensor trên CPU
    """
    is_wiener = isinstance(model, WienerFilter)

    # Wiener: numpy-based, chạy trên CPU
    # NN: chạy trên DEVICE
    current = X.clone() if is_wiener else X.to(DEVICE)

    preds = []
    for k in range(horizon):
        if is_wiener:
            p = model.predict(current)                       # (B, 2) CPU torch
        else:
            with torch.no_grad():
                p = model.predict(current) if hasattr(model, 'predict') \
                    else model(current)                       # (B, 2) DEVICE torch

        preds.append(p.detach().cpu() if not is_wiener else p)

        # Sliding window: bỏ bước đầu, thêm bước vừa predict vào cuối
        # current và p phải cùng device
        p_unsq  = p.unsqueeze(1)                              # (B, 1, 2)
        current = torch.cat([current[:, 1:, :], p_unsq], dim=1)

    return torch.stack(preds, dim=1)                          # (B, horizon, 2)


# ============================================================
# HÀM 17: Lấy multi-step ground-truth từ h_raw
# ============================================================
def get_multistep_test_data(velocity, seed, max_horizon=10, seq_len=50,
                             train_ratio=0.8):
    """
    Tạo (X_test, y_test_multi) với y có max_horizon bước future.
    """
    path  = os.path.join(CONFIG['data_dir'],
                         f'channel_v{velocity}kmh_s{seed}.npz')
    d     = np.load(path)
    h_raw = d['h_raw']
    scale = float(d['scale'][0])

    h_ri = np.stack([h_raw.real, h_raw.imag], axis=1).astype(np.float32)

    X, y_multi = [], []
    for i in range(len(h_ri) - seq_len - max_horizon + 1):
        X.append(h_ri[i:i+seq_len])
        y_multi.append(h_ri[i+seq_len:i+seq_len+max_horizon])
    X       = np.array(X) / scale
    y_multi = np.array(y_multi) / scale

    split = int(len(X) * train_ratio)
    return (torch.FloatTensor(X[split:]),
            torch.FloatTensor(y_multi[split:]))


# ============================================================
# HÀM 18: Tính NMSE theo horizon (cho 1 model, 1 seed)
# ============================================================
def compute_nmse_multistep(model, velocity, seed, horizons,
                           snr_db=20, batch=512):
    """
    Tính NMSE tại mỗi horizon trong list horizons.
    """
    max_h = max(horizons)
    X, y_multi = get_multistep_test_data(velocity, seed, max_horizon=max_h)

    sig_pow   = float(torch.mean(X ** 2).item())
    noise_std = np.sqrt(sig_pow / 10 ** (snr_db / 10))
    torch.manual_seed(0)
    X_noisy = X + torch.randn_like(X) * noise_std

    preds_all = []
    for i in range(0, len(X_noisy), batch):
        Xb = X_noisy[i:i+batch]
        p  = predict_multistep_iterative(model, Xb, horizon=max_h)  # (b,h,2)
        preds_all.append(p)
    preds = torch.cat(preds_all, dim=0).numpy()                     # (N,h,2)
    targets = y_multi.numpy()

    nmse_per_h = []
    for k in horizons:
        idx = k - 1
        err = preds[:, idx, :] - targets[:, idx, :]
        nmse = np.sum(err ** 2) / (np.sum(targets[:, idx, :] ** 2) + 1e-10)
        nmse_per_h.append(10 * np.log10(nmse + 1e-10))
    return nmse_per_h


# ============================================================
# HÀM 19: Multi-step evaluation qua nhiều seed → mean ± std
# ============================================================
def evaluate_multistep_mean_std(name, velocity, horizons, snr_db=20):
    runs = []
    for s in SEEDS:
        if name == 'Wiener':
            model = build_wiener(velocity, s, horizon=1)
        else:
            try:
                model = build_model(name)
                model = load_checkpoint(model, name, velocity, s)
            except FileNotFoundError:
                return None, None                          # chưa train
        runs.append(compute_nmse_multistep(model, velocity, s,
                                           horizons, snr_db=snr_db))
    arr = np.array(runs)                                   # (n_seeds, n_h)
    return arr.mean(axis=0).tolist(), arr.std(axis=0).tolist()


# ============================================================
# HÀM 20: Vẽ NMSE vs Horizon
# ============================================================
def plot_nmse_vs_horizon(velocities=None, horizons=None,
                         model_list=None, snr_db=20):
    """
    Channel-aging chart: NMSE tăng dần khi horizon tăng.
    Vẽ riêng cho mỗi velocity, mỗi đường = 1 model.
    """
    if velocities is None:
        velocities = VELOCITIES
    if horizons is None:
        horizons = [1, 2, 3, 5, 7, 10]
    if model_list is None:
        model_list = MAIN_PLOT_MODELS

    os.makedirs(CONFIG['out_dir'], exist_ok=True)
    fig, axes = plt.subplots(1, len(velocities), figsize=(16, 5))
    if len(velocities) == 1:
        axes = [axes]

    print(f"\n--- Multi-step prediction (SNR={snr_db}dB) ---")
    multistep_results = {v: {} for v in velocities}

    for ax, velocity in zip(axes, velocities):
        print(f"\n  v = {velocity} km/h:")
        for name in model_list:
            mean_v, std_v = evaluate_multistep_mean_std(
                name, velocity, horizons, snr_db=snr_db
            )
            if mean_v is None:
                print(f"    {name:<22} [skip — chưa có checkpoint]")
                continue
            multistep_results[velocity][name] = (mean_v, std_v)
            print(f"    {name:<22} NMSE @ h=1,3,5,10: "
                  f"{mean_v[0]:.1f} / {mean_v[2]:.1f} / "
                  f"{mean_v[3]:.1f} / {mean_v[-1]:.1f} dB")

            mean_arr = np.array(mean_v)
            std_arr  = np.array(std_v)
            lw = 2.5 if name == 'CNN-LSTM-Attention' else 1.6
            ls = '-'  if name == 'CNN-LSTM-Attention' else '--'

            ax.plot(horizons, mean_arr,
                    color=COLORS[name], marker=MARKERS[name],
                    linewidth=lw, linestyle=ls,
                    markersize=7, label=name)
            ax.fill_between(horizons,
                            mean_arr - std_arr,
                            mean_arr + std_arr,
                            color=COLORS[name], alpha=0.10)

        ax.set_xlabel('Bước dự đoán future (k)', fontsize=12)
        ax.set_ylabel('NMSE (dB)', fontsize=12)
        ax.set_title(f'v = {velocity} km/h', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.suptitle(
        f'Multi-step Prediction — Channel Aging Effect\n'
        f'(SNR = {snr_db} dB | Iterative prediction | mean ± std qua 3 seeds)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    out = os.path.join(CONFIG['out_dir'], 'nmse_vs_horizon.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")
    return multistep_results


# ============================================================
# CHẠY CHÍNH
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Channel Prediction — Multi-Seed Evaluation")
    print(f"  Seeds   : {SEEDS}")
    print(f"  SNR     : {SNR_RANGE} dB")
    print(f"  Models  : {ALL_MODELS}")
    print("=" * 60)

    results = {v: {} for v in VELOCITIES}

    # ----------------------------------------------------------
    # Bước 1: Đánh giá NMSE vs SNR cho TẤT CẢ model
    # ----------------------------------------------------------
    for velocity in VELOCITIES:
        print(f"\n--- v = {velocity} km/h ---")
        for name in ALL_MODELS:
            print(f"  {name}...", end=' ', flush=True)
            try:
                mean_v, std_v = evaluate_mean_std(name, velocity)
            except FileNotFoundError as e:
                print(f"SKIP ({e})")
                continue
            results[velocity][name] = (mean_v, std_v)
            idxs = [SNR_RANGE.index(s) for s in [0, 10, 20, 30]]
            vals = [f"{mean_v[i]:.1f}+/-{std_v[i]:.1f}" for i in idxs]
            print(f"NMSE(0,10,20,30) = {vals}")

    # ----------------------------------------------------------
    # Bước 2: Biểu đồ chính (Wiener + LSTM + GRU + Transformer + Proposed)
    # ----------------------------------------------------------
    print("\nVẽ NMSE vs SNR (chính, 5 model)...")
    plot_nmse_vs_snr(results, model_list=MAIN_PLOT_MODELS, suffix='main')

    # Vẽ thêm chart đầy đủ 8 model (cho phụ lục)
    print("Vẽ NMSE vs SNR (đầy đủ, 8 model)...")
    plot_nmse_vs_snr(results, model_list=ALL_MODELS, suffix='all')

    # ----------------------------------------------------------
    # Bước 3: Ablation Study (LSTM / CNN-LSTM / LSTM-Attn / Full)
    # ----------------------------------------------------------
    print("\nVẽ Ablation Study...")
    plot_ablation_study(results)

    # ----------------------------------------------------------
    # Bước 3b: NMSE Gain so với Wiener — chart key insight
    # ----------------------------------------------------------
    print("\nVẽ NMSE Gain so với Wiener...")
    plot_nmse_gain_over_wiener(results, snr_target=20)

    # ----------------------------------------------------------
    # Bước 4: Bảng tổng kết
    # ----------------------------------------------------------
    print_summary_table(results, snr_target=20)
    # Lưu CSV cho báo cáo (complexity sẽ được append sau Bước 6)
    save_summary_csv(results, complexity=None)

    # ----------------------------------------------------------
    # Bước 5: Time-series và Attention
    # ----------------------------------------------------------
    print("\nVẽ time series (seed=42)...")
    for v in VELOCITIES:
        plot_timeseries(velocity=v, snr_db=20, seed=42)

    print("\nVẽ attention weights...")
    plot_attention_weights(velocity=300, seed=42)

    np.save('results/evaluation_multiseed.npy', results)

    # ----------------------------------------------------------
    # Bước 6: Complexity Analysis
    # ----------------------------------------------------------
    print("\n" + "=" * 55)
    print("  COMPLEXITY ANALYSIS")
    complexity = compute_model_complexity()
    plot_nmse_vs_params(complexity, results, velocity=120)
    np.save('results/complexity.npy', complexity)
    # Save lại CSV với cả complexity
    save_summary_csv(results, complexity=complexity)

    # ----------------------------------------------------------
    # Bước 7: BER Analysis — gộp 3 velocity vào 1 figure cho báo cáo
    # ----------------------------------------------------------
    print("\n" + "=" * 55)
    print("  BER ANALYSIS")
    plot_ber_combined(velocities=VELOCITIES, seed=42)
    # Vẽ thêm chart riêng cho từng velocity (làm phụ lục)
    for v in VELOCITIES:
        plot_ber_vs_snr(velocity=v, seed=42)

    # ----------------------------------------------------------
    # Bước 8 (PHASE 4 mới): Multi-step prediction — Channel Aging
    # ----------------------------------------------------------
    print("\n" + "=" * 55)
    print("  MULTI-STEP PREDICTION — Channel Aging")
    print("=" * 55)
    horizons = [1, 2, 3, 5, 7, 10]
    multistep = plot_nmse_vs_horizon(
        velocities = VELOCITIES,
        horizons   = horizons,
        model_list = MAIN_PLOT_MODELS,
        snr_db     = 20,
    )
    np.save('results/multistep_results.npy', multistep)

    print("\n" + "=" * 60)
    print("  DONE — Tất cả biểu đồ và file kết quả đã lưu ở results/")
    print("=" * 60)