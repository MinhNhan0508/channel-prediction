import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.baselines import get_baseline
from models.proposed  import CNNLSTMAttention

# ============================================================
# CẤU HÌNH
# ============================================================
FEATURE_SIZE = 256   # 2 × 64 × 2
N_TX         = 64
N_RX         = 2

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VELOCITIES  = [30, 120, 300]
SEEDS       = [42, 123, 456]
MODEL_NAMES = ['RNN', 'LSTM', 'GRU', 'Transformer', 'CNN-LSTM-Attention']
SNR_RANGE   = list(range(0, 31, 5))

COLORS = {
    'RNN'              : '#E74C3C',
    'LSTM'             : '#3498DB',
    'GRU'              : '#2ECC71',
    'Transformer'      : '#F39C12',
    'CNN-LSTM-Attention': '#9B59B6',
}
MARKERS = {
    'RNN': 'o', 'LSTM': 's', 'GRU': '^',
    'Transformer': 'D', 'CNN-LSTM-Attention': '*'
}

CONFIG = {
    'hidden_size' : 128,
    'num_layers'  : 2,
    'feature_size': FEATURE_SIZE,
    'data_dir'    : 'data',
    'ckpt_dir'    : 'results/checkpoints',
    'out_dir'     : 'results',
    'batch_size'  : 256,
}


# ============================================================
# HÀM 1: Tạo model MIMO
# ============================================================
def build_model(name):
    h, nl, fs = CONFIG['hidden_size'], CONFIG['num_layers'], FEATURE_SIZE
    if name == 'CNN-LSTM-Attention':
        return CNNLSTMAttention(
            input_size=fs, cnn_channels=64, hidden_size=256,
            num_layers=nl, num_heads=8, output_size=fs
        )
    return get_baseline(name, h, nl, fs)


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
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(DEVICE).eval()
    return model


# ============================================================
# HÀM 3: Load test data MIMO
# ============================================================
def load_test_data(velocity, seed):
    path = os.path.join(
        CONFIG['data_dir'],
        f'mimo_v{velocity}kmh_s{seed}.npz'
    )
    d = np.load(path)
    return (torch.FloatTensor(d['X_test']),
            torch.FloatTensor(d['y_test']))


# ============================================================
# HÀM 4: Tính NMSE tại 1 mức SNR
# ============================================================
def compute_nmse_at_snr(model, X_test, y_test, snr_db):
    """
    Đánh giá NMSE tại mức SNR cụ thể.

    Với Massive MIMO, NMSE được tính trên toàn bộ
    vector kênh 128 chiều (không chỉ 2 chiều như SISO).
    """
    sig_pow   = torch.mean(X_test**2).item()
    noise_std = np.sqrt(sig_pow / (10**(snr_db/10)))

    loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=CONFIG['batch_size'], shuffle=False
    )

    preds, targets = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb    = Xb.to(DEVICE)
            Xn    = Xb + torch.randn_like(Xb) * noise_std
            pred  = model.predict(Xn) \
                    if hasattr(model, 'predict') \
                    else model(Xn)
            preds.append(pred.cpu().numpy())
            targets.append(yb.numpy())

    p = np.concatenate(preds)
    t = np.concatenate(targets)
    return 10 * np.log10(np.sum((p-t)**2) / (np.sum(t**2) + 1e-10) + 1e-10)


# ============================================================
# HÀM 5: Evaluate 1 model qua tất cả SNR
# ============================================================
def evaluate_mean_std(name, velocity):
    all_runs = []
    for seed in SEEDS:
        model = build_model(name)
        model = load_checkpoint(model, name, velocity, seed)
        X, y  = load_test_data(velocity, seed)
        nmse_list = [compute_nmse_at_snr(model, X, y, snr)
                     for snr in SNR_RANGE]
        all_runs.append(nmse_list)

    arr = np.array(all_runs)
    return arr.mean(axis=0).tolist(), arr.std(axis=0).tolist()


# ============================================================
# HÀM 6: Vẽ NMSE vs SNR với error band
# ============================================================
def plot_nmse_vs_snr(results):
    os.makedirs(CONFIG['out_dir'], exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = [
        f'v = 30 km/h\n(Đô thị, fd≈67Hz)',
        f'v = 120 km/h\n(Cao tốc, fd≈267Hz)',
        f'v = 300 km/h\n(Tàu cao tốc, fd≈667Hz)'
    ]

    for ax, velocity, title in zip(axes, VELOCITIES, titles):
        for name in MODEL_NAMES:
            mean_v, std_v = results[velocity][name]
            mean_a = np.array(mean_v)
            std_a  = np.array(std_v)
            lw = 2.5 if name == 'CNN-LSTM-Attention' else 1.5
            ls = '-'  if name == 'CNN-LSTM-Attention' else '--'

            ax.plot(SNR_RANGE, mean_a,
                    color=COLORS[name], marker=MARKERS[name],
                    linewidth=lw, linestyle=ls,
                    markersize=7, label=name)
            ax.fill_between(SNR_RANGE,
                            mean_a - std_a, mean_a + std_a,
                            color=COLORS[name], alpha=0.12)

        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('NMSE (dB)', fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')

    plt.suptitle(
        f'Massive MIMO Channel Prediction — NMSE vs SNR\n'
        f'N_TX={N_TX}, N_RX={N_RX} | Mean±Std qua {len(SEEDS)} seeds',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    out = os.path.join(CONFIG['out_dir'], 'mimo_nmse_vs_snr.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out}")


# ============================================================
# HÀM 7: Bảng tổng kết tại SNR = 20 dB
# ============================================================
def print_summary_table(results, snr_target=20):
    idx = SNR_RANGE.index(snr_target)
    print("\n" + "=" * 75)
    print(f"  Massive MIMO — NMSE (dB) tại SNR = {snr_target} dB")
    print(f"  N_TX={N_TX}, N_RX={N_RX} | Mean ± Std qua {len(SEEDS)} seeds")
    print("=" * 75)
    print(f"  {'Model':<22} | {'30 km/h':>16} | "
          f"{'120 km/h':>16} | {'300 km/h':>16}")
    print("  " + "-" * 73)

    for name in MODEL_NAMES:
        mk  = " ★" if name == 'CNN-LSTM-Attention' else "  "
        row = f"  {name:<22}{mk}|"
        for v in VELOCITIES:
            m, s = results[v][name]
            row += f" {m[idx]:>6.2f}±{s[idx]:>4.2f} dB |"
        print(row)

    print("=" * 75)
    print("  ★ = Model đề xuất (CNN-LSTM-Attention)")
    print("  Số càng âm càng tốt")


# ============================================================
# HÀM 8: Vẽ time series dự đoán vs thực tế
# ============================================================
def plot_timeseries(velocity=300, snr_db=20, seed=42,
                    n_show=200, ant_idx=0):
    """
    Vẽ so sánh kênh thật vs dự đoán.

    Với Massive MIMO, vẽ biên độ kênh của 1 anten cụ thể
    (ant_idx) thay vì toàn bộ 64 anten.

    Args:
        ant_idx : chỉ số anten cần vẽ (0 đến N_TX-1)
    """
    os.makedirs(CONFIG['out_dir'], exist_ok=True)
    X_test, y_test = load_test_data(velocity, seed)

    sig_pow   = torch.mean(X_test**2).item()
    noise_std = np.sqrt(sig_pow / (10**(snr_db/10)))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    y_true = y_test[:n_show].numpy()

    # Lấy biên độ kênh của anten ant_idx
    # y shape: (n_show, 256) = (n_show, 128_real + 128_imag)
    h_real = y_true[:, ant_idx]                        # real của anten ant_idx
    h_imag = y_true[:, ant_idx + N_TX * N_RX]          # imag của anten ant_idx
    h_true_amp = np.sqrt(h_real**2 + h_imag**2)

    for ax, name, color in zip(
        axes,
        ['LSTM', 'CNN-LSTM-Attention'],
        [COLORS['LSTM'], COLORS['CNN-LSTM-Attention']]
    ):
        model = build_model(name)
        model = load_checkpoint(model, name, velocity, seed)
        X_s   = X_test[:n_show].to(DEVICE)
        X_n   = X_s + torch.randn_like(X_s) * noise_std

        with torch.no_grad():
            pred = model.predict(X_n) \
                   if hasattr(model, 'predict') \
                   else model(X_n)

        pred_np = pred.cpu().numpy()
        p_real  = pred_np[:, ant_idx]
        p_imag  = pred_np[:, ant_idx + N_TX * N_RX]
        h_pred_amp = np.sqrt(p_real**2 + p_imag**2)

        t   = np.arange(n_show)
        err = pred_np - y_true
        nmse = 10 * np.log10(
            np.sum(err**2) / (np.sum(y_true**2) + 1e-10))

        ax.plot(t, h_true_amp, color='#2ECC71', linewidth=1.2,
                label=f'Kênh thật — Anten #{ant_idx}', alpha=0.9)
        ax.plot(t, h_pred_amp, color=color, linewidth=1.2,
                linestyle='--',
                label=f'{name} (dự đoán)', alpha=0.9)
        ax.fill_between(t, h_true_amp, h_pred_amp,
                        alpha=0.15, color=color)
        ax.set_ylabel('Biên độ |h(t)|', fontsize=11)
        ax.set_title(
            f'{name} — v={velocity} km/h, SNR={snr_db} dB'
            f' | NMSE (toàn bộ) = {nmse:.2f} dB',
            fontsize=11
        )
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel('Bước thời gian', fontsize=11)
    plt.suptitle(
        f'Massive MIMO Channel Prediction\n'
        f'N_TX={N_TX}, N_RX={N_RX} | '
        f'v={velocity} km/h | SNR={snr_db} dB | seed={seed}',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    out = os.path.join(
        CONFIG['out_dir'],
        f'mimo_timeseries_v{velocity}kmh.png'
    )
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out}")


# ============================================================
# HÀM 9: Visualize Attention Weights
# ============================================================
def plot_attention_weights(velocity=300, seed=42, n_samples=3):
    """
    Attention weights cho Massive MIMO.
    Phân phối đều hơn SISO nhờ Global Average Pooling.
    """
    os.makedirs(CONFIG['out_dir'], exist_ok=True)
    X_test, _ = load_test_data(velocity, seed)

    model = build_model('CNN-LSTM-Attention')
    model = load_checkpoint(model, 'CNN-LSTM-Attention', velocity, seed)

    with torch.no_grad():
        _, weights = model(X_test[:n_samples].to(DEVICE))
    weights = weights.cpu().numpy()

    fig, axes = plt.subplots(1, n_samples, figsize=(14, 3))
    for i, ax in enumerate(axes):
        ax.bar(range(len(weights[i])), weights[i],
               color='#9B59B6', alpha=0.7)
        ax.set_xlabel('Bước thời gian quá khứ', fontsize=10)
        ax.set_ylabel('Attention weight', fontsize=10)
        ax.set_title(
            f'Mẫu {i+1} — v={velocity} km/h\n'
            f'max={weights[i].max():.3f}',
            fontsize=9
        )
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        f'Attention Weights — CNN-LSTM-Attention (Massive MIMO)\n'
        f'N_TX={N_TX} | Global Average Pooling — phân phối đều',
        fontsize=12
    )
    plt.tight_layout()
    out = os.path.join(CONFIG['out_dir'], 'mimo_attention_weights.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out}")


# ============================================================
# CHẠY CHÍNH
# ============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  Massive MIMO Channel Prediction — Evaluation")
    print(f"  N_TX={N_TX}, N_RX={N_RX}, Feature={FEATURE_SIZE}")
    print(f"  SNR range : {SNR_RANGE} dB")
    print(f"  Seeds     : {SEEDS}")
    print("=" * 65)

    results = {v: {} for v in VELOCITIES}

    for velocity in VELOCITIES:
        print(f"\n--- v = {velocity} km/h ---")
        for name in MODEL_NAMES:
            print(f"  {name}...", end=' ', flush=True)
            mean_v, std_v = evaluate_mean_std(name, velocity)
            results[velocity][name] = (mean_v, std_v)
            idxs = [SNR_RANGE.index(s)
                    for s in [0, 10, 20, 30] if s in SNR_RANGE]
            vals = [f"{mean_v[i]:.1f}±{std_v[i]:.1f}" for i in idxs]
            print(f"NMSE(0,10,20,30dB) = {vals}")

    print("\nVẽ NMSE vs SNR...")
    plot_nmse_vs_snr(results)

    print_summary_table(results, snr_target=20)

    print("\nVẽ time series...")
    for v in VELOCITIES:
        plot_timeseries(velocity=v, snr_db=20, seed=42, ant_idx=0)

    print("\nVẽ attention weights...")
    plot_attention_weights(velocity=300, seed=42)

    np.save('results/evaluation_mimo.npy', results)
    print("\nXong!")