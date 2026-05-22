import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.baselines import get_baseline

# ============================================================
# CẤU HÌNH
# ============================================================
N_TX         = 64
N_RX         = 2
FEATURE_SIZE = 2 * N_TX * N_RX   # = 256

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VELOCITIES  = [30, 120, 300]
SEEDS       = [42, 123, 456]
MODEL_NAMES = ['RNN', 'LSTM', 'GRU', 'Transformer']
SNR_RANGE   = list(range(0, 31, 5))

# Cấu hình riêng từng model — PHẢI KHỚP với train.py
PER_MODEL_CONFIG = {
    'RNN'        : {'hidden_size': 290},
    'LSTM'       : {'hidden_size': 135},
    'GRU'        : {'hidden_size': 160},
    'Transformer': {'hidden_size': 120, 'nhead': 4, 'dim_feedforward': 480},
}

COLORS = {
    'RNN'        : '#E74C3C',
    'LSTM'       : '#3498DB',
    'GRU'        : '#2ECC71',
    'Transformer': '#F39C12',
}
MARKERS = {
    'RNN': 'o', 'LSTM': 's', 'GRU': '^', 'Transformer': 'D',
}

CONFIG = {
    'num_layers'  : 2,
    'feature_size': FEATURE_SIZE,
    'data_dir'    : 'data',
    'ckpt_dir'    : 'results/checkpoints',
    'out_dir'     : 'results',
    'batch_size'  : 256,
}


# ============================================================
# HÀM 1: Tạo model — KHỚP với train.py
# ============================================================
def build_model(name):
    cfg = PER_MODEL_CONFIG[name]
    fs  = FEATURE_SIZE
    nl  = CONFIG['num_layers']

    if name == 'Transformer':
        return get_baseline(
            'Transformer',
            hidden_size     = cfg['hidden_size'],
            num_layers      = nl,
            feature_size    = fs,
            nhead           = cfg['nhead'],
            dim_feedforward = cfg['dim_feedforward'],
        )
    return get_baseline(name,
                        hidden_size  = cfg['hidden_size'],
                        num_layers   = nl,
                        feature_size = fs)


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
# HÀM 3: Load test data
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
    sig_pow   = torch.mean(X_test**2).item()
    noise_std = np.sqrt(sig_pow / (10**(snr_db/10)))

    loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=CONFIG['batch_size'], shuffle=False
    )

    preds, targets = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb   = Xb.to(DEVICE)
            Xn   = Xb + torch.randn_like(Xb) * noise_std
            pred = model(Xn)
            preds.append(pred.cpu().numpy())
            targets.append(yb.numpy())

    p = np.concatenate(preds)
    t = np.concatenate(targets)
    return 10 * np.log10(np.sum((p-t)**2) / (np.sum(t**2) + 1e-10) + 1e-10)


# ============================================================
# HÀM 5: Evaluate qua tất cả seeds
# ============================================================
def evaluate_mean_std(name, velocity):
    """Đánh giá qua TẤT CẢ seeds → mean + std."""
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
# HÀM 6: Vẽ NMSE vs SNR
# ============================================================
def plot_nmse_vs_snr(results):
    """3 subplot (3 velocity) với error band."""
    os.makedirs(CONFIG['out_dir'], exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = [
        'v = 30 km/h\n(Đô thị, fd≈67Hz)',
        'v = 120 km/h\n(Cao tốc, fd≈267Hz)',
        'v = 300 km/h\n(Tàu cao tốc, fd≈667Hz)'
    ]

    for ax, velocity, title in zip(axes, VELOCITIES, titles):
        for name in MODEL_NAMES:
            mean_v, std_v = results[velocity][name]
            mean_a = np.array(mean_v)
            std_a  = np.array(std_v)

            lw = 2.5 if name == 'Transformer' else 1.5
            ls = '-'  if name == 'Transformer' else '--'

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

    fig.suptitle(
        f'Massive MIMO Channel Prediction — Fair Comparison (~400K params per model)\n'
        f'N_TX={N_TX}, N_RX={N_RX} | Mean±Std qua {len(SEEDS)} seeds',
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout()
    out = os.path.join(CONFIG['out_dir'], 'mimo_nmse_vs_snr.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out}")


# ============================================================
# HÀM 7: Bảng tổng kết
# ============================================================
def print_summary_table(results, snr_target=20):
    """Bảng so sánh tại 1 SNR (mặc định 20 dB)."""
    idx = SNR_RANGE.index(snr_target)
    print("\n" + "=" * 75)
    print(f"  Massive MIMO — NMSE (dB) tại SNR = {snr_target} dB")
    print(f"  N_TX={N_TX}, N_RX={N_RX} | Mean ± Std qua {len(SEEDS)} seeds")
    print(f"  Fair Comparison: ~400K params per model")
    print("=" * 75)
    print(f"  {'Model':<14} | {'30 km/h':>16} | "
          f"{'120 km/h':>16} | {'300 km/h':>16}")
    print("  " + "-" * 73)

    for name in MODEL_NAMES:
        mk  = " ★" if name == 'Transformer' else "  "
        row = f"  {name:<14}{mk}|"
        for v in VELOCITIES:
            m, s = results[v][name]
            row += f" {m[idx]:>6.2f}±{s[idx]:>4.2f} dB |"
        print(row)

    print("=" * 75)
    print("  ★ = Model đề xuất (Transformer) | NMSE càng âm càng tốt")


# ============================================================
# HÀM 8: Vẽ time series
# ============================================================
def plot_timeseries(velocity=300, snr_db=20, seed=42,
                    n_show=200, ant_idx=0):
    """Vẽ kênh thật vs dự đoán — so sánh LSTM và Transformer."""
    os.makedirs(CONFIG['out_dir'], exist_ok=True)
    X_test, y_test = load_test_data(velocity, seed)

    sig_pow   = torch.mean(X_test**2).item()
    noise_std = np.sqrt(sig_pow / (10**(snr_db/10)))

    n_complex  = N_TX * N_RX  # = 128
    y_true     = y_test[:n_show].numpy()
    h_real     = y_true[:, ant_idx]
    h_imag     = y_true[:, ant_idx + n_complex]
    h_true_amp = np.sqrt(h_real**2 + h_imag**2)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for ax, name, color in zip(
        axes,
        ['LSTM', 'Transformer'],
        [COLORS['LSTM'], COLORS['Transformer']]
    ):
        model = build_model(name)
        model = load_checkpoint(model, name, velocity, seed)
        X_s   = X_test[:n_show].to(DEVICE)
        X_n   = X_s + torch.randn_like(X_s) * noise_std

        with torch.no_grad():
            pred = model(X_n)

        pred_np    = pred.cpu().numpy()
        p_real     = pred_np[:, ant_idx]
        p_imag     = pred_np[:, ant_idx + n_complex]
        h_pred_amp = np.sqrt(p_real**2 + p_imag**2)

        t    = np.arange(n_show)
        err  = pred_np - y_true
        nmse = 10 * np.log10(
            np.sum(err**2) / (np.sum(y_true**2) + 1e-10))

        ax.plot(t, h_true_amp, color='#2ECC71', linewidth=1.2,
                label=f'Kênh thật — chỉ số #{ant_idx}', alpha=0.9)
        ax.plot(t, h_pred_amp, color=color, linewidth=1.2,
                linestyle='--',
                label=f'{name} (dự đoán)', alpha=0.9)
        ax.fill_between(t, h_true_amp, h_pred_amp,
                        alpha=0.15, color=color)
        ax.set_ylabel('Biên độ |h(t)|', fontsize=11)
        ax.set_title(
            f'{name} — v={velocity} km/h, SNR={snr_db} dB'
            f' | NMSE (toàn bộ {FEATURE_SIZE} chiều) = {nmse:.2f} dB',
            fontsize=11
        )
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel('Bước thời gian', fontsize=11)
    fig.suptitle(
        f'Massive MIMO Channel Prediction — Fair Comparison\n'
        f'N_TX={N_TX}, N_RX={N_RX} | '
        f'v={velocity} km/h | SNR={snr_db} dB | seed={seed}',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    out = os.path.join(
        CONFIG['out_dir'],
        f'mimo_timeseries_v{velocity}kmh.png'
    )
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out}")


# ============================================================
# CHẠY CHÍNH
# ============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  Massive MIMO Channel Prediction — Evaluation")
    print(f"  Fair Comparison: ~400K params per model")
    print(f"  Device    : {DEVICE}")
    print(f"  N_TX={N_TX}, N_RX={N_RX}, Feature={FEATURE_SIZE}")
    print(f"  SNR range : {SNR_RANGE} dB")
    print(f"  Seeds     : {SEEDS}")
    print(f"  Models    : {MODEL_NAMES}")
    print(f"  Proposed  : Transformer")
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

    np.save('results/evaluation_mimo.npy', results)
    print("\nXong!")
