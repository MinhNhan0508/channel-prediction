import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.baselines import get_baseline
from models.proposed  import CNNLSTMAttention


# ============================================================
# CẤU HÌNH
# ============================================================
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
    h = CONFIG['hidden_size']
    num_layers = CONFIG['num_layers']
    if name == 'CNN-LSTM-Attention':
        return CNNLSTMAttention(
            hidden_size=h, num_layers=num_layers,
            cnn_channels=32, num_heads=2
        )
    return get_baseline(name, h, num_layers)


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
# HÀM 7: Vẽ NMSE vs SNR với error band
# ============================================================
def plot_nmse_vs_snr(results):
    os.makedirs(CONFIG['out_dir'], exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles    = ['v = 30 km/h\n(Đô thị)',
                 'v = 120 km/h\n(Cao tốc)',
                 'v = 300 km/h\n(Tàu cao tốc)']

    for ax, velocity, title in zip(axes, VELOCITIES, titles):
        for name in MODEL_NAMES:
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
    out = os.path.join(CONFIG['out_dir'], 'nmse_vs_snr_multiseed.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out}")


# ============================================================
# HÀM 8: Bảng tổng kết tại SNR = 20 dB
# ============================================================
def print_summary_table(results, snr_target=20):
    idx = SNR_RANGE.index(snr_target)
    print("\n" + "=" * 75)
    print(f"  NMSE (dB) tại SNR = {snr_target} dB — "
          f"Mean ± Std qua {len(SEEDS)} seeds")
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
    print("  ★ = Model đề xuất")


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
    plt.show()
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
    plt.show()
    print(f"Saved: {out}")


# ============================================================
# CHẠY CHÍNH
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Channel Prediction — Multi-Seed Evaluation")
    print(f"  Seeds   : {SEEDS}")
    print(f"  SNR     : {SNR_RANGE} dB")
    print("=" * 60)

    results = {v: {} for v in VELOCITIES}

    for velocity in VELOCITIES:
        print(f"\n--- v = {velocity} km/h ---")
        for name in MODEL_NAMES:
            print(f"  {name}...", end=' ', flush=True)
            mean_v, std_v = evaluate_mean_std(name, velocity)
            results[velocity][name] = (mean_v, std_v)
            idxs = [SNR_RANGE.index(s) for s in [0,10,20,30]]
            vals = [f"{mean_v[i]:.1f}±{std_v[i]:.1f}" for i in idxs]
            print(f"NMSE(0,10,20,30) = {vals}")

    print("\nVẽ NMSE vs SNR...")
    plot_nmse_vs_snr(results)

    print_summary_table(results, snr_target=20)

    print("\nVẽ time series (seed=42)...")
    for v in VELOCITIES:
        plot_timeseries(velocity=v, snr_db=20, seed=42)

    print("\nVẽ attention weights...")
    plot_attention_weights(velocity=300, seed=42)

    np.save('results/evaluation_multiseed.npy', results)
    print("\nXong!")