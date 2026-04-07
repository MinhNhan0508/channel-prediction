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
MODEL_NAMES = ['RNN', 'LSTM', 'GRU', 'Transformer', 'CNN-LSTM-Attention']
COLORS      = {
    'RNN'              : '#E74C3C',   # đỏ
    'LSTM'             : '#3498DB',   # xanh dương
    'GRU'              : '#2ECC71',   # xanh lá
    'Transformer'      : '#F39C12',   # cam
    'CNN-LSTM-Attention': '#9B59B6',  # tím - model đề xuất
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
    # Dải SNR để vẽ đồ thị NMSE vs SNR (dB)
    'snr_range'  : list(range(0, 31, 5)),  # 0, 5, 10, ..., 30 dB
}


# ============================================================
# HÀM 1: Tạo lại tất cả model (cùng config với train.py)
# ============================================================
def build_model(name):
    h = CONFIG['hidden_size']
    num_layers = CONFIG['num_layers']
    if name == 'CNN-LSTM-Attention':
        return CNNLSTMAttention(hidden_size=h, num_layers=num_layers, cnn_channels=32)
    return get_baseline(name, h, num_layers)


# ============================================================
# HÀM 2: Load checkpoint đã train
# ============================================================
def load_checkpoint(model, name, velocity):
    """
    Load weights tốt nhất đã lưu từ train.py.
    """
    path = os.path.join(
        CONFIG['ckpt_dir'],
        f'{name}_v{velocity}kmh.pth'
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")

    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# HÀM 3: Tính NMSE (dB) tại 1 mức SNR
# ============================================================
def compute_nmse_at_snr(model, X_test, y_test, snr_db):
    """
    Tính NMSE tại một mức SNR cụ thể.

    Quy trình:
      1. Thêm nhiễu AWGN vào input X_test tương ứng với SNR
      2. Cho model dự đoán
      3. Tính NMSE giữa dự đoán và ground truth

    NMSE (dB) = 10 * log10( ||y_pred - y_true||² / ||y_true||² )
    Số càng âm càng tốt. Ví dụ: -15 dB tốt hơn -10 dB.

    Args:
        model   : model đã load weights
        X_test  : input tensor (N, seq_len, 2)
        y_test  : target tensor (N, 2)
        snr_db  : mức SNR tính bằng dB

    Returns:
        nmse_db : NMSE tính bằng dB (float)
    """
    # Tính noise power từ SNR
    # SNR = signal_power / noise_power
    # noise_power = signal_power / SNR
    signal_power = torch.mean(X_test ** 2).item()
    snr_linear   = 10 ** (snr_db / 10)
    noise_power  = signal_power / snr_linear
    noise_std    = np.sqrt(noise_power)

    dataset = TensorDataset(X_test, y_test)
    loader  = DataLoader(dataset, batch_size=CONFIG['batch_size'],
                         shuffle=False)

    all_preds  = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # Thêm nhiễu Gaussian vào input
            noise   = torch.randn_like(X_batch) * noise_std
            X_noisy = X_batch + noise

            # Forward pass
            if hasattr(model, 'predict'):
                pred = model.predict(X_noisy)
            else:
                pred = model(X_noisy)

            all_preds.append(pred.cpu())
            all_targets.append(y_batch.cpu())

    preds   = torch.cat(all_preds,   dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    # Tính NMSE
    error        = preds - targets
    nmse_linear  = (np.sum(error ** 2)) / (np.sum(targets ** 2))
    nmse_db      = 10 * np.log10(nmse_linear + 1e-10)  # +1e-10 tránh log(0)

    return nmse_db


# ============================================================
# HÀM 4: Đánh giá 1 model trên toàn bộ SNR range
# ============================================================
def evaluate_model(name, velocity, X_test, y_test):
    """
    Đánh giá 1 model tại tất cả mức SNR.

    Returns:
        list NMSE(dB) tương ứng với CONFIG['snr_range']
    """
    model = build_model(name)
    model = load_checkpoint(model, name, velocity)

    nmse_list = []
    for snr in CONFIG['snr_range']:
        nmse = compute_nmse_at_snr(model, X_test, y_test, snr)
        nmse_list.append(nmse)

    return nmse_list


# ============================================================
# HÀM 5: Load test data
# ============================================================
def load_test_data(velocity):
    path = os.path.join(CONFIG['data_dir'], f'channel_v{velocity}kmh.npz')
    d    = np.load(path)
    X    = torch.FloatTensor(d['X_test'])
    y    = torch.FloatTensor(d['y_test'])
    return X, y


# ============================================================
# HÀM 6: Vẽ đồ thị NMSE vs SNR cho 3 tốc độ
# ============================================================
def plot_nmse_vs_snr(results):
    """
    Vẽ 3 subplot NMSE vs SNR cho 3 tốc độ.
    results: dict[velocity][model_name] = list of NMSE(dB)

    Đây là hình chính trong báo cáo đồ án.
    """
    os.makedirs(CONFIG['out_dir'], exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles    = ['v = 30 km/h\n(Đô thị)',
                 'v = 120 km/h\n(Cao tốc)',
                 'v = 300 km/h\n(Tàu cao tốc)']

    for ax, velocity, title in zip(axes, VELOCITIES, titles):
        for name in MODEL_NAMES:
            nmse_vals = results[velocity][name]
            lw  = 2.5 if name == 'CNN-LSTM-Attention' else 1.5
            zord = 5  if name == 'CNN-LSTM-Attention' else 3

            ax.plot(
                CONFIG['snr_range'], nmse_vals,
                color     = COLORS[name],
                marker    = MARKERS[name],
                linewidth = lw,
                markersize= 7,
                label     = name,
                zorder    = zord,
                linestyle = '-' if name == 'CNN-LSTM-Attention' else '--'
            )

        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('NMSE (dB)', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')

        # Thêm chú thích "Proposed" cho model đề xuất
        ax.annotate(
            'Proposed',
            xy     = (CONFIG['snr_range'][-1],
                      results[velocity]['CNN-LSTM-Attention'][-1]),
            xytext = (-40, 10),
            textcoords='offset points',
            color  = COLORS['CNN-LSTM-Attention'],
            fontsize=9,
            fontweight='bold'
        )

    plt.suptitle(
        'NMSE vs SNR — So sánh các mô hình Channel Prediction\n',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    out = os.path.join(CONFIG['out_dir'], 'nmse_vs_snr.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out}")


# ============================================================
# HÀM 7: In bảng tổng kết NMSE tại SNR = 20 dB
# ============================================================
def print_summary_table(results, snr_target=20):
    """
    In bảng so sánh NMSE tại 1 mức SNR cụ thể.
    SNR = 20 dB là mức phổ biến nhất trong paper.
    """
    idx = CONFIG['snr_range'].index(snr_target)

    print("\n" + "=" * 65)
    print(f"  NMSE (dB) tại SNR = {snr_target} dB")
    print("  (Số càng âm càng tốt)")
    print("=" * 65)
    print(f"  {'Model':<22} | {'30 km/h':>9} | "
          f"{'120 km/h':>9} | {'300 km/h':>9}")
    print("  " + "-" * 57)

    for name in MODEL_NAMES:
        vals = [results[v][name][idx] for v in VELOCITIES]
        marker = " ★" if name == 'CNN-LSTM-Attention' else "  "
        print(f"  {name:<22}{marker}| {vals[0]:>8.2f} | "
              f"{vals[1]:>8.2f} | {vals[2]:>8.2f}")

    print("=" * 65)
    print("  ★ = Model đề xuất")


# ============================================================
# HÀM 8: Visualize Attention Weights
# ============================================================
def plot_attention_weights(velocity=300, n_samples=3):
    """
    Visualize attention weights của CNN-LSTM-Attention.
    Cho thấy model đang 'chú ý' vào bước thời gian nào trong chuỗi.
    Đây là hình minh họa tính giải thích được của model đề xuất.
    """
    X_test, y_test = load_test_data(velocity)

    model = build_model('CNN-LSTM-Attention')
    model = load_checkpoint(model, 'CNN-LSTM-Attention', velocity)

    # Lấy n_samples mẫu đầu
    X_sample = X_test[:n_samples].to(DEVICE)

    with torch.no_grad():
        _, weights = model(X_sample)

    weights = weights.cpu().numpy()  # (n_samples, seq_len)

    fig, axes = plt.subplots(1, n_samples, figsize=(14, 3))
    for i, ax in enumerate(axes):
        ax.bar(range(len(weights[i])), weights[i],
               color='#9B59B6', alpha=0.7)
        ax.set_xlabel('Bước thời gian (quá khứ)', fontsize=10)
        ax.set_ylabel('Attention weight', fontsize=10)
        ax.set_title(f'Mẫu {i+1} — v={velocity} km/h', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        'Attention Weights — CNN-LSTM-Attention (Model đề xuất)\n'
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
    print("=" * 55)
    print("  Channel Prediction — Evaluation")
    print(f"  SNR range: {CONFIG['snr_range']} dB")
    print(f"  Models   : {MODEL_NAMES}")
    print("=" * 55)

    # Thu thập kết quả
    results = {v: {} for v in VELOCITIES}

    for velocity in VELOCITIES:
        print(f"\n--- Velocity = {velocity} km/h ---")
        X_test, y_test = load_test_data(velocity)

        for name in MODEL_NAMES:
            print(f"  Evaluating {name}...", end=' ', flush=True)
            nmse_list = evaluate_model(name, velocity, X_test, y_test)
            results[velocity][name] = nmse_list

            # In NMSE tại SNR = 0, 10, 20, 30 dB
            snrs  = CONFIG['snr_range']
            idxs  = [snrs.index(s) for s in [0, 10, 20, 30]
                     if s in snrs]
            vals  = [f"{nmse_list[i]:.1f}" for i in idxs]
            print(f"NMSE(0,10,20,30dB) = {vals}")

    # Vẽ đồ thị NMSE vs SNR
    print("\nVẽ đồ thị NMSE vs SNR...")
    plot_nmse_vs_snr(results)

    # In bảng tổng kết
    print_summary_table(results, snr_target=20)

    # Visualize attention weights
    print("\nVẽ attention weights...")
    plot_attention_weights(velocity=300)

    # Lưu kết quả số
    np.save('results/evaluation_results.npy', results)
    print("\nSaved: results/evaluation_results.npy")
    print("\nHoàn tất!")