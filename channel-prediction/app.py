import os
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.baselines import get_baseline

# ============================================================
# CẤU HÌNH MASSIVE MIMO
# ============================================================
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_TX         = 64
N_RX         = 2
FEATURE_SIZE = 2 * N_TX * N_RX   # = 256
N_COMPLEX    = N_TX * N_RX        # = 128

VELOCITIES      = [30, 120, 300]
VELOCITY_LABELS = {
    30 : "30 km/h — Đô thị",
    120: "120 km/h — Cao tốc",
    300: "300 km/h — Tàu cao tốc"
}
MODEL_NAMES = ['RNN', 'LSTM', 'GRU', 'Transformer']
DEMO_SEED   = 42
SNR_RANGE   = list(range(0, 31, 5))

# Cấu hình riêng từng model — KHỚP với train.py và evaluate.py
PER_MODEL_CONFIG = {
    'RNN'        : {'hidden_size': 290},
    'LSTM'       : {'hidden_size': 135},
    'GRU'        : {'hidden_size': 160},
    'Transformer': {'hidden_size': 120, 'nhead': 4, 'dim_feedforward': 480},
}

COLORS = {
    'RNN'        : '#E74C3C',
    'LSTM'       : '#3498DB',
    'GRU'        : '#3F4474',
    'Transformer': '#F39C12',
}

CKPT_DIR = 'results/checkpoints'
DATA_DIR = 'data'

# Style
DARK_BG  = '#0D1117'
CARD_BG  = '#161B22'
BORDER   = '#30363D'
TEXT_PRI = '#F0F6FC'
TEXT_SEC = '#8B949E'
ACCENT   = '#F39C12'
ACCENT2  = '#3498DB'

plt.rcParams.update({
    'figure.facecolor': DARK_BG, 'axes.facecolor': CARD_BG,
    'axes.edgecolor': BORDER, 'axes.labelcolor': TEXT_PRI,
    'xtick.color': TEXT_SEC, 'ytick.color': TEXT_SEC,
    'text.color': TEXT_PRI, 'grid.color': BORDER,
    'grid.alpha': 0.5, 'legend.facecolor': CARD_BG,
    'legend.edgecolor': BORDER, 'font.family': 'monospace',
})

_model_cache = {}
_data_cache  = {}


def _build_model(name):
    """Tạo model — PHẢI KHỚP với train.py và evaluate.py."""
    cfg = PER_MODEL_CONFIG[name]
    if name == 'Transformer':
        return get_baseline(
            'Transformer',
            hidden_size     = cfg['hidden_size'],
            num_layers      = 2,
            feature_size    = FEATURE_SIZE,
            nhead           = cfg['nhead'],
            dim_feedforward = cfg['dim_feedforward'],
        )
    return get_baseline(name,
                        hidden_size  = cfg['hidden_size'],
                        num_layers   = 2,
                        feature_size = FEATURE_SIZE)


def get_model(name, velocity, seed=DEMO_SEED):
    key  = f"{name}_v{velocity}_s{seed}"
    if key not in _model_cache:
        model = _build_model(name)
        path  = os.path.join(CKPT_DIR,
                             f'{name}_v{velocity}kmh_s{seed}.pth')
        if not os.path.exists(path):
            return None
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        model.to(DEVICE).eval()
        _model_cache[key] = model
    return _model_cache[key]


def get_test_data(velocity, seed=DEMO_SEED):
    key = f"v{velocity}_s{seed}"
    if key not in _data_cache:
        path = os.path.join(DATA_DIR,
                            f'mimo_v{velocity}kmh_s{seed}.npz')
        if not os.path.exists(path):
            return None, None
        d = np.load(path)
        _data_cache[key] = (
            torch.FloatTensor(d['X_test']),
            torch.FloatTensor(d['y_test'])
        )
    return _data_cache[key]


def add_noise(X, snr_db):
    sig_pow   = torch.mean(X**2).item()
    noise_std = np.sqrt(sig_pow / (10**(snr_db/10)))
    return X + torch.randn_like(X) * noise_std


def run_predict(model, X_noisy):
    with torch.no_grad():
        X_noisy = X_noisy.to(DEVICE)
        return model(X_noisy).cpu().numpy()


def compute_nmse(pred, target):
    err = pred - target
    return 10 * np.log10(
        np.sum(err**2) / (np.sum(target**2) + 1e-10) + 1e-10
    )


def measure_latency(model, X_sample, n_runs=50):
    X = X_sample[:1].to(DEVICE)
    for _ in range(5):
        run_predict(model, X)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        run_predict(model, X)
    return (time.perf_counter() - t0) / n_runs * 1000


def get_amplitude(arr, ant_idx=0):
    h_real = arr[:, ant_idx]
    h_imag = arr[:, ant_idx + N_COMPLEX]
    return np.sqrt(h_real**2 + h_imag**2)


# ============================================================
# TAB 1: DỰ ĐOÁN
# ============================================================
def predict_channel(velocity_label, snr_db, n_show,
                    ant_idx, selected_models):
    velocity       = int(velocity_label.split()[0])
    X_test, y_test = get_test_data(velocity)

    if X_test is None:
        return None, "Không tìm thấy data. Chạy generate_channel.py trước!"
    if not selected_models:
        return None, "Vui lòng chọn ít nhất 1 model"

    n_show   = int(n_show)
    ant_idx  = int(ant_idx)
    X_s      = X_test[:n_show]
    y_true   = y_test[:n_show].numpy()
    X_n      = add_noise(X_s, snr_db)

    h_true_amp = get_amplitude(y_true, ant_idx)
    t          = np.arange(n_show)

    n_models  = len(selected_models)
    fig, axes = plt.subplots(n_models, 1,
                              figsize=(14, 3.5*n_models + 1.5))
    if n_models == 1:
        axes = [axes]
    fig.patch.set_facecolor(DARK_BG)

    results_text = [
        f"**Massive MIMO | v={velocity} km/h | SNR={snr_db} dB | "
        f"Chỉ số #{ant_idx} | N_TX={N_TX}, N_RX={N_RX}**\n",
        f"{'Model':<14} {'NMSE (dB)':>12} {'Latency':>12}",
        "-" * 40
    ]

    for ax, name in zip(axes, selected_models):
        model = get_model(name, velocity)
        if model is None:
            ax.text(0.5, 0.5,
                    f'Chưa có checkpoint\n{name}',
                    ha='center', va='center',
                    color=TEXT_SEC, transform=ax.transAxes, fontsize=12)
            continue

        pred_np    = run_predict(model, X_n)
        h_pred_amp = get_amplitude(pred_np, ant_idx)
        nmse       = compute_nmse(pred_np, y_true)
        latency    = measure_latency(model, X_s)
        color      = COLORS[name]
        lw         = 2.0 if name == 'Transformer' else 1.4

        ax.plot(t, h_true_amp, color='#2ECC71', linewidth=1.5,
                label=f'Kênh thật — #{ant_idx}', alpha=0.9, zorder=3)
        ax.plot(t, h_pred_amp, color=color, linewidth=lw, linestyle='--',
                label=f'{name} (dự đoán)', alpha=0.9, zorder=4)
        ax.fill_between(t, h_true_amp, h_pred_amp,
                        alpha=0.2, color=color, zorder=2)

        ax.set_ylabel('|h(t)|', fontsize=10)
        ax.set_title(
            f'{name}  |  NMSE = {nmse:.2f} dB  |  '
            f'Latency = {latency:.2f} ms',
            fontsize=11, color=color, pad=8
        )
        ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        star = " ★" if name == 'Transformer' else "  "
        results_text.append(
            f"{name+star:<16} {nmse:>10.2f} dB {latency:>8.2f} ms"
        )

    axes[-1].set_xlabel('Bước thời gian', fontsize=11)
    fig.tight_layout()
    return fig, "\n".join(results_text)


# ============================================================
# TAB 2: NMSE vs SNR
# ============================================================
def compare_nmse(velocity_label):
    velocity       = int(velocity_label.split()[0])
    X_test, y_test = get_test_data(velocity)

    if X_test is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    y_np = y_test.numpy()

    for name in MODEL_NAMES:
        model = get_model(name, velocity)
        if model is None:
            continue

        nmse_list = []
        for snr in SNR_RANGE:
            X_n  = add_noise(X_test, snr)
            pred = run_predict(model, X_n)
            nmse_list.append(compute_nmse(pred, y_np))

        lw     = 3.0 if name == 'Transformer' else 1.8
        ls     = '-'  if name == 'Transformer' else '--'
        marker = 'D'  if name == 'Transformer' else 'o'
        ms     = 10   if name == 'Transformer' else 6
        zord   = 5    if name == 'Transformer' else 3

        ax.plot(SNR_RANGE, nmse_list,
                color=COLORS[name], linewidth=lw, linestyle=ls,
                marker=marker, markersize=ms, label=name, zorder=zord)

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('NMSE (dB)', fontsize=12)
    ax.set_title(
        f'Massive MIMO — NMSE vs SNR — Fair Comparison (~400K params)\n'
        f'v={velocity} km/h | N_TX={N_TX}, N_RX={N_RX} | seed={DEMO_SEED}',
        fontsize=12, color=TEXT_PRI
    )
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    return fig


# ============================================================
# TAB 3: THÔNG SỐ MODEL
# ============================================================
def show_model_stats():
    rows = []
    x_dummy = torch.randn(1, 50, FEATURE_SIZE).to(DEVICE)

    for name in MODEL_NAMES:
        model  = _build_model(name).to(DEVICE)
        params = sum(p.numel() for p in model.parameters())
        try:
            lat     = measure_latency(model, x_dummy, n_runs=20)
            lat_str = f"{lat:.2f} ms"
        except Exception:
            lat_str = "N/A"

        star = " ★" if name == 'Transformer' else ""
        rows.append({
            "Model"     : name + star,
            "Parameters": f"{params:,}",
            "Latency"   : lat_str,
        })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(DARK_BG)

    names  = [r["Model"] for r in rows]
    params = [int(r["Parameters"].replace(",", "")) for r in rows]
    colors = [COLORS[n.replace(" ★", "")] for n in names]

    ax = axes[0]
    bars = ax.barh(names, params, color=colors, alpha=0.85, height=0.5)
    for bar, p in zip(bars, params):
        ax.text(p + 200, bar.get_y() + bar.get_height()/2,
                f'{p:,}', va='center', fontsize=9, color=TEXT_PRI)
    ax.axvline(400_000, color='white', linestyle='--',
               linewidth=1.0, alpha=0.5, label='400K target')
    ax.set_title('Model Parameters — Fair Comparison', fontsize=11, color=TEXT_PRI)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    ax2 = axes[1]
    _, _, autotexts = ax2.pie(
        params, labels=names, colors=colors,
        autopct='%1.1f%%', startangle=90, pctdistance=0.8,
        textprops={'color': TEXT_PRI, 'fontsize': 9}
    )
    for at in autotexts:
        at.set_color(DARK_BG)
        at.set_fontweight('bold')

    fig.tight_layout()

    md  = f"**Massive MIMO — N_TX={N_TX}, N_RX={N_RX}, "
    md += f"Feature={FEATURE_SIZE} | Fair Comparison (~400K params)**\n\n"
    md += "| Model | Parameters | Latency |\n|---|---|---|\n"
    for r in rows:
        md += (f"| **{r['Model']}** | {r['Parameters']} "
               f"| {r['Latency']} |\n")
    return fig, md


CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {
    --bg-primary: #0D1117; --bg-card: #161B22; --border: #30363D;
    --text-primary: #F0F6FC; --accent: #F39C12; --accent-blue: #3498DB;
}
body, .gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
}
.gr-button {
    background: linear-gradient(135deg, var(--accent), var(--accent-blue)) !important;
    border: none !important; color: white !important;
    font-weight: 600 !important; border-radius: 8px !important;
    padding: 10px 24px !important;
}
"""


def build_app():
    with gr.Blocks(css=CSS, title="Massive MIMO Channel Prediction",
                   theme=gr.themes.Base()) as demo:

        gr.Markdown(
            f"# Massive MIMO Channel Prediction — Fair Comparison\n"
            f"N_TX={N_TX} · N_RX={N_RX} · ~400K params per model · "
            f"Proposed: **Transformer**"
        )

        with gr.Tabs():
            with gr.TabItem("Dự đoán kênh"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vel_1 = gr.Dropdown(
                            choices=[VELOCITY_LABELS[v] for v in VELOCITIES],
                            value=VELOCITY_LABELS[120],
                            label="Tốc độ"
                        )
                        snr_1  = gr.Slider(0, 30, step=5, value=20, label="SNR (dB)")
                        n_show = gr.Slider(50, 300, step=50, value=200,
                                           label="Số mẫu")
                        ant_idx = gr.Slider(0, N_COMPLEX-1, step=1, value=0,
                                            label=f"Chỉ số kênh (0-{N_COMPLEX-1})")
                        models_1 = gr.CheckboxGroup(
                            choices=MODEL_NAMES,
                            value=['LSTM', 'Transformer'],
                            label="Models"
                        )
                        btn_1 = gr.Button("Dự đoán", variant="primary")
                    with gr.Column(scale=3):
                        plot_1  = gr.Plot()
                        stats_1 = gr.Markdown()

                btn_1.click(predict_channel,
                            [vel_1, snr_1, n_show, ant_idx, models_1],
                            [plot_1, stats_1])

            with gr.TabItem("NMSE vs SNR"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vel_2 = gr.Dropdown(
                            choices=[VELOCITY_LABELS[v] for v in VELOCITIES],
                            value=VELOCITY_LABELS[120],
                            label="Tốc độ"
                        )
                        btn_2 = gr.Button("Vẽ", variant="primary")
                    with gr.Column(scale=3):
                        plot_2 = gr.Plot()
                btn_2.click(compare_nmse, [vel_2], [plot_2])

            with gr.TabItem("Thông số"):
                btn_3   = gr.Button("Load", variant="primary")
                plot_3  = gr.Plot()
                table_3 = gr.Markdown()
                btn_3.click(show_model_stats, [], [plot_3, table_3])

    return demo


if __name__ == "__main__":
    print(f"Device: {DEVICE} | N_TX={N_TX}, N_RX={N_RX}")
    print(f"Models: {MODEL_NAMES} | Proposed: Transformer")
    app = build_app()
    app.launch(share=True, server_port=7860, inbrowser=True)
