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
from models.proposed  import CNNLSTMAttention

# ============================================================
# CẤU HÌNH MASSIVE MIMO
# ============================================================
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_TX         = 64
N_RX         = 2
FEATURE_SIZE = 2 * N_TX * N_RX   # = 256

VELOCITIES      = [30, 120, 300]
VELOCITY_LABELS = {
    30 : "30 km/h — Đô thị",
    120: "120 km/h — Cao tốc",
    300: "300 km/h — Tàu cao tốc"
}
MODEL_NAMES = ['RNN', 'LSTM', 'GRU', 'Transformer', 'CNN-LSTM-Attention']
DEMO_SEED   = 42
SNR_RANGE   = list(range(0, 31, 5))

COLORS = {
    'RNN'              : '#E74C3C',
    'LSTM'             : '#3498DB',
    'GRU'              : '#3F4474',
    'Transformer'      : '#F39C12',
    'CNN-LSTM-Attention': '#9B59B6',
}

CKPT_DIR = 'results/checkpoints'
DATA_DIR = 'data'

# ============================================================
# STYLE — Dark theme
# ============================================================
DARK_BG  = '#0D1117'
CARD_BG  = '#161B22'
BORDER   = '#30363D'
TEXT_PRI = '#F0F6FC'
TEXT_SEC = '#8B949E'
ACCENT   = '#9B59B6'
ACCENT2  = '#3498DB'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor'  : CARD_BG,
    'axes.edgecolor'  : BORDER,
    'axes.labelcolor' : TEXT_PRI,
    'xtick.color'     : TEXT_SEC,
    'ytick.color'     : TEXT_SEC,
    'text.color'      : TEXT_PRI,
    'grid.color'      : BORDER,
    'grid.alpha'      : 0.5,
    'legend.facecolor': CARD_BG,
    'legend.edgecolor': BORDER,
    'font.family'     : 'monospace',
})

# ============================================================
# CACHE — tránh load lại nhiều lần
# ============================================================
_model_cache = {}
_data_cache  = {}


def _build_model(name):
    """Tạo model MIMO với đúng feature_size=256."""
    if name == 'CNN-LSTM-Attention':
        return CNNLSTMAttention(
            input_size   = FEATURE_SIZE,
            cnn_channels = 64,
            hidden_size  = 256,
            num_layers   = 2,
            num_heads    = 8,
            output_size  = FEATURE_SIZE,
        )
    return get_baseline(name,
                        hidden_size  = 128,
                        num_layers   = 2,
                        feature_size = FEATURE_SIZE)


def get_model(name, velocity, seed=DEMO_SEED):
    key  = f"{name}_v{velocity}_s{seed}"
    if key not in _model_cache:
        model = _build_model(name)
        # ✅ Tên file MIMO: mimo_v{v}kmh_s{seed}.pth
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
        # ✅ Tên file MIMO: mimo_v{v}kmh_s{seed}.npz
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


# ============================================================
# HÀM TIỆN ÍCH
# ============================================================
def add_noise(X, snr_db):
    sig_pow   = torch.mean(X**2).item()
    noise_std = np.sqrt(sig_pow / (10**(snr_db/10)))
    return X + torch.randn_like(X) * noise_std


def run_predict(model, X_noisy):
    with torch.no_grad():
        X_noisy = X_noisy.to(DEVICE)
        if hasattr(model, 'predict'):
            return model.predict(X_noisy).cpu().numpy()
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
    """
    Lấy biên độ kênh của 1 anten từ output MIMO.
    arr shape: (N, 256) = (N, 128_real + 128_imag)
    """
    h_real = arr[:, ant_idx]
    h_imag = arr[:, ant_idx + N_TX * N_RX]
    return np.sqrt(h_real**2 + h_imag**2)


# ============================================================
# TAB 1: DỰ ĐOÁN KÊNH
# ============================================================
def predict_channel(velocity_label, snr_db, n_show,
                    ant_idx, selected_models):
    velocity       = int(velocity_label.split()[0])
    X_test, y_test = get_test_data(velocity)

    if X_test is None:
        return None, "❌ Không tìm thấy data. Chạy generate_data.py trước!"
    if not selected_models:
        return None, "⚠️ Vui lòng chọn ít nhất 1 model"

    n_show   = int(n_show)
    ant_idx  = int(ant_idx)
    X_s      = X_test[:n_show]
    y_true   = y_test[:n_show].numpy()
    X_n      = add_noise(X_s, snr_db)

    # Biên độ kênh của anten ant_idx
    h_true_amp = get_amplitude(y_true, ant_idx)
    t          = np.arange(n_show)

    n_models  = len(selected_models)
    fig, axes = plt.subplots(n_models, 1,
                              figsize=(14, 3.5*n_models + 1.5))
    if n_models == 1:
        axes = [axes]
    fig.patch.set_facecolor(DARK_BG)

    results_text = [
        f"📡 **Massive MIMO | v={velocity} km/h | SNR={snr_db} dB | "
        f"Anten #{ant_idx} | N_TX={N_TX}**\n",
        f"{'Model':<22} {'NMSE (dB)':>12} {'Latency':>12}",
        "-" * 48
    ]

    for ax, name in zip(axes, selected_models):
        model = get_model(name, velocity)
        if model is None:
            ax.text(0.5, 0.5,
                    f'Chưa có checkpoint\n{name}\nChạy train.py trước',
                    ha='center', va='center',
                    color=TEXT_SEC, transform=ax.transAxes, fontsize=12)
            continue

        pred_np    = run_predict(model, X_n)
        h_pred_amp = get_amplitude(pred_np, ant_idx)

        # NMSE tính trên toàn bộ 128 chiều (không chỉ 1 anten)
        nmse    = compute_nmse(pred_np, y_true)
        latency = measure_latency(model, X_s)
        color   = COLORS[name]
        lw      = 2.0 if name == 'CNN-LSTM-Attention' else 1.4

        ax.plot(t, h_true_amp,
                color='#2ECC71', linewidth=1.5,
                label=f'Kênh thật — Anten #{ant_idx}',
                alpha=0.9, zorder=3)
        ax.plot(t, h_pred_amp,
                color=color, linewidth=lw, linestyle='--',
                label=f'{name} (dự đoán)', alpha=0.9, zorder=4)
        ax.fill_between(t, h_true_amp, h_pred_amp,
                        alpha=0.2, color=color, zorder=2)

        ax.set_ylabel('Biên độ |h(t)|', fontsize=10)
        ax.set_title(
            f'{name}  |  NMSE = {nmse:.2f} dB  |  '
            f'Latency = {latency:.2f} ms',
            fontsize=11, color=color, pad=8
        )
        ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        star = " ★" if name == 'CNN-LSTM-Attention' else "  "
        results_text.append(
            f"{name+star:<24} {nmse:>10.2f} dB {latency:>8.2f} ms"
        )

    axes[-1].set_xlabel('Bước thời gian', fontsize=11)
    plt.tight_layout()
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

        lw     = 3.0 if name == 'CNN-LSTM-Attention' else 1.8
        ls     = '-'  if name == 'CNN-LSTM-Attention' else '--'
        marker = '*'  if name == 'CNN-LSTM-Attention' else 'o'
        ms     = 10   if name == 'CNN-LSTM-Attention' else 6
        zord   = 5    if name == 'CNN-LSTM-Attention' else 3

        ax.plot(SNR_RANGE, nmse_list,
                color=COLORS[name], linewidth=lw, linestyle=ls,
                marker=marker, markersize=ms,
                label=name, zorder=zord)

        if name == 'CNN-LSTM-Attention':
            ax.annotate(
                'Proposed ★',
                xy=(SNR_RANGE[-1], nmse_list[-1]),
                xytext=(SNR_RANGE[-1]-6, nmse_list[-1]+1.5),
                color=COLORS[name], fontsize=9,
                arrowprops=dict(arrowstyle='->', color=COLORS[name], lw=1.2)
            )

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('NMSE (dB)', fontsize=12)
    ax.set_title(
        f'Massive MIMO — NMSE vs SNR\n'
        f'v={velocity} km/h | N_TX={N_TX}, N_RX={N_RX} | seed={DEMO_SEED}',
        fontsize=12, color=TEXT_PRI
    )
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    return fig


# ============================================================
# TAB 3: ATTENTION WEIGHTS
# ============================================================
def show_attention(velocity_label, sample_idx):
    velocity  = int(velocity_label.split()[0])
    X_test, _ = get_test_data(velocity)

    if X_test is None:
        return None, "❌ Không tìm thấy data"

    model = get_model('CNN-LSTM-Attention', velocity)
    if model is None:
        return None, "❌ Chưa có checkpoint CNN-LSTM-Attention"

    sample_idx = int(sample_idx)
    X_s = X_test[sample_idx:sample_idx+1].to(DEVICE)

    with torch.no_grad():
        _, weights = model(X_s)
    weights = weights.cpu().numpy()[0]  # (seq_len,)

    seq_len  = len(weights)
    top3_idx = np.argsort(weights)[-3:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5),
                             gridspec_kw={'width_ratios': [2, 1]})
    fig.patch.set_facecolor(DARK_BG)

    # Bar chart
    ax = axes[0]
    bar_colors = [ACCENT if i in top3_idx else '#4A5568'
                  for i in range(seq_len)]
    ax.bar(range(seq_len), weights,
           color=bar_colors, alpha=0.85, width=0.8)
    for i in top3_idx:
        ax.annotate(
            f't-{i}\n{weights[i]:.3f}',
            xy=(i, weights[i]),
            xytext=(i, weights[i]+0.003),
            ha='center', fontsize=8, color=ACCENT
        )
    ax.set_xlabel('Bước thời gian quá khứ', fontsize=11)
    ax.set_ylabel('Attention Weight', fontsize=11)
    ax.set_title(
        f'Attention Weights — Massive MIMO\n'
        f'v={velocity} km/h | Mẫu #{sample_idx} | '
        f'Global Avg Pooling (không collapse)',
        fontsize=11, color=TEXT_PRI
    )
    ax.grid(True, alpha=0.3, axis='y')

    # Cumulative chart
    ax2 = axes[1]
    cum = np.cumsum(weights[::-1])[::-1]
    ax2.fill_betweenx(range(seq_len), cum, alpha=0.6, color=ACCENT2)
    ax2.set_xlabel('Cumulative weight', fontsize=10)
    ax2.set_ylabel('Bước thời gian', fontsize=10)
    ax2.set_title('Tích lũy attention', fontsize=10, color=TEXT_PRI)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    plt.tight_layout()

    summary  = "**Top-3 bước thời gian quan trọng nhất:**\n"
    for rank, i in enumerate(top3_idx, 1):
        summary += (f"  {rank}. t-{i}: "
                    f"weight={weights[i]:.4f} "
                    f"({weights[i]*100:.1f}%)\n")
    summary += (f"\nMax weight : {weights.max():.4f}  "
                f"(phải < 0.15 để không collapse)\n"
                f"Entropy    : "
                f"{-np.sum(weights*np.log(weights+1e-10)):.3f}")

    return fig, summary


# ============================================================
# TAB 4: THÔNG SỐ MODEL
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

        star = " ★" if name == 'CNN-LSTM-Attention' else ""
        rows.append({
            "Model"       : name + star,
            "Parameters"  : f"{params:,}",
            "Latency"     : lat_str,
            "Kiến trúc"   : _get_arch_desc(name),
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
    ax.set_title(
        f'Model Parameters — Massive MIMO (Feature={FEATURE_SIZE})',
        fontsize=11, color=TEXT_PRI
    )
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        params, labels=names, colors=colors,
        autopct='%1.1f%%', startangle=90, pctdistance=0.8,
        textprops={'color': TEXT_PRI, 'fontsize': 9}
    )
    for at in autotexts:
        at.set_color(DARK_BG)
        at.set_fontweight('bold')

    plt.tight_layout()

    md  = f"**Massive MIMO — N_TX={N_TX}, N_RX={N_RX}, "
    md += f"Feature size={FEATURE_SIZE}**\n\n"
    md += "| Model | Parameters | Latency | Kiến trúc |\n"
    md += "|---|---|---|---|\n"
    for r in rows:
        md += (f"| **{r['Model']}** | {r['Parameters']} "
               f"| {r['Latency']} | {r['Kiến trúc']} |\n")
    return fig, md


def _get_arch_desc(name):
    return {
        'RNN'              : f'Vanilla RNN, input={FEATURE_SIZE}',
        'LSTM'             : f'LSTM + gates, input={FEATURE_SIZE}',
        'GRU'              : f'GRU lightweight, input={FEATURE_SIZE}',
        'Transformer'      : f'Multi-Head Attention, d={FEATURE_SIZE}',
        'CNN-LSTM-Attention': f'CNN+LSTM+MHA, 8heads ★',
    }.get(name, '')


# ============================================================
# CSS + HEADER
# ============================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {
    --bg-primary: #0D1117; --bg-card: #161B22; --border: #30363D;
    --text-primary: #F0F6FC; --text-secondary: #8B949E;
    --accent: #9B59B6; --accent-blue: #3498DB;
}
body, .gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
}
.gr-panel, .gr-box, .gr-block {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
.gr-button {
    background: linear-gradient(135deg, var(--accent), var(--accent-blue)) !important;
    border: none !important; color: white !important;
    font-weight: 600 !important; border-radius: 8px !important;
    padding: 10px 24px !important;
}
.gr-button:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }
#header-banner {
    background: linear-gradient(135deg, #0D1117 0%, #1a0a2e 50%, #0D1117 100%);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 28px 36px; margin-bottom: 16px;
}
code, .gr-textbox textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important; background: #0D1117 !important;
    color: #79C0FF !important;
}
"""

HEADER_HTML = f"""
<div id="header-banner">
    <div style="display:flex; align-items:center; gap:16px;">
        <div style="font-size:48px;">📡</div>
        <div>
            <h1 style="margin:0; font-size:22px; font-weight:700;
                background:linear-gradient(90deg,#9B59B6,#3498DB);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                Massive MIMO Channel Prediction
            </h1>
            <p style="margin:4px 0 0; color:#8B949E; font-size:13px; font-family:monospace;">
                CNN-LSTM-Attention  ·  N_TX={N_TX} BS Antennas  ·
                Rayleigh Fading  ·  30 / 120 / 300 km/h
            </p>
        </div>
        <div style="margin-left:auto;">
            <span style="background:#1a2a1a; color:#2ECC71; padding:4px 12px;
                border-radius:20px; font-size:12px; font-family:monospace;">
                ● MIMO Mode
            </span>
        </div>
    </div>
</div>
"""


# ============================================================
# BUILD GRADIO APP
# ============================================================
def build_app():
    with gr.Blocks(css=CSS, title="Massive MIMO Channel Prediction",
                   theme=gr.themes.Base()) as demo:

        gr.HTML(HEADER_HTML)

        with gr.Tabs():

            # TAB 1: DỰ ĐOÁN
            with gr.TabItem("📈  Dự đoán kênh"):
                gr.Markdown(
                    f"### Dự đoán kênh Massive MIMO "
                    f"({N_TX} anten BS × {N_RX} anten UE)"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        vel_1 = gr.Dropdown(
                            choices=[VELOCITY_LABELS[v] for v in VELOCITIES],
                            value=VELOCITY_LABELS[120],
                            label="🚀 Tốc độ di chuyển"
                        )
                        snr_1 = gr.Slider(
                            0, 30, step=5, value=20,
                            label="📶 SNR (dB)"
                        )
                        n_show = gr.Slider(
                            50, 300, step=50, value=200,
                            label="📊 Số mẫu hiển thị"
                        )
                        ant_idx = gr.Slider(
                            0, N_TX-1, step=1, value=0,
                            label=f"🔢 Anten hiển thị (0 - {N_TX-1})"
                        )
                        models_1 = gr.CheckboxGroup(
                            choices=MODEL_NAMES,
                            value=['LSTM', 'CNN-LSTM-Attention'],
                            label="🤖 Chọn model so sánh"
                        )
                        btn_1 = gr.Button("▶  Dự đoán", variant="primary")
                    with gr.Column(scale=3):
                        plot_1  = gr.Plot(label="Time Series Prediction")
                        stats_1 = gr.Markdown(label="Kết quả NMSE & Latency")

                btn_1.click(
                    fn=predict_channel,
                    inputs=[vel_1, snr_1, n_show, ant_idx, models_1],
                    outputs=[plot_1, stats_1]
                )

            # TAB 2: NMSE vs SNR
            with gr.TabItem("📊  So sánh NMSE"):
                gr.Markdown(
                    "### NMSE vs SNR — So sánh 5 model "
                    f"(NMSE tính trên cả {FEATURE_SIZE} chiều)"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        vel_2 = gr.Dropdown(
                            choices=[VELOCITY_LABELS[v] for v in VELOCITIES],
                            value=VELOCITY_LABELS[120],
                            label="🚀 Tốc độ di chuyển"
                        )
                        btn_2 = gr.Button("▶  Vẽ đồ thị", variant="primary")
                        gr.Markdown(
                            "**Đọc kết quả:**\n"
                            "- NMSE càng âm càng tốt\n"
                            "- -20 dB = sai số 1% tín hiệu\n"
                            "- CNN-LSTM-Attention (★) phải thấp nhất"
                        )
                    with gr.Column(scale=3):
                        plot_2 = gr.Plot(label="NMSE vs SNR")

                btn_2.click(
                    fn=compare_nmse,
                    inputs=[vel_2],
                    outputs=[plot_2]
                )

            # TAB 3: ATTENTION WEIGHTS
            with gr.TabItem("🔍  Attention Weights"):
                gr.Markdown(
                    "### Explainability — Bước thời gian nào được chú ý?\n"
                    "Global Average Pooling giúp phân phối đều, "
                    "không bị collapse như SISO"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        vel_3 = gr.Dropdown(
                            choices=[VELOCITY_LABELS[v] for v in VELOCITIES],
                            value=VELOCITY_LABELS[300],
                            label="🚀 Tốc độ di chuyển"
                        )
                        sample_idx = gr.Slider(
                            0, 99, step=1, value=0,
                            label="🎲 Chọn mẫu test"
                        )
                        btn_3 = gr.Button(
                            "▶  Phân tích Attention", variant="primary"
                        )
                    with gr.Column(scale=3):
                        plot_3  = gr.Plot(label="Attention Weights")
                        stats_3 = gr.Markdown(label="Phân tích")

                btn_3.click(
                    fn=show_attention,
                    inputs=[vel_3, sample_idx],
                    outputs=[plot_3, stats_3]
                )

            # TAB 4: THÔNG SỐ MODEL
            with gr.TabItem("⚙️  Thông số Model"):
                gr.Markdown(
                    f"### So sánh kiến trúc — "
                    f"Input={FEATURE_SIZE} (N_TX={N_TX}×2)"
                )
                btn_4   = gr.Button("▶  Load thông số", variant="primary")
                plot_4  = gr.Plot(label="Model Parameters")
                table_4 = gr.Markdown(label="Bảng so sánh")
                btn_4.click(
                    fn=show_model_stats,
                    inputs=[],
                    outputs=[plot_4, table_4]
                )

        gr.HTML("""
        <div style="text-align:center; padding:20px; color:#4A5568;
                    font-family:monospace; font-size:12px; margin-top:16px;">
            Đồ án tốt nghiệp · Massive MIMO Channel Prediction
            · CNN-LSTM-Attention · HCMUTE 2026
        </div>
        """)

    return demo


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Massive MIMO Channel Prediction Dashboard")
    print(f"  Device       : {DEVICE}")
    print(f"  N_TX={N_TX}, N_RX={N_RX}, Feature={FEATURE_SIZE}")
    print(f"  Models       : {MODEL_NAMES}")
    print("=" * 60)

    print("\nKiểm tra checkpoints...")
    missing = []
    for name in MODEL_NAMES:
        for v in VELOCITIES:
            path = os.path.join(
                CKPT_DIR, f'{name}_v{v}kmh_s{DEMO_SEED}.pth'
            )
            if not os.path.exists(path):
                missing.append(f'{name}_v{v}kmh_s{DEMO_SEED}.pth')

    if missing:
        print(f"⚠️  Thiếu {len(missing)} checkpoint:")
        for p in missing[:5]:
            print(f"   {p}")
        print("   → Chạy train.py trước!")
    else:
        print("✅ Tất cả checkpoints sẵn sàng!\n")

    print("\nKiểm tra data...")
    missing_data = []
    for v in VELOCITIES:
        path = os.path.join(DATA_DIR, f'mimo_v{v}kmh_s{DEMO_SEED}.npz')
        if not os.path.exists(path):
            missing_data.append(path)

    if missing_data:
        print(f"⚠️  Thiếu data:")
        for p in missing_data:
            print(f"   {p}")
        print("   → Chạy generate_data.py trước!")
    else:
        print("✅ Data sẵn sàng!\n")

    app = build_app()
    app.launch(
        share      = True,
        server_port= 7860,
        show_error = True,
        inbrowser  = True
    )