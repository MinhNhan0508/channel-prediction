import os
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr

# Thêm đường dẫn để import models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.baselines import get_baseline
from models.proposed  import CNNLSTMAttention
from models.classical import WienerFilter

# ============================================================
# CẤU HÌNH
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VELOCITIES    = [30, 120, 300]
VELOCITY_LABELS = {30: "30 km/h — Đô thị", 120: "120 km/h — Cao tốc", 300: "300 km/h — Tàu cao tốc"}
# Wiener đặt đầu list để hội đồng thấy ngay: "có cả baseline cổ điển"
MODEL_NAMES   = ['Wiener', 'RNN', 'LSTM', 'GRU', 'Transformer', 'CNN-LSTM-Attention']
DEMO_SEED     = 42
SNR_RANGE     = list(range(0, 31, 5))

COLORS = {
    'Wiener'           : '#7F8C8D',  # xám — baseline cổ điển (đồng nhất với evaluate.py)
    'RNN'              : '#E74C3C',
    'LSTM'             : '#3498DB',
    'GRU'              : "#3F4474",
    'Transformer'      : '#F39C12',
    'CNN-LSTM-Attention': '#9B59B6',
}

CKPT_DIR = 'results/checkpoints'
DATA_DIR = 'data'

# ============================================================
# STYLE — Dark theme chuyên nghiệp
# ============================================================
DARK_BG   = '#0D1117'
CARD_BG   = '#161B22'
BORDER    = '#30363D'
TEXT_PRI  = '#F0F6FC'
TEXT_SEC  = '#8B949E'
ACCENT    = '#9B59B6'
ACCENT2   = '#3498DB'

plt.rcParams.update({
    'figure.facecolor' : DARK_BG,
    'axes.facecolor'   : CARD_BG,
    'axes.edgecolor'   : BORDER,
    'axes.labelcolor'  : TEXT_PRI,
    'xtick.color'      : TEXT_SEC,
    'ytick.color'      : TEXT_SEC,
    'text.color'       : TEXT_PRI,
    'grid.color'       : BORDER,
    'grid.alpha'       : 0.5,
    'legend.facecolor' : CARD_BG,
    'legend.edgecolor' : BORDER,
    'font.family'      : 'monospace',
})

# ============================================================
# CACHE — Tránh load lại model nhiều lần
# ============================================================
_model_cache = {}
_data_cache  = {}


def get_model(name, velocity, seed=DEMO_SEED):
    """
    Load model từ cache hoặc tạo mới.
    - NN models: load checkpoint .pth từ results/checkpoints/
    - Wiener   : fit closed-form từ X_train/y_train trong file .npz
                 (không có checkpoint vì là linear MMSE giải tích)
    """
    key = f"{name}_v{velocity}_s{seed}"
    if key in _model_cache:
        return _model_cache[key]

    # ---- Wiener filter: fit closed-form, không có checkpoint ----
    if name == 'Wiener':
        path = os.path.join(DATA_DIR,
                            f'channel_v{velocity}kmh_s{seed}.npz')
        if not os.path.exists(path):
            return None
        d = np.load(path)
        wiener = WienerFilter(seq_len=50, horizon=1, ridge=1e-4)
        wiener.fit(d['X_train'], d['y_train'])
        _model_cache[key] = wiener
        return wiener

    # ---- Neural network: load checkpoint ----
    model = _build_model(name)
    path  = os.path.join(CKPT_DIR, f'{name}_v{velocity}kmh_s{seed}.pth')
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.to(DEVICE)
    model.eval()
    _model_cache[key] = model
    return _model_cache[key]


def get_test_data(velocity, seed=DEMO_SEED):
    key = f"v{velocity}_s{seed}"
    if key not in _data_cache:
        path = os.path.join(DATA_DIR, f'channel_v{velocity}kmh_s{seed}.npz')
        if not os.path.exists(path):
            return None, None
        d = np.load(path)
        _data_cache[key] = (
            torch.FloatTensor(d['X_test']),
            torch.FloatTensor(d['y_test'])
        )
    return _data_cache[key]


def _build_model(name):
    """Tạo model (chưa load weights / chưa fit)."""
    if name == 'Wiener':
        return WienerFilter(seq_len=50, horizon=1, ridge=1e-4)
    if name == 'CNN-LSTM-Attention':
        return CNNLSTMAttention(
            hidden_size=64, num_layers=2,
            cnn_channels=32, num_heads=2
        )
    return get_baseline(name, hidden_size=64, num_layers=2)


# ============================================================
# HÀM TIỆN ÍCH
# ============================================================
def add_noise(X, snr_db):
    signal_power = torch.mean(X ** 2).item()
    noise_std    = np.sqrt(signal_power / (10 ** (snr_db / 10)))
    return X + torch.randn_like(X) * noise_std


def predict(model, X_noisy):
    with torch.no_grad():
        X_noisy = X_noisy.to(DEVICE)
        if hasattr(model, 'predict'):
            return model.predict(X_noisy).cpu().numpy()
        return model(X_noisy).cpu().numpy()


def compute_nmse(pred, target):
    err = pred - target
    return 10 * np.log10(np.sum(err**2) / (np.sum(target**2) + 1e-10))


def measure_latency(model, X_sample, n_runs=50):
    X = X_sample[:1].to(DEVICE)
    # Warmup
    for _ in range(5):
        _ = predict(model, X)
    # Measure
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = predict(model, X)
    return (time.perf_counter() - t0) / n_runs * 1000  # ms


# ============================================================
# TAB 1: DỰ ĐOÁN KÊNH TRUYỀN
# ============================================================
def predict_channel(velocity_label, snr_db, n_show, selected_models):
    velocity = int(velocity_label.split()[0])
    X_test, y_test = get_test_data(velocity)

    if X_test is None:
        return None, "❌ Không tìm thấy data. Kiểm tra thư mục `data/`"

    if not selected_models:
        return None, "⚠️ Vui lòng chọn ít nhất 1 model"

    n_show = int(n_show)
    X_s    = X_test[:n_show]
    y_true = y_test[:n_show].numpy()
    X_n    = add_noise(X_s, snr_db)

    n_models = len(selected_models)
    fig_h    = 3.5 * n_models + 1.5
    fig, axes = plt.subplots(n_models, 1, figsize=(14, fig_h))
    if n_models == 1:
        axes = [axes]

    fig.patch.set_facecolor(DARK_BG)

    h_true_amp = np.sqrt(y_true[:, 0]**2 + y_true[:, 1]**2)
    t = np.arange(n_show)

    results_text = []
    results_text.append(f"📡 **v = {velocity} km/h | SNR = {snr_db} dB | seed = {DEMO_SEED}**\n")
    results_text.append(f"{'Model':<22} {'NMSE (dB)':>12} {'Latency':>12}")
    results_text.append("-" * 48)

    for ax, name in zip(axes, selected_models):
        model = get_model(name, velocity)
        if model is None:
            # Wiener cần file data; NN cần checkpoint
            err_msg = (f'Thiếu data\n{name}' if name == 'Wiener'
                       else f'Chưa có checkpoint\n{name}')
            ax.text(0.5, 0.5, err_msg,
                    ha='center', va='center', color=TEXT_SEC,
                    transform=ax.transAxes, fontsize=12)
            continue

        pred_np    = predict(model, X_n)
        h_pred_amp = np.sqrt(pred_np[:, 0]**2 + pred_np[:, 1]**2)

        nmse    = compute_nmse(pred_np, y_true)
        latency = measure_latency(model, X_s)
        color   = COLORS[name]

        lw = 2.0 if name == 'CNN-LSTM-Attention' else 1.4

        ax.plot(t, h_true_amp,
                color='#2ECC71', linewidth=1.5,
                label='Kênh thật |h(t)|', alpha=0.9, zorder=3)
        ax.plot(t, h_pred_amp,
                color=color, linewidth=lw,
                linestyle='--', label=f'{name} (dự đoán)', alpha=0.9, zorder=4)

        # Vùng sai số — giống evaluate.py
        ax.fill_between(t, h_true_amp, h_pred_amp,
                        alpha=0.25, color=color,
                        label='Error Number', zorder=2)

        ax.set_ylabel('Biên độ |h(t)|', fontsize=10)
        ax.set_title(
            f'{name}  |  NMSE = {nmse:.2f} dB  |  Latency = {latency:.2f} ms',
            fontsize=11, color=color, pad=8
        )
        ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        star = " ★" if name == 'CNN-LSTM-Attention' else "  "
        results_text.append(f"{name+star:<24} {nmse:>10.2f} dB {latency:>8.2f} ms")

    axes[-1].set_xlabel('Bước thời gian', fontsize=11)

    plt.tight_layout()

    summary = "\n".join(results_text)
    return fig, summary


# ============================================================
# TAB 2: SO SÁNH NMSE vs SNR
# ============================================================
def compare_nmse(velocity_label):
    velocity = int(velocity_label.split()[0])
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
            X_n   = add_noise(X_test, snr)
            pred  = predict(model, X_n)
            nmse  = compute_nmse(pred, y_np)
            nmse_list.append(nmse)

        lw     = 3.0  if name == 'CNN-LSTM-Attention' else 1.8
        ls     = '-'  if name == 'CNN-LSTM-Attention' else '--'
        ms     = 9    if name == 'CNN-LSTM-Attention' else 6
        marker = '*'  if name == 'CNN-LSTM-Attention' else 'o'
        zord   = 5    if name == 'CNN-LSTM-Attention' else 3

        ax.plot(SNR_RANGE, nmse_list,
                color=COLORS[name], linewidth=lw, linestyle=ls,
                marker=marker, markersize=ms, label=name, zorder=zord)

        if name == 'CNN-LSTM-Attention':
            ax.annotate('Proposed ★',
                        xy=(SNR_RANGE[-1], nmse_list[-1]),
                        xytext=(SNR_RANGE[-1] - 6, nmse_list[-1] + 1.5),
                        color=COLORS[name], fontsize=9,
                        arrowprops=dict(arrowstyle='->', color=COLORS[name], lw=1.2))

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('NMSE (dB)', fontsize=12)
    ax.set_title(
        f'NMSE vs SNR  ·  v = {velocity} km/h  ·  seed = {DEMO_SEED}\n',
        fontsize=12, color=TEXT_PRI
    )
    ax.legend(fontsize=10, loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    return fig


# ============================================================
# TAB 3: ATTENTION WEIGHTS
# ============================================================
def show_attention(velocity_label, sample_idx):
    velocity = int(velocity_label.split()[0])
    X_test, _ = get_test_data(velocity)

    if X_test is None:
        return None, "❌ Không tìm thấy data"

    model = get_model('CNN-LSTM-Attention', velocity)
    if model is None:
        return None, "❌ Chưa có checkpoint CNN-LSTM-Attention"

    sample_idx = int(sample_idx)
    X_s = X_test[sample_idx:sample_idx + 1].to(DEVICE)

    with torch.no_grad():
        _, weights = model(X_s)
    weights = weights.cpu().numpy()[0]   # (seq_len,)

    seq_len = len(weights)
    top3_idx = np.argsort(weights)[-3:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5),
                             gridspec_kw={'width_ratios': [2, 1]})
    fig.patch.set_facecolor(DARK_BG)

    # Bar chart
    ax = axes[0]
    bar_colors = [ACCENT if i in top3_idx else '#4A5568'
                  for i in range(seq_len)]
    ax.bar(range(seq_len), weights, color=bar_colors, alpha=0.85, width=0.8)
    for i in top3_idx:
        ax.annotate(f't-{i}\n{weights[i]:.3f}',
                    xy=(i, weights[i]),
                    xytext=(i, weights[i] + 0.005),
                    ha='center', fontsize=8, color=ACCENT)

    ax.set_xlabel('Bước thời gian quá khứ', fontsize=11)
    ax.set_ylabel('Attention Weight', fontsize=11)
    ax.set_title(
        f'Attention Weights  ·  v = {velocity} km/h  ·  Mẫu #{sample_idx}',
        fontsize=12, color=TEXT_PRI
    )
    ax.grid(True, alpha=0.3, axis='y')

    # Heatmap-style cumulative
    ax2 = axes[1]
    cum = np.cumsum(weights[::-1])[::-1]
    ax2.fill_betweenx(range(seq_len), cum, alpha=0.6, color=ACCENT2)
    ax2.set_xlabel('Cumulative weight', fontsize=10)
    ax2.set_ylabel('Bước thời gian', fontsize=10)
    ax2.set_title('Tích lũy attention\n', fontsize=10, color=TEXT_PRI)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    plt.tight_layout()

    # Text summary
    summary = "**Top-3 bước thời gian quan trọng nhất:**\n"
    for rank, i in enumerate(top3_idx, 1):
        summary += f"  {rank}. t-{i}: weight = {weights[i]:.4f} ({weights[i]*100:.1f}%)\n"
    summary += f"\nMax weight: {weights.max():.4f} | "
    summary += f"Entropy: {-np.sum(weights * np.log(weights + 1e-10)):.3f}"

    return fig, summary


# ============================================================
# TAB 4: THÔNG SỐ MODEL
# ============================================================
def show_model_stats():
    rows = []
    for name in MODEL_NAMES:
        # Wiener: cần fit trên 1 (velocity, seed) đại diện để có .W
        # Lý do: WienerFilter không có .parameters(), chỉ có .num_params()
        if name == 'Wiener':
            model = get_model('Wiener', velocity=120, seed=DEMO_SEED)
            if model is None:
                # Fallback nếu thiếu file npz: chỉ lấy num_params từ instance chưa fit (=0)
                model = _build_model('Wiener')
            params = model.num_params() if model.num_params() > 0 \
                     else 50 * 2 * 2     # seq_len=50, in=2, out=2 → ước tính
        else:
            model  = _build_model(name)
            params = sum(p.numel() for p in model.parameters())

        # Latency estimate
        x_dummy = torch.randn(1, 50, 2)
        try:
            if isinstance(model, WienerFilter):
                # Wiener chạy CPU, không cần .to(DEVICE)
                lat = measure_latency(model, x_dummy, n_runs=20)
            else:
                lat = measure_latency(
                    model.to(DEVICE),
                    x_dummy, n_runs=20
                )
            lat_str = f"{lat:.2f} ms"
        except Exception:
            lat_str = "N/A"

        star = " ★" if name == 'CNN-LSTM-Attention' else ""
        rows.append({
            "Model"       : name + star,
            "Parameters"  : f"{params:,}",
            "Latency (CPU)": lat_str,
            "Kiến trúc"   : _get_arch_desc(name),
        })

    # ----------------------------------------------------------
    # Vẽ bar chart số params (giữ Wiener — show off "Wiener tí hon")
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(DARK_BG)

    names  = [r["Model"] for r in rows]
    params = [int(r["Parameters"].replace(",", "")) for r in rows]
    colors = [COLORS[n.replace(" ★", "")] for n in names]

    ax = axes[0]
    bars = ax.barh(names, params, color=colors, alpha=0.85, height=0.5)
    for bar, p in zip(bars, params):
        ax.text(p + 500, bar.get_y() + bar.get_height()/2,
                f'{p:,}', va='center', fontsize=9, color=TEXT_PRI)
    ax.set_title('Model Parameters', fontsize=12, color=TEXT_PRI)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    # ----------------------------------------------------------
    # Pie chart — BỎ WIENER (Option A): Wiener=200 params quá nhỏ
    # so với LSTM ~50k → miếng pie xíu xiu, không đẹp.
    # Chỉ show phân bổ giữa các model NN.
    # ----------------------------------------------------------
    ax2 = axes[1]
    nn_indices = [i for i, n in enumerate(names)
                  if n.replace(" ★", "") != 'Wiener']
    pie_names  = [names[i]  for i in nn_indices]
    pie_params = [params[i] for i in nn_indices]
    pie_colors = [colors[i] for i in nn_indices]

    wedges, texts, autotexts = ax2.pie(
        pie_params, labels=pie_names, colors=pie_colors,
        autopct='%1.1f%%', startangle=90,
        pctdistance=0.8,
        textprops={'color': TEXT_PRI, 'fontsize': 9}
    )
    for at in autotexts:
        at.set_color(DARK_BG)
        at.set_fontweight('bold')

    ax2.set_title('Phân bổ Params (chỉ NN models)\n'
                  'Wiener = 200 params, tách riêng',
                  fontsize=11, color=TEXT_PRI, pad=10)

    plt.tight_layout()

    # Markdown table
    md = "| Model | Parameters | Latency | Kiến trúc |\n"
    md += "|---|---|---|---|\n"
    for r in rows:
        md += f"| **{r['Model']}** | {r['Parameters']} | {r['Latency (CPU)']} | {r['Kiến trúc']} |\n"

    return fig, md


def _get_arch_desc(name):
    descs = {
        'Wiener'           : 'Linear MMSE — closed-form (cổ điển)',
        'RNN'              : 'Vanilla RNN, 2 layers',
        'LSTM'             : 'LSTM + gates, 2 layers',
        'GRU'              : 'GRU (lightweight LSTM)',
        'Transformer'      : 'Multi-Head Self-Attention',
        'CNN-LSTM-Attention': 'CNN + LSTM + Multi-Head Attn ★',
    }
    return descs.get(name, '')


# ============================================================
# CUSTOM CSS
# ============================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg-primary   : #0D1117;
    --bg-card      : #161B22;
    --border       : #30363D;
    --text-primary : #F0F6FC;
    --text-secondary: #8B949E;
    --accent       : #9B59B6;
    --accent-blue  : #3498DB;
    --accent-green : #2ECC71;
    --accent-orange: #F39C12;
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
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

.gr-button:hover {
    opacity: 0.85 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(155, 89, 182, 0.4) !important;
}

.gr-slider input[type='range'] {
    accent-color: var(--accent) !important;
}

.gr-tab-nav button {
    color: var(--text-secondary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
}

.gr-tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

#header-banner {
    background: linear-gradient(135deg, #0D1117 0%, #1a0a2e 50%, #0D1117 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}

#header-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(155,89,182,0.08) 0%, transparent 60%);
    pointer-events: none;
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}

code, .gr-textbox textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    background: #0D1117 !important;
    color: #79C0FF !important;
}
"""

# ============================================================
# HEADER HTML
# ============================================================
HEADER_HTML = """
<div id="header-banner">
    <div style="display:flex; align-items:center; gap:16px;">
        <div style="font-size:48px;">📡</div>
        <div>
            <h1 style="margin:0; font-size:22px; font-weight:700;
                       background: linear-gradient(90deg, #9B59B6, #3498DB);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-family: 'Space Grotesk', sans-serif;">
                Channel Prediction Dashboard
            </h1>
            <p style="margin:4px 0 0; color:#8B949E; font-size:13px; font-family:monospace;">
                CNN-LSTM-Attention  ·  Rayleigh Fading  ·  30 / 120 / 300 km/h
            </p>
        </div>
        <div style="margin-left:auto; text-align:right;">
            <span style="background:#1a2a1a; color:#2ECC71; padding:4px 12px;
                         border-radius:20px; font-size:12px; font-family:monospace;">
                ● Model loaded
            </span>
        </div>
    </div>
</div>
"""


# ============================================================
# BUILD GRADIO APP
# ============================================================
def build_app():
    with gr.Blocks(css=CSS, title="Channel Prediction Dashboard",
                   theme=gr.themes.Base()) as demo:

        gr.HTML(HEADER_HTML)

        with gr.Tabs():

            # --------------------------------------------------
            # TAB 1: DỰ ĐOÁN
            # --------------------------------------------------
            with gr.TabItem("📈  Dự đoán kênh"):
                gr.Markdown("### Dự đoán biên độ kênh truyền theo thời gian")

                with gr.Row():
                    with gr.Column(scale=1):
                        vel_1 = gr.Dropdown(
                            choices=[VELOCITY_LABELS[v] for v in VELOCITIES],
                            value=VELOCITY_LABELS[120],
                            label="🚀 Tốc độ di chuyển"
                        )
                        snr_1 = gr.Slider(
                            minimum=0, maximum=30, step=5, value=20,
                            label="📶 SNR (dB)"
                        )
                        n_show = gr.Slider(
                            minimum=50, maximum=300, step=50, value=200,
                            label="📊 Số mẫu hiển thị"
                        )
                        models_1 = gr.CheckboxGroup(
                            choices=MODEL_NAMES,
                            value=['Wiener', 'LSTM', 'CNN-LSTM-Attention'],
                            label="🤖 Chọn model so sánh"
                        )
                        btn_1 = gr.Button("▶  Dự đoán", variant="primary")

                    with gr.Column(scale=3):
                        plot_1   = gr.Plot(label="Time Series Prediction")
                        stats_1  = gr.Markdown(label="Kết quả NMSE & Latency")

                btn_1.click(
                    fn=predict_channel,
                    inputs=[vel_1, snr_1, n_show, models_1],
                    outputs=[plot_1, stats_1]
                )

            # --------------------------------------------------
            # TAB 2: NMSE vs SNR
            # --------------------------------------------------
            with gr.TabItem("📊  So sánh NMSE"):
                gr.Markdown("### NMSE theo mức SNR — So sánh tất cả model")

                with gr.Row():
                    with gr.Column(scale=1):
                        vel_2 = gr.Dropdown(
                            choices=[VELOCITY_LABELS[v] for v in VELOCITIES],
                            value=VELOCITY_LABELS[120],
                            label="🚀 Tốc độ di chuyển"
                        )
                        btn_2 = gr.Button("▶  Vẽ đồ thị", variant="primary")
                        gr.Markdown("""
**Hướng dẫn đọc:**
- Trục X: SNR (dB)
- Trục Y: NMSE (dB) 
                        """)

                    with gr.Column(scale=3):
                        plot_2 = gr.Plot(label="NMSE vs SNR")

                btn_2.click(
                    fn=compare_nmse,
                    inputs=[vel_2],
                    outputs=[plot_2]
                )

            # --------------------------------------------------
            # TAB 3: ATTENTION WEIGHTS
            # --------------------------------------------------
            with gr.TabItem("🔍  Attention Weights"):
                gr.Markdown("""
### Explainability — Model chú ý vào bước thời gian nào?
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        vel_3 = gr.Dropdown(
                            choices=[VELOCITY_LABELS[v] for v in VELOCITIES],
                            value=VELOCITY_LABELS[300],
                            label="🚀 Tốc độ di chuyển"
                        )
                        sample_idx = gr.Slider(
                            minimum=0, maximum=99, step=1, value=0,
                            label="🎲 Chọn mẫu test"
                        )
                        btn_3 = gr.Button("▶  Phân tích Attention", variant="primary")

                    with gr.Column(scale=3):
                        plot_3  = gr.Plot(label="Attention Weights")
                        stats_3 = gr.Markdown(label="Phân tích")

                btn_3.click(
                    fn=show_attention,
                    inputs=[vel_3, sample_idx],
                    outputs=[plot_3, stats_3]
                )

            # --------------------------------------------------
            # TAB 4: THÔNG SỐ MODEL
            # --------------------------------------------------
            with gr.TabItem("⚙️  Thông số Model"):
                gr.Markdown("### So sánh kiến trúc và độ phức tạp các model")

                btn_4    = gr.Button("▶  Load thông số", variant="primary")
                plot_4   = gr.Plot(label="Model Parameters")
                table_4  = gr.Markdown(label="Bảng so sánh")

                btn_4.click(
                    fn=show_model_stats,
                    inputs=[],
                    outputs=[plot_4, table_4]
                )

        # Footer
        gr.HTML("""
        <div style="text-align:center; padding:20px; color:#4A5568;
                    font-family:monospace; font-size:12px; margin-top:16px;">
            Đồ án tốt nghiệp · CNN-LSTM-Attention for High-Mobility Channel Prediction
            · HCMUTE 2026
        </div>
        """)

    return demo


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Channel Prediction Dashboard")
    print(f"  Device : {DEVICE}")
    print(f"  Models : {MODEL_NAMES}")
    print("=" * 60)
    print("\nKiểm tra checkpoints và data...")

    missing = []
    for name in MODEL_NAMES:
        for v in VELOCITIES:
            if name == 'Wiener':
                # Wiener không có checkpoint, cần file data .npz để fit
                path = os.path.join(DATA_DIR,
                                    f'channel_v{v}kmh_s{DEMO_SEED}.npz')
            else:
                path = os.path.join(CKPT_DIR,
                                    f'{name}_v{v}kmh_s{DEMO_SEED}.pth')
            if not os.path.exists(path):
                missing.append(path)

    if missing:
        print(f"⚠️  Thiếu {len(missing)} file:")
        for p in missing[:5]:
            print(f"   {p}")
    else:
        print("✅ Tất cả checkpoints + data đã sẵn sàng!\n")

    app = build_app()
    app.launch(
        share=True,           
        server_port=7860,
        show_error=True,
        inbrowser=True       
    )