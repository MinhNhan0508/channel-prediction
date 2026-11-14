"""Local Gradio explorer for verified, synthetic MIMO channel predictions."""
import argparse
import os
from pathlib import Path

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.inference import InferenceBundle, nmse_per_frame
from src.references import ar_predict, kalman_predict, hold_last

APP_CSS = """
.gradio-container { max-width: 1440px !important; margin: auto !important; }
.controls-panel { padding: 12px !important; border-radius: 12px !important; }
.action-row { justify-content: flex-start; margin: 8px 0 4px; }
.result-panel { height: 420px !important; min-height: 420px !important; overflow: auto; }
.result-plot img { height: 100% !important; width: 100%; object-fit: contain; }
.controls-panel, .result-panel { border: 1px solid #494055 !important; }
.results-table th, .results-table td { border-color: #494055 !important; }
.results-table table { font-size: 12px !important; }
.results-table th, .results-table td { padding: 7px 8px !important; }
body, .gradio-container { background: #19171f !important; color-scheme: dark; }
@media (max-width: 640px) {
    .controls-panel { padding: 6px !important; }
    .action-row > button { width: 100%; }
}
"""

def demo_theme():
    theme = gr.themes.Soft(font=['Arial', 'sans-serif'], font_mono=['Consolas', 'monospace'])
    theme.set(
        border_color_primary='#494055', border_color_accent='#8d72a5',
        border_color_accent_subdued='#62516f', panel_border_color='#494055',
        block_label_border_color='#494055',
        button_primary_border_color='#71558b', button_primary_border_color_hover='#81639c',
        button_secondary_border_color='#494055', button_secondary_border_color_hover='#62516f',
        input_border_color_hover='#62516f', input_border_color_focus='#9b7fb4',
        body_background_fill='#19171f', body_text_color='#ddd6e5',
        body_text_color_subdued='#b3a8bf',
        background_fill_primary='#24212d', background_fill_secondary='#302b3b',
        block_background_fill='#24212d', block_border_color='#494055',
        block_label_background_fill='#24212d', block_label_text_color='#cbc0d6',
        block_title_text_color='#ddd6e5',
        input_background_fill='#302b3b', input_border_color='#5c506b',
        input_placeholder_color='#ab9bb8',
        table_border_color='#494055', table_even_background_fill='#2b2635',
        table_odd_background_fill='#24212d',
        button_primary_background_fill='#71558b',
        button_primary_background_fill_hover='#81639c',
        button_primary_text_color='#f1ebf6',
        button_secondary_background_fill='#302b3b', button_secondary_text_color='#ddd6e5',
        slider_color='#a28aba',
    )
    values = theme.to_dict()['theme']
    # Use the same muted palette for either browser theme.
    theme.set(**{key: values[key[:-5]] for key in values
                 if key.endswith('_dark') and key[:-5] in values})
    return theme


ROOT = Path(__file__).resolve().parent
LABELS = {'dast': 'DAST', 'transformer': 'Transformer', 'fixed_gate': 'Fixed gate',
          'no_pad': 'No AR prior', 'no_film': 'No FiLM', 'lstm': 'LSTM', 'gru': 'GRU',
          'rnn': 'RNN', 'cnn': 'CNN', 'ar': 'AR(4), unbounded', 'ar_bounded': 'AR(4), bounded',
          'kalman': 'Complex Kalman', 'no_prediction': 'Hold last'}


def compare(bundle, velocity, index, seed, rf, antenna, comparison='Model benchmark'):
    velocity, index, seed, rf, antenna = map(int, [velocity, index, seed, rf, antenna])
    history, target = bundle.example(velocity, index)
    if comparison not in ['Model benchmark', 'Component ablations']:
        raise ValueError('Unknown comparison group.')
    if not 0 <= rf <= 7 or not 0 <= antenna <= 3:
        raise ValueError('Coefficient index outside the channel matrix.')
    variants = (['dast', 'transformer', 'lstm', 'gru', 'rnn', 'cnn'] if comparison == 'Model benchmark'
                else ['dast', 'transformer', 'fixed_gate', 'no_pad', 'no_film'])
    predictions = {v: bundle.predict(f'{v}_seed{seed}', history[None])[0] for v in variants}
    if comparison == 'Model benchmark':
        predictions.update(ar=ar_predict(history[None])[0], ar_bounded=ar_predict(history[None], bounded=True)[0],
                           kalman=kalman_predict(history[None])[0], no_prediction=hold_last(history[None])[0])
    colors = plt.get_cmap('tab10').colors
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    fig.patch.set_facecolor('#24212d')
    for ax in axes:
        ax.set_facecolor('#24212d')
        ax.tick_params(colors='#cbc0d6')
        ax.xaxis.label.set_color('#cbc0d6')
        ax.yaxis.label.set_color('#cbc0d6')
        ax.title.set_color('#e2d9eb')
        for spine in ax.spines.values():
            spine.set_color('#71627f')
    observed = np.linalg.norm(history[:, rf, antenna], axis=-1)
    future = np.linalg.norm(target[:, rf, antenna], axis=-1)
    axes[0].plot(np.arange(-9, 1), observed, 'o-', color='#a396af', label='Noisy history')
    axes[0].plot(np.arange(1, 6), future, 'o-', color='#eee7f5', label='Clean future')
    axes[0].axvline(.5, color='#888888', linestyle='--', linewidth=1)
    rows = []
    for (variant, prediction), color in zip(predictions.items(), colors):
        ratios = nmse_per_frame(prediction, target)
        score = float(10 * np.log10(max(float(ratios.mean()), 1e-12)))
        axes[0].plot(np.arange(1, 6), np.linalg.norm(prediction[:, rf, antenna], axis=-1),
                     '.--', color=color, label=LABELS[variant])
        axes[1].plot(np.arange(1, 6), 10 * np.log10(ratios.clip(1e-12)), 'o-',
                     color=color, label=LABELS[variant])
        rows.append([LABELS[variant], f'{score:.3f}', str(seed) if variant in variants else '-'])
    axes[0].set(xlabel='Frame relative to last observation', ylabel='Channel coefficient magnitude',
                title=f'RF chain {rf} / UE antenna {antenna}')
    axes[1].set(xlabel='Forecast horizon (frames)', ylabel='NMSE (dB; all coefficients)', xticks=range(1, 6),
                title='Single-sequence prediction error')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, ncol=2, facecolor='#302b3b', edgecolor='#5c506b', labelcolor='#e2d9eb')
    plt.close(fig)
    caption = (f'Simulated example {index} | {velocity} km/h | seed {seed}. '
               'NMSE is calculated over all channel coefficients; lower is better.')
    return fig, rows, caption


def create_app(bundle):
    with gr.Blocks(title='DAST · MIMO Channel Prediction') as demo:
        gr.Markdown('# MIMO Channel Prediction\n'
                    'Compare DAST with Transformer, LSTM, GRU, RNN, CNN and classical references. '
                    '**Simulation only:** Sionna CDL-A, history-only observations, 10 → 5 frames.')
        with gr.Group(elem_classes='controls-panel'):
            with gr.Row(equal_height=True):
                comparison = gr.Dropdown(['Model benchmark', 'Component ablations'],
                                         value='Model benchmark', label='Comparison',
                                         scale=1, min_width=220)
                velocity = gr.Dropdown(bundle.manifest['velocities_kmh'], value=70,
                                       label='Velocity (km/h)', scale=1, min_width=220)
                seed = gr.Dropdown([42, 123, 2024], value=42, label='Training seed',
                                   scale=1, min_width=220)
            with gr.Row(equal_height=True):
                example = gr.Slider(0, bundle.manifest['examples_per_velocity'] - 1,
                                    value=0, step=1, label='Test example', scale=1, min_width=220)
                rf = gr.Slider(0, 7, value=0, step=1, label='RF chain (plot only)',
                               scale=1, min_width=220)
                antenna = gr.Slider(0, 3, value=0, step=1, label='UE antenna (plot only)',
                                    scale=1, min_width=220)
            with gr.Row(elem_classes='action-row'):
                button = gr.Button('Run prediction', variant='primary', size='lg',
                                   scale=0, min_width=200)
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=320):
                plot = gr.Plot(show_label=False, elem_classes=['result-panel', 'result-plot'])
            with gr.Column(scale=1, min_width=330):
                table = gr.Dataframe(headers=['Model', 'NMSE (dB)', 'Seed'],
                                     show_label=False, interactive=False,
                                     max_height=420, wrap=True, buttons=[],
                                     column_widths=['48%', '27%', '25%'],
                                     elem_classes=['result-panel', 'results-table'])
        caption = gr.Markdown()
        def predict(*args):
            return compare(bundle, *args)
        button.click(predict, [velocity, example, seed, rf, antenna, comparison], [plot, table, caption], api_name='predict')
        demo.load(predict, [velocity, example, seed, rf, antenna, comparison], [plot, table, caption])
    return demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'artifacts/benchmark-v1')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    torch.set_num_threads(2)
    bundle = InferenceBundle(args.bundle)
    create_app(bundle).queue(default_concurrency_limit=1).launch(
        server_name='127.0.0.1', server_port=args.port, share=False, inbrowser=False,
        footer_links=[], css=APP_CSS, theme=demo_theme())
