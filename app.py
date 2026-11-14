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

ROOT = Path(__file__).resolve().parent
LABELS = {'dast': 'DAST', 'transformer': 'Transformer', 'fixed_gate': 'Fixed gate',
          'no_pad': 'No AR prior', 'no_film': 'No FiLM'}


def compare(bundle, velocity, index, seed, rf, antenna):
    velocity, index, seed, rf, antenna = map(int, [velocity, index, seed, rf, antenna])
    history, target = bundle.example(velocity, index)
    variants = ['dast', 'transformer', 'fixed_gate', 'no_pad', 'no_film']
    predictions = {v: bundle.predict(f'{v}_seed{seed}', history[None])[0] for v in variants}
    colors = ['#1668b2', '#e07a24', '#2c9255', '#b34b68', '#7953a8']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    observed = np.linalg.norm(history[:, rf, antenna], axis=-1)
    future = np.linalg.norm(target[:, rf, antenna], axis=-1)
    axes[0].plot(np.arange(-9, 1), observed, 'o-', color='#687782', label='Noisy history')
    axes[0].plot(np.arange(1, 6), future, 'o-', color='#222222', label='Clean future')
    axes[0].axvline(.5, color='#888888', linestyle='--', linewidth=1)
    rows = []
    for (variant, prediction), color in zip(predictions.items(), colors):
        ratios = nmse_per_frame(prediction, target)
        score = float(10 * np.log10(max(float(ratios.mean()), 1e-12)))
        axes[0].plot(np.arange(1, 6), np.linalg.norm(prediction[:, rf, antenna], axis=-1),
                     '.--', color=color, label=LABELS[variant])
        axes[1].plot(np.arange(1, 6), 10 * np.log10(ratios.clip(1e-12)), 'o-',
                     color=color, label=LABELS[variant])
        rows.append([LABELS[variant], f'{score:.3f}', seed])
    axes[0].set(xlabel='Frame relative to last observation', ylabel='Channel coefficient magnitude',
                title=f'RF chain {rf} / UE antenna {antenna}')
    axes[1].set(xlabel='Forecast horizon (frames)', ylabel='NMSE (dB; all coefficients)', xticks=range(1, 6),
                title='Single-sequence prediction error')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    plt.close(fig)
    caption = (f'**Synthetic test example {index} · {velocity} km/h · training seed {seed}.** '
               'The magnitude plot shows one coefficient; NMSE uses all 8 × 4 complex coefficients. '
               'A single example is not the aggregate benchmark. '
               'Forecast sampling interval: 0.125 ms. No model is trained from this page.')
    return fig, rows, caption


def create_app(bundle):
    with gr.Blocks(title='DAST · MIMO Channel Prediction') as demo:
        gr.Markdown('# MIMO Channel Prediction\n'
                    'Compare a Transformer with an autoregressive hybrid and controlled ablations. '
                    '**Simulation only:** Sionna CDL-A, history-only observations, 10 → 5 frames.')
        with gr.Row():
            velocity = gr.Dropdown(bundle.manifest['velocities_kmh'], value=70, label='Velocity (km/h)')
            example = gr.Slider(0, bundle.manifest['examples_per_velocity'] - 1, value=0, step=1, label='Test example index')
            seed = gr.Dropdown([42, 123, 2024], value=42, label='Training seed')
            rf = gr.Slider(0, 7, value=0, step=1, label='RF chain (plot only)')
            antenna = gr.Slider(0, 3, value=0, step=1, label='UE antenna (plot only)')
        button = gr.Button('Run prediction', variant='primary')
        plot = gr.Plot(label='Prediction and error')
        table = gr.Dataframe(headers=['Model', 'Example NMSE (dB)', 'Training seed'], interactive=False)
        caption = gr.Markdown()
        gr.Markdown('Examples are the first eight test sequences at each velocity, selected without score filtering. '
                    'Beam selection assumes ideal clean historical pilots. '
                    'See `docs/results/README.md` for the complete 15-run benchmark and limitations.')
        def predict(*args):
            return compare(bundle, *args)
        button.click(predict, [velocity, example, seed, rf, antenna], [plot, table, caption], api_name='predict')
        demo.load(predict, [velocity, example, seed, rf, antenna], [plot, table, caption])
    return demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'artifacts/formal-v1')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    torch.set_num_threads(2)
    bundle = InferenceBundle(args.bundle)
    create_app(bundle).queue(default_concurrency_limit=1).launch(
        server_name='127.0.0.1', server_port=args.port, share=False, inbrowser=False)
