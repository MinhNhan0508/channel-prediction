"""Render tables and figures from a complete multi-seed evaluation."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = {'dast': 'DAST', 'transformer': 'Transformer', 'fixed_gate': 'Fixed gate',
          'no_pad': 'No AR prior', 'no_film': 'No FiLM'}
COLORS = ['#1668b2', '#e07a24', '#2c9255', '#b34b68', '#7953a8']


def report(evaluation_path, output):
    payload = json.loads(Path(evaluation_path).read_text(encoding='utf-8'))
    plan = payload['experiment']['plan']
    variants = [v['id'] for v in plan['variants']]
    velocities = plan['test_velocities']
    expected = {(v, s, k) for v in variants for s in plan['seeds'] for k in velocities}
    records = payload['individual']
    actual = {(r['variant'], r['seed'], r['velocity_kmh']) for r in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError('Evaluation must include each declared variant, seed and velocity exactly once.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in variants:
        for velocity in velocities:
            group = [r['nmse_db'] for r in records if r['variant'] == variant and r['velocity_kmh'] == velocity]
            if not np.isfinite(group).all():
                raise ValueError('Nonfinite score.')
            rows.append({'variant': variant, 'velocity_kmh': velocity,
                         'mean_db': float(np.mean(group)), 'std_population_db': float(np.std(group))})
    for filename, data in [('scores.csv', records), ('summary.csv', rows)]:
        with (output / filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(8.4, 4.8), layout='constrained')
    for variant, color in zip(variants, COLORS):
        group = [r for r in rows if r['variant'] == variant]
        ax.errorbar(velocities, [r['mean_db'] for r in group],
                    yerr=[r['std_population_db'] for r in group],
                    marker='o', capsize=4, color=color, label=LABELS[variant])
    ax.set(xlabel='Velocity (km/h)', ylabel='Test NMSE (dB; lower is better)', xticks=velocities,
           title='Synthetic CDL-A channel prediction · 10 history / 5 forecast frames')
    ax.grid(alpha=0.2)
    ax.legend(ncol=2)
    fig.text(0.13, -0.025, 'Mean ± population SD over 3 training seeds; 500 test sequences per velocity.', fontsize=9)
    for ext in ['png', 'svg']:
        fig.savefig(output / f'nmse_by_velocity.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    # Positive differences mean the ablation performs worse than DAST.
    fig, ax = plt.subplots(figsize=(8.4, 4.6), layout='constrained')
    base = {r['velocity_kmh']: r['mean_db'] for r in rows if r['variant'] == 'dast'}
    x = np.arange(len(velocities))
    for offset, variant in enumerate(['fixed_gate', 'no_pad', 'no_film']):
        group = {r['velocity_kmh']: r['mean_db'] for r in rows if r['variant'] == variant}
        ax.bar(x + (offset - 1) * .24, [group[v] - base[v] for v in velocities], .24,
               label=LABELS[variant], color=COLORS[offset + 2])
    ax.axhline(0, color='#333333', linewidth=.8)
    ax.set(xticks=x, xticklabels=velocities, xlabel='Velocity (km/h)',
           ylabel='Ablation NMSE − DAST NMSE (dB)', title='Component ablations · negative values favor the ablation')
    ax.grid(axis='y', alpha=.2)
    ax.legend()
    for ext in ['png', 'svg']:
        fig.savefig(output / f'ablation_delta.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    lines = ['# Formal evaluation results', '',
             'Synthetic CDL-A, history-only observations. Lower NMSE is better.', '',
             'Each cell is mean ± population standard deviation in dB over seeds 42, 123 and 2024.',
             'Each seed is evaluated on the same 500 test sequences at each velocity.', '',
             '| Variant | ' + ' | '.join(f'{v} km/h' for v in velocities) + ' |',
             '|---|' + '---:|' * len(velocities)]
    for variant in variants:
        group = [r for r in rows if r['variant'] == variant]
        lines.append('| ' + LABELS[variant] + ' | ' + ' | '.join(
            f"{r['mean_db']:.3f} ± {r['std_population_db']:.3f}" for r in group) + ' |')
    lines += ['', '![Test NMSE](nmse_by_velocity.png)', '', '![Ablations](ablation_delta.png)', '',
              '## Interpretation', '',
              '- DAST improves on the vanilla Transformer in all five tested velocity scenarios.',
              '- Removing the AR prior loses most of that improvement under this recipe.',
              '- Removing FiLM gives nearly the same scores. These results do not establish a consistent FiLM benefit.',
              '- The fixed gate is slightly better at 90 and 120 km/h. Input-dependent gating does not win everywhere.',
              '- Small ablation differences and three training seeds do not establish statistical significance.',
              '- These scores do not measure real-channel generalization, BER, radio throughput or deployment latency.', '',
              '## Files and metric', '',
              '`scores.csv` contains all 75 individual scores; `summary.csv` contains 25 aggregate rows.',
              'For each seed, average the per-sample, per-horizon **linear** NMSE ratios before converting to dB.',
              'Then compute mean/std across the three seed scores. Error bars describe seed variability,',
              'not confidence intervals. Checkpoints are selected by validation NMSE, never by test score.', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Reported {len(records)} scores to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report(args.evaluation, args.output)
