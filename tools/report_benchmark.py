"""Combine compatible learned-model experiments and deterministic references."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NEURAL = ['dast', 'transformer', 'lstm', 'gru', 'rnn', 'cnn']
REFERENCES = ['ar', 'ar_bounded', 'kalman', 'no_prediction']
LABELS = dict(dast='DAST', transformer='Transformer', lstm='LSTM', gru='GRU', rnn='RNN', cnn='CNN',
              ar='AR(4), unbounded', ar_bounded='AR(4), bounded', kalman='Complex Kalman',
              no_prediction='Hold last / No-Prediction')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def collect(formal_path, baseline_path, reference_path):
    formal, baseline, reference = map(read, [formal_path, baseline_path, reference_path])
    left, right = formal['experiment'], baseline['experiment']
    if left['data_provenance'] != right['data_provenance'] or left['data_provenance'] != reference['data_provenance']:
        raise ValueError('Cannot combine different datasets or observation policies.')
    if left['training_config'] != right['training_config']:
        raise ValueError('Learned models used different training recipes.')
    for setting in ['seeds', 'test_velocities', 'expected_split_counts']:
        if left['plan'][setting] != right['plan'][setting]:
            raise ValueError(f'Experiment mismatch: {setting}')
    for name, digest in left['source_sha256'].items():
        if name.startswith(('src/models/', 'src/data/', 'src/training/')) and right['source_sha256'].get(name) != digest:
            raise ValueError(f'Core source differs: {name}')
    if reference.get('state') != 'complete':
        raise ValueError('Reference evaluation is incomplete.')
    velocities, seeds = left['plan']['test_velocities'], left['plan']['seeds']
    records = [r for payload in [formal, baseline] for r in payload['individual'] if r['variant'] in NEURAL]
    index = {(r['variant'], r['seed'], r['velocity_kmh']): r for r in records}
    expected = {(v, s, k) for v in NEURAL for s in seeds for k in velocities}
    if set(index) != expected or len(records) != len(expected):
        raise ValueError('Need all six learned models and every declared training seed.')
    ref_index = {(r['variant'], r['velocity_kmh']): r for r in reference['scores']}
    if set(ref_index) != {(v, k) for v in REFERENCES for k in velocities} or len(ref_index) != len(reference['scores']):
        raise ValueError('Incomplete or duplicate classical-reference scores.')
    for method in REFERENCES:
        with np.load(Path(reference_path).parent / (method + '.npz'), allow_pickle=False) as archive:
            for velocity in velocities:
                raw = archive[f'nmse_v{velocity}']
                expected_n = left['plan']['expected_split_counts'][f'test_v{velocity}']
                row = ref_index[method, velocity]
                if raw.shape != (expected_n, 5) or int(row['n']) != expected_n or not np.isfinite(raw).all() or (raw < 0).any():
                    raise ValueError('Invalid reference raw data.')
                if not np.isclose(10 * np.log10(raw.mean()), row['nmse_db'], atol=1e-7):
                    raise ValueError('Reference score does not match raw NMSE.')
    summary = []
    for method in NEURAL + REFERENCES:
        for velocity in velocities:
            if method in NEURAL:
                scores = [index[method, seed, velocity]['nmse_db'] for seed in seeds]
                summary.append(dict(variant=method, velocity_kmh=velocity, mean_db=float(np.mean(scores)),
                                    std_population_db=float(np.std(scores)), runs=len(seeds), kind='trained'))
            else:
                summary.append(dict(variant=method, velocity_kmh=velocity,
                                    mean_db=ref_index[method, velocity]['nmse_db'],
                                    std_population_db=None, runs=1, kind='deterministic'))
    if not all(np.isfinite(row['mean_db']) for row in summary):
        raise ValueError('Nonfinite benchmark score.')
    return summary, velocities, records, reference


def report(formal, baseline, references, output):
    rows, velocities, individual, reference = collect(formal, baseline, references)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'benchmark.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'benchmark.json').write_text(json.dumps(dict(summary=rows, individual_neural=individual,
        deterministic=reference, scope='Same history-only dataset and recipe; extension after initial formal study.'), indent=2), encoding='utf-8')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors = plt.get_cmap('tab10').colors
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout='constrained')
    for ax, methods, title in [(axes[0], NEURAL, 'Learned models'),
                                (axes[1], ['dast'] + REFERENCES, 'Classical references and DAST')]:
        for method in methods:
            group = [r for r in rows if r['variant'] == method]
            color = colors[(NEURAL + REFERENCES).index(method)]
            values = [r['mean_db'] for r in group]
            if method in NEURAL:
                ax.errorbar(velocities, values, yerr=[r['std_population_db'] for r in group],
                            marker='o', capsize=3, color=color, label=LABELS[method])
            else:
                ax.plot(velocities, values, '--s', color=color, label=LABELS[method])
        ax.set(title=title, xlabel='Velocity (km/h)', ylabel='Test NMSE (dB; lower is better)', xticks=velocities)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('MIMO channel prediction · 10 history / 5 forecast frames · synthetic CDL-A')
    fig.text(.1, -.025, 'Neural: mean ± population SD across 3 training seeds. Classical: one deterministic evaluation. 500 sequences / velocity.', fontsize=9)
    for extension in ['png', 'svg']:
        fig.savefig(output / f'model_comparison.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    text = ['# Model benchmark', '',
            'All methods use the same held-out history-only CDL-A observations and targets.',
            'Neural models share the same recipe and validation-based checkpoint selection.',
            'The four additional neural baselines extend the initial DAST/Transformer study.', '',
            '| Method | ' + ' | '.join(f'{v} km/h' for v in velocities) + ' |',
            '|---|' + '---:|' * len(velocities)]
    for method in NEURAL + REFERENCES:
        group = [r for r in rows if r['variant'] == method]
        cells = [f"{r['mean_db']:.3f} ± {r['std_population_db']:.3f}" if r['kind'] == 'trained'
                 else f"{r['mean_db']:.3f}" for r in group]
        text.append('| ' + LABELS[method] + ' | ' + ' | '.join(cells) + ' |')
    text += ['', '![Model comparison](model_comparison.png)', '',
             '- Standard deviations describe three training seeds, not confidence intervals.',
             '- Deterministic references are evaluated once; no artificial seed variability is added.',
             '- AR(4) is coefficient-wise least-squares prediction, not a full angular-delay PAD reproduction.',
             '- Bounded AR exposes the effect of the amplitude cap used by the DAST prior.',
             '- Kalman uses a complex Hermitian covariance and Joseph update. Its Q/R heuristic is fitted from history.',
             '  It differs from the original real-covariance approximation; old thesis scores are not reused.',
             '- No-Prediction repeats the last noisy observation; zero dB does not define this baseline.',
             '- Component ablations belong in a separate figure; they are not independent model families.', '']
    (output / 'README.md').write_text('\n'.join(text), encoding='utf-8')
    print(f'Wrote benchmark: {len(individual)} neural groups + {len(reference["scores"])} deterministic groups.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formal', required=True, type=Path)
    parser.add_argument('--baselines', required=True, type=Path)
    parser.add_argument('--references', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report(args.formal, args.baselines, args.references, args.output)
