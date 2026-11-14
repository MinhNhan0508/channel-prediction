"""Evaluate declared classical references once on the formal held-out sequences."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.provenance import data_content_fingerprint, dataset_provenance
from src.inference import nmse_per_frame
from src.references import ar_predict, hold_last, kalman_predict

POLICY = {
    'order': 4, 'ridge': 0.001, 'bounded_amplitude_factor': 3.0,
    'kalman': 'complex covariance, Joseph update; AR estimated from history',
    'kalman_q_r': 'each half of mean AR residual power plus 1e-6',
    'no_prediction': 'repeat last noisy observation for all five forecast frames',
    'seed_policy': 'deterministic references run once, not copied across training seeds',
}


def evaluate(data, experiment_path, output):
    data, output = Path(data), Path(output)
    experiment = json.loads(Path(experiment_path).read_text(encoding='utf-8'))
    provenance = dataset_provenance(data)
    if provenance != experiment['data_provenance']:
        raise ValueError('References and learned models must use the identical dataset manifest.')
    metadata = json.loads((data / 'metadata.json').read_text(encoding='utf-8'))
    if data_content_fingerprint(data) != metadata['data_content_sha256']:
        raise ValueError('Dataset bytes changed.')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty reference output directory.')
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(state='evaluating', data_provenance=provenance,
                    data_content_sha256=metadata['data_content_sha256'], policy=POLICY,
                    source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                   for name in ['src/references.py', 'src/inference.py', 'tools/evaluate_references.py']},
                    methods=['ar', 'ar_bounded', 'kalman', 'no_prediction'])
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    methods = {'ar': ar_predict, 'ar_bounded': lambda h: ar_predict(h, bounded=True),
               'kalman': kalman_predict, 'no_prediction': hold_last}
    arrays = {method: {} for method in methods}
    records = []
    for velocity in experiment['plan']['test_velocities']:
        histories, targets = [], []
        for path in sorted((data / f'test_v{velocity}').glob('sample_*.npz')):
            with np.load(path, allow_pickle=False) as sample:
                histories.append(sample['He_history'])
                targets.append(sample['He_future'])
        history, target = np.stack(histories), np.stack(targets)
        expected = experiment['plan']['expected_split_counts'][f'test_v{velocity}']
        if len(history) != expected:
            raise ValueError('Incomplete test scenario.')
        for name, method in methods.items():
            prediction = np.concatenate([method(history[first:first+64]) for first in range(0, len(history), 64)])
            ratios = nmse_per_frame(prediction, target)
            if not np.isfinite(ratios).all():
                raise FloatingPointError('Nonfinite reference metric.')
            arrays[name][f'nmse_v{velocity}'] = ratios
            score = float(10 * np.log10(ratios.mean()))
            records.append(dict(variant=name, velocity_kmh=velocity, n=len(history), nmse_db=score))
            print(f'{name} v{velocity}: {score:.4f} dB', flush=True)
    for name, raw in arrays.items():
        np.savez(output / (name + '.npz'), **raw)
    manifest['state'] = 'complete'
    manifest['scores'] = records
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.data, args.experiment, args.output)
