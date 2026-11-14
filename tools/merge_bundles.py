"""Combine compatible inference bundles without mixing datasets or normalizers."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import InferenceBundle


def merge(directories, output):
    bundles = [InferenceBundle(path) for path in directories]
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory.')
    first = bundles[0]
    models, experiments = [], []
    for bundle in bundles:
        for key in ['dataset', 'normalization_scale', 'model_source_sha256', 'velocities_kmh',
                    'examples_per_velocity', 'tensor_shapes', 'license']:
            if bundle.manifest[key] != first.manifest[key]:
                raise ValueError(f'Incompatible bundles: {key}')
        if set(bundle.examples) != set(first.examples) or any(
                not np.array_equal(bundle.examples[k], first.examples[k]) for k in first.examples):
            raise ValueError('Bundles use different demo examples.')
        identity = {'id': bundle.manifest['experiment_id'], 'sha256': bundle.manifest['experiment_sha256']}
        experiments.append(identity)
        for entry in bundle.manifest['models']:
            record = copy.deepcopy(entry)
            record['experiment'] = identity
            models.append(record)
    if len({m['run_id'] for m in models}) != len(models) or len({m['weights'] for m in models}) != len(models):
        raise ValueError('Duplicate model identifiers or weight filenames.')
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for bundle in bundles:
        for entry in bundle.manifest['models']:
            name = entry['weights']
            shutil.copy2(bundle.directory / name, output / name)
            files[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    for name in ['examples.npz', 'LICENSE']:
        shutil.copy2(first.directory / name, output / name)
        files[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    manifest = copy.deepcopy(first.manifest)
    manifest.pop('experiment_sha256')
    manifest['experiment_id'] = 'benchmark_causal_v1'
    manifest.update(experiments=experiments, models=models, files_sha256=files,
                    runtime_source_sha256={'src/references.py': hashlib.sha256((ROOT / 'src/references.py').read_bytes()).hexdigest()})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Merged {len(models)} models with identical data, normalization and examples.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundles', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    merge(args.bundles, args.output)
