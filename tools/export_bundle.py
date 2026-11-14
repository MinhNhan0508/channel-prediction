"""Export inference-only weights and fixed examples from a completed experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.provenance import data_content_fingerprint, dataset_provenance


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(results, data, output, experiment_source=None):
    results, data, output = map(Path, [results, data, output])
    experiment_source = Path(experiment_source) if experiment_source else ROOT
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty export directory.')
    experiment = json.loads((results / 'experiment.json').read_text(encoding='utf-8'))
    plan = experiment['plan']
    identity = hashlib.sha256(json.dumps(experiment, sort_keys=True).encode()).hexdigest()
    if dataset_provenance(data) != experiment['data_provenance']:
        raise ValueError('Dataset manifest differs from training.')
    metadata = json.loads((data / 'metadata.json').read_text(encoding='utf-8'))
    if data_content_fingerprint(data) != metadata['data_content_sha256']:
        raise ValueError('Dataset bytes changed.')
    for name, expected in experiment['source_sha256'].items():
        if digest(experiment_source / name) != expected:
            raise ValueError(f'Experiment source changed: {name}')
    for path in (ROOT / 'src/models').rglob('*.py'):
        if digest(path) != digest(experiment_source / path.relative_to(ROOT)):
            raise ValueError('Current inference model source differs from the archived experiment.')
    models, checkpoints = [], []
    # Validate every run before writing any artifact.
    for item in plan['runs']:
        run_id = f"{item['variant']}_seed{item['seed']}"
        run = results / 'runs' / run_id
        record = json.loads((run / 'training.json').read_text(encoding='utf-8'))
        path = run / 'checkpoints/checkpoint_best.pt'
        if record['experiment_sha256'] != identity or digest(path) != record['checkpoint_sha256']:
            raise ValueError(f'Checkpoint identity differs: {run_id}')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['config']['data_provenance'] != experiment['data_provenance'] or saved['config']['seed'] != item['seed']:
            raise ValueError(f'Checkpoint configuration differs: {run_id}')
        checkpoints.append(saved['model_state_dict'])
        models.append(dict(run_id=run_id, variant=item['variant'], seed=item['seed'], model=record['model'],
                           config=experiment['models'][item['variant']], weights=run_id + '.pt',
                           params=record['params'], best_epoch=record['best_epoch'],
                           training_checkpoint_sha256=record['checkpoint_sha256']))
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for entry, weights in zip(models, checkpoints):
        path = output / entry['weights']
        torch.save({'schema_version': 1, 'state_dict': weights}, path)
        files[path.name] = digest(path)
    fixture = {}
    for velocity in plan['test_velocities']:
        paths = sorted((data / f'test_v{velocity}').glob('sample_*.npz'))[:8]
        if len(paths) != 8:
            raise ValueError('Need at least eight test examples per velocity.')
        history, target = [], []
        for path in paths:
            with np.load(path, allow_pickle=False) as sample:
                history.append(sample['He_history'])
                target.append(sample['He_future'])
        fixture[f'history_v{velocity}'] = np.stack(history)
        fixture[f'target_v{velocity}'] = np.stack(target)
    np.savez_compressed(output / 'examples.npz', **fixture)
    files['examples.npz'] = digest(output / 'examples.npz')
    shutil.copy2(ROOT / 'LICENSE', output / 'LICENSE')
    files['LICENSE'] = digest(output / 'LICENSE')
    with np.load(data / 'normalize_stats.npz', allow_pickle=False) as stats:
        scale = float(stats['scale'])
    manifest = dict(schema_version=1, license='MIT', experiment_id=plan['id'], observation_policy='history_only_v1',
                    experiment_sha256=identity, dataset=experiment['data_provenance'], normalization_scale=scale,
                    models=models, files_sha256=files, velocities_kmh=plan['test_velocities'], examples_per_velocity=8,
                    example_selection='First 8 lexicographically sorted test sequences per velocity; no score filtering.',
                    runtime=experiment['runtime'], tensor_shapes={'history': [10, 8, 4, 2], 'target': [5, 8, 4, 2]},
                    model_source_sha256={p.relative_to(ROOT).as_posix(): digest(p)
                                         for p in sorted((ROOT / 'src/models').rglob('*.py'))})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Exported {len(models)} inference models and 40 fixed examples.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--experiment-source', type=Path, help='Archived frozen source; defaults to current checkout.')
    args = parser.parse_args()
    export(args.results, args.data, args.output, args.experiment_source)
