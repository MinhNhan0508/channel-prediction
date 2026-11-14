"""Recompute all published scores from saved per-sequence/per-horizon NMSE arrays."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def check(directory):
    directory = Path(directory)
    provenance = json.loads((directory / 'provenance.json').read_text(encoding='utf-8'))
    plan = provenance['experiment']['plan']
    with (directory / 'scores.csv').open(encoding='utf-8', newline='') as stream:
        scores = list(csv.DictReader(stream))
    index = {(r['variant'], int(r['seed']), int(r['velocity_kmh'])): r for r in scores}
    expected = {(v['id'], s, k) for v in plan['variants'] for s in plan['seeds'] for k in plan['test_velocities']}
    if set(index) != expected or len(scores) != len(expected):
        raise ValueError('Missing, duplicate or unexpected score group.')
    for variant in plan['variants']:
        for seed in plan['seeds']:
            with np.load(directory / 'raw' / f"{variant['id']}_seed{seed}.npz", allow_pickle=False) as archive:
                for velocity in plan['test_velocities']:
                    raw = archive[f'nmse_v{velocity}']
                    n = plan['expected_split_counts'][f'test_v{velocity}']
                    if raw.shape != (n, 5) or not np.isfinite(raw).all() or (raw < 0).any():
                        raise ValueError('Invalid raw NMSE values.')
                    row = index[variant['id'], seed, velocity]
                    if int(row['n']) != n or not np.isclose(10 * np.log10(raw.mean()), float(row['nmse_db']), atol=1e-7):
                        raise ValueError('Raw NMSE disagrees with published score.')
    with (directory / 'summary.csv').open(encoding='utf-8', newline='') as stream:
        summary = list(csv.DictReader(stream))
    groups = {(r['variant'], int(r['velocity_kmh'])) for r in summary}
    if groups != {(v['id'], k) for v in plan['variants'] for k in plan['test_velocities']} or len(summary) != len(groups):
        raise ValueError('Incomplete summary.')
    for row in summary:
        values = [float(index[row['variant'], s, int(row['velocity_kmh'])]['nmse_db']) for s in plan['seeds']]
        if not np.isclose(np.mean(values), float(row['mean_db']), atol=1e-10):
            raise ValueError('Incorrect summary mean.')
        if not np.isclose(np.std(values), float(row['std_population_db']), atol=1e-10):
            raise ValueError('Incorrect summary standard deviation.')
    print(f'PASS: {len(scores)} individual scores and {len(summary)} aggregates recomputed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1] / 'docs/results')
    check(parser.parse_args().results)
