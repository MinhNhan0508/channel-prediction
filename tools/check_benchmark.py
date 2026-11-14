"""Verify the complete model comparison against published per-frame NMSE arrays."""
import argparse
import json
from pathlib import Path

import numpy as np
from tools.report_benchmark import collect


def check(directory):
    directory = Path(directory)
    rows, velocities, records, _ = collect(directory / 'input/formal.json',
        directory / 'input/baselines.json', directory / 'input/references/manifest.json')
    for record in records:
        path = directory / 'raw' / f"{record['variant']}_seed{record['seed']}.npz"
        with np.load(path, allow_pickle=False) as archive:
            raw = archive[f"nmse_v{record['velocity_kmh']}"]
            if raw.shape != (record['n'], 5) or not np.isfinite(raw).all() or (raw < 0).any():
                raise ValueError('Invalid learned-model raw NMSE.')
            if not np.isclose(10 * np.log10(raw.mean()), record['nmse_db'], atol=1e-7):
                raise ValueError('Learned-model score does not match raw NMSE.')
    published = json.loads((directory / 'benchmark.json').read_text(encoding='utf-8'))
    if published['summary'] != rows:
        raise ValueError('Published aggregate table differs from verified inputs.')
    print(f'PASS: {len(records)} neural groups, 20 reference groups, {len(rows)} aggregate rows.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1] / 'docs/benchmark-results')
    check(parser.parse_args().results)
