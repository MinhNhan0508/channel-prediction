"""Load every exported model and run CPU predictions on the fixed demo examples."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import InferenceBundle, nmse_per_frame


def smoke(directory):
    torch.set_num_threads(2)
    bundle = InferenceBundle(directory)
    rows = []
    for run_id in bundle.entries:
        for velocity in bundle.manifest['velocities_kmh']:
            history = bundle.examples[f'history_v{velocity}']
            target = bundle.examples[f'target_v{velocity}']
            prediction = bundle.predict(run_id, history)
            rows.append(dict(run_id=run_id, velocity_kmh=velocity, examples=len(history),
                             nmse_db=float(10 * np.log10(nmse_per_frame(prediction, target).mean()))))
        print(f'PASS {run_id}', flush=True)
    return dict(scope='CPU inference smoke test on 40 fixed examples; not the full benchmark.',
                models=len(bundle.entries), predictions=sum(r['examples'] for r in rows), scores=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'artifacts/benchmark-v1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = smoke(args.bundle)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"PASS: {result['models']} models, {result['predictions']} example predictions")
