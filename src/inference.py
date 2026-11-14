"""Verified inference bundles for the history-only channel-prediction experiment."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.models.utils import build_model, is_spatial_model


class InferenceBundle:
    def __init__(self, directory, device='cpu'):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / 'manifest.json').read_text(encoding='utf-8'))
        manifest = self.manifest
        if manifest.get('schema_version') != 1 or manifest.get('observation_policy') != 'history_only_v1':
            raise ValueError('Unsupported bundle schema or observation policy.')
        source_root = Path(__file__).resolve().parents[1]
        if not manifest.get('model_source_sha256'):
            raise ValueError('Missing model source identity.')
        for name, digest in manifest['model_source_sha256'].items():
            path = (source_root / name).resolve()
            if not path.is_relative_to(source_root / 'src/models') or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Model source differs from exported experiment: {name}')
        for name, digest in manifest.get('runtime_source_sha256', {}).items():
            path = (source_root / name).resolve()
            if not path.is_relative_to(source_root / 'src') or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Inference source differs from bundle: {name}')
        self.scale = float(manifest['normalization_scale'])
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Invalid normalization scale.')
        self.device = torch.device(device)
        self.models = {}
        self.entries = {m['run_id']: m for m in manifest['models']}
        if len(self.entries) != len(manifest['models']):
            raise ValueError('Duplicate model identifier.')
        for name in manifest['files_sha256']:
            self._verified_path(name)
        for entry in self.entries.values():
            if entry['weights'] not in manifest['files_sha256']:
                raise ValueError('Model weights lack an integrity digest.')
        with np.load(self._verified_path('examples.npz'), allow_pickle=False) as archive:
            self.examples = {name: archive[name].copy() for name in archive.files}
        for velocity in manifest['velocities_kmh']:
            for field, frames in [('history', 10), ('target', 5)]:
                array = self.examples[f'{field}_v{velocity}']
                if array.shape != (manifest['examples_per_velocity'], frames, 8, 4, 2):
                    raise ValueError('Unexpected example tensor shape.')
                if not np.isfinite(array).all():
                    raise ValueError('Nonfinite example.')

    def _verified_path(self, name):
        path = (self.directory / name).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError('Artifact path escapes bundle directory.')
        expected = self.manifest['files_sha256'].get(name)
        if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Artifact integrity check failed: {name}')
        return path

    def model(self, run_id):
        if run_id not in self.entries:
            raise ValueError(f'Unknown run: {run_id}')
        if run_id not in self.models:
            entry = self.entries[run_id]
            saved = torch.load(self._verified_path(entry['weights']), map_location='cpu', weights_only=True)
            if saved.get('schema_version') != 1 or set(saved) != {'schema_version', 'state_dict'}:
                raise ValueError('Expected inference-only weights.')
            model = build_model(entry['model'], entry['config'])
            model.load_state_dict(saved['state_dict'], strict=True)
            self.models[run_id] = model.to(self.device).eval()
        return self.models[run_id]

    def predict(self, run_id, history):
        history = np.asarray(history, dtype=np.float32)
        if history.ndim != 5 or history.shape[1:] != (10, 8, 4, 2) or not len(history):
            raise ValueError('Expected nonempty history shaped (batch, 10, 8, 4, 2).')
        if not np.isfinite(history).all():
            raise ValueError('History contains nonfinite values.')
        entry = self.entries.get(run_id)
        if entry is None:
            raise ValueError(f'Unknown run: {run_id}')
        x = torch.from_numpy(history / self.scale).to(self.device)
        if not is_spatial_model(entry['model']):
            x = x.flatten(2)
        with torch.inference_mode():
            output = self.model(run_id)(x).reshape(-1, 5, 8, 4, 2).cpu().numpy() * self.scale
        if not np.isfinite(output).all():
            raise FloatingPointError('Nonfinite model output.')
        return output

    def example(self, velocity, index):
        if velocity not in self.manifest['velocities_kmh']:
            raise ValueError('Unknown velocity.')
        if isinstance(index, bool) or int(index) != index or not 0 <= index < self.manifest['examples_per_velocity']:
            raise ValueError('Invalid example index.')
        return (self.examples[f'history_v{velocity}'][int(index)].copy(),
                self.examples[f'target_v{velocity}'][int(index)].copy())


def nmse_per_frame(prediction, target):
    """Linear NMSE over all spatial coefficients and real/imaginary components."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or target.shape[-3:] != (8, 4, 2):
        raise ValueError('Prediction and target shapes must agree.')
    return np.sum((prediction - target) ** 2, axis=(-3, -2, -1)) / np.maximum(
        np.sum(target ** 2, axis=(-3, -2, -1)), 1e-12)
