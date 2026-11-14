"""Bundle integrity and inference contract tests using small synthetic fixtures."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import InferenceBundle, nmse_per_frame
from src.models.utils import build_model


class InferenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        config = dict(p_frames=10, q_frames=5, input_dim=64, d_model=8, n_heads=2,
                      num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=16, dropout=0.)
        model = build_model('transformer', config)
        torch.save({'schema_version': 1, 'state_dict': model.state_dict()}, self.directory / 'model.pt')
        rng = np.random.default_rng(42)
        np.savez(self.directory / 'examples.npz',
                 history_v30=rng.normal(size=(2, 10, 8, 4, 2)).astype('float32'),
                 target_v30=rng.normal(size=(2, 5, 8, 4, 2)).astype('float32'))
        self.manifest = dict(schema_version=1, observation_policy='history_only_v1', normalization_scale=2.,
                             velocities_kmh=[30], examples_per_velocity=2,
                             models=[dict(run_id='tiny', model='transformer', weights='model.pt', config=config)],
                             model_source_sha256={'src/models/utils.py': self.digest(ROOT / 'src/models/utils.py')},
                             files_sha256={n: self.digest(self.directory / n) for n in ['model.pt', 'examples.npz']})
        self.save_manifest()

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def save_manifest(self):
        (self.directory / 'manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_prediction_and_example_contract(self):
        bundle = InferenceBundle(self.directory)
        history, target = bundle.example(30, 0)
        result = bundle.predict('tiny', history[None])
        self.assertEqual(result.shape, (1, 5, 8, 4, 2))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(target.shape, (5, 8, 4, 2))
        self.assertIs(bundle.model('tiny'), bundle.model('tiny'))
        history[:] = 0
        self.assertGreater(np.abs(bundle.example(30, 0)[0]).sum(), 0)

    def test_modified_weights_rejected(self):
        with (self.directory / 'model.pt').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            InferenceBundle(self.directory)

    def test_wrong_policy_rejected(self):
        self.manifest['observation_policy'] = 'legacy_window'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'policy'):
            InferenceBundle(self.directory)

    def test_model_source_mismatch_rejected(self):
        self.manifest['model_source_sha256']['src/models/utils.py'] = '0' * 64
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'source differs'):
            InferenceBundle(self.directory)

    def test_path_escape_rejected(self):
        self.manifest['files_sha256']['../outside.pt'] = '0' * 64
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'escapes'):
            InferenceBundle(self.directory)

    def test_invalid_inputs_rejected(self):
        bundle = InferenceBundle(self.directory)
        with self.assertRaises(ValueError):
            bundle.example(30, -1)
        with self.assertRaises(ValueError):
            bundle.example(30, .5)
        with self.assertRaises(ValueError):
            bundle.predict('unknown', np.ones((1, 10, 8, 4, 2)))
        with self.assertRaises(ValueError):
            bundle.predict('tiny', np.ones((1, 9, 8, 4, 2)))
        with self.assertRaises(ValueError):
            bundle.predict('tiny', np.full((1, 10, 8, 4, 2), np.nan))

    def test_nmse_definition(self):
        target = np.ones((2, 5, 8, 4, 2))
        np.testing.assert_array_equal(nmse_per_frame(target, target), np.zeros((2, 5)))
        np.testing.assert_array_equal(nmse_per_frame(np.zeros_like(target), target), np.ones((2, 5)))
        with self.assertRaises(ValueError):
            nmse_per_frame(target[:1], target)


if __name__ == '__main__':
    unittest.main()
