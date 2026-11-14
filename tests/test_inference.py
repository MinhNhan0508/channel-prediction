"""Bundle integrity and inference contract tests using small synthetic fixtures."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import shutil

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import InferenceBundle, nmse_per_frame
from src.models.utils import build_model
from tools.merge_bundles import merge


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
        (self.directory / 'LICENSE').write_text('MIT test fixture')
        self.manifest.update(license='MIT', dataset={'manifest_sha256': 'same'}, experiment_id='test',
                             experiment_sha256='test-hash', tensor_shapes={'history': [10, 8, 4, 2], 'target': [5, 8, 4, 2]})
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

    def second_bundle(self):
        directory = self.directory / 'second'
        directory.mkdir()
        for source, destination in [('model.pt', 'second.pt'), ('examples.npz', 'examples.npz'), ('LICENSE', 'LICENSE')]:
            shutil.copy2(self.directory / source, directory / destination)
        manifest = json.loads(json.dumps(self.manifest))
        manifest['models'][0].update(run_id='second', weights='second.pt')
        manifest['files_sha256']['second.pt'] = manifest['files_sha256'].pop('model.pt')
        return directory, manifest

    def test_compatible_bundles_merge_and_predict(self):
        second, manifest = self.second_bundle()
        (second / 'manifest.json').write_text(json.dumps(manifest))
        merged = self.directory / 'merged'
        merge([self.directory, second], merged)
        bundle = InferenceBundle(merged)
        self.assertEqual(len(bundle.entries), 2)
        history, _ = bundle.example(30, 0)
        np.testing.assert_array_equal(bundle.predict('tiny', history[None]), bundle.predict('second', history[None]))

    def test_merge_rejects_different_normalization(self):
        second, manifest = self.second_bundle()
        manifest['normalization_scale'] = 3.
        (second / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'normalization_scale'):
            merge([self.directory, second], self.directory / 'merged')


if __name__ == '__main__':
    unittest.main()
