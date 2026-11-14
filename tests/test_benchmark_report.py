"""The main comparison must not silently omit a model, seed or reference."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.report_benchmark import collect


class BenchmarkReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        plan = dict(seeds=[42, 123, 2024], test_velocities=[30, 50, 70, 90, 120],
                    expected_split_counts={f'test_v{v}': 500 for v in [30, 50, 70, 90, 120]})
        experiment = dict(data_provenance={'manifest_sha256': 'shared', 'observation_policy': 'history_only_v1'},
                          training_config={'n_epochs': 300, 'batch_size': 64}, plan=plan,
                          source_sha256={'src/models/example.py': 'same'})
        self.formal = dict(experiment=copy.deepcopy(experiment), individual=[])
        self.baseline = dict(experiment=copy.deepcopy(experiment), individual=[])
        for payload, methods in [(self.formal, ['dast', 'transformer']),
                                  (self.baseline, ['lstm', 'gru', 'rnn', 'cnn'])]:
            payload['individual'] = [dict(variant=m, seed=s, velocity_kmh=v, n=500, nmse_db=-2.)
                                     for m in methods for s in plan['seeds'] for v in plan['test_velocities']]
        self.reference = dict(state='complete', data_provenance=copy.deepcopy(experiment['data_provenance']),
                              scores=[dict(variant=m, velocity_kmh=v, n=500, nmse_db=1.)
                                      for m in ['ar', 'ar_bounded', 'kalman', 'no_prediction'] for v in plan['test_velocities']])
        for method in ['ar', 'ar_bounded', 'kalman', 'no_prediction']:
            np.savez(self.folder / (method + '.npz'), **{
                f'nmse_v{v}': np.full((500, 5), 10 ** .1) for v in plan['test_velocities']})

    def evaluate(self):
        paths = []
        for name, payload in [('formal', self.formal), ('baseline', self.baseline), ('reference', self.reference)]:
            path = self.folder / (name + '.json')
            path.write_text(json.dumps(payload), encoding='utf-8')
            paths.append(path)
        return collect(*paths)

    def test_complete_report_has_six_neural_models_and_four_references(self):
        summary, _, records, _ = self.evaluate()
        self.assertEqual(len(summary), 50)
        self.assertEqual(len(records), 90)
        self.assertTrue(all(row['runs'] == 1 and row['std_population_db'] is None
                            for row in summary if row['kind'] == 'deterministic'))

    def test_missing_cnn_seed_rejected(self):
        self.baseline['individual'] = [r for r in self.baseline['individual'] if not
                                      (r['variant'] == 'cnn' and r['seed'] == 2024)]
        with self.assertRaisesRegex(ValueError, 'all six'):
            self.evaluate()

    def test_mismatched_dataset_rejected(self):
        self.baseline['experiment']['data_provenance']['manifest_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'different datasets'):
            self.evaluate()

    def test_mismatched_recipe_rejected(self):
        self.baseline['experiment']['training_config']['n_epochs'] = 500
        with self.assertRaisesRegex(ValueError, 'recipes'):
            self.evaluate()

    def test_changed_model_implementation_rejected(self):
        self.baseline['experiment']['source_sha256']['src/models/example.py'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Core source'):
            self.evaluate()

    def test_duplicated_deterministic_reference_rejected(self):
        self.reference['scores'].append(self.reference['scores'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.evaluate()

    def test_tampered_reference_score_rejected(self):
        self.reference['scores'][0]['nmse_db'] = -20.
        with self.assertRaisesRegex(ValueError, 'raw NMSE'):
            self.evaluate()


if __name__ == '__main__':
    unittest.main()
