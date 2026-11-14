"""Behavioral checks for classical benchmark references."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.references import ar_predict, hold_last, kalman_predict
from src.models.pad import pad_predict
from src.models.pad_torch import pad_extrapolate


class ReferencesTest(unittest.TestCase):
    def setUp(self):
        self.history = np.random.default_rng(17).normal(size=(3, 10, 8, 4, 2)).astype(np.float32)

    def test_zero_history_is_zero_and_finite(self):
        zero = np.zeros_like(self.history)
        for method in [hold_last, ar_predict, kalman_predict]:
            np.testing.assert_array_equal(method(zero), np.zeros((3, 5, 8, 4, 2)))

    def test_hold_last_repeats_last_noisy_observation(self):
        expected = np.repeat(self.history[:, -1:], 5, axis=1)
        np.testing.assert_array_equal(hold_last(self.history), expected)

    def test_unbounded_ar_matches_original_numpy_baseline(self):
        np.testing.assert_allclose(ar_predict(self.history), pad_predict(self.history), rtol=5e-4, atol=2e-4)

    def test_bounded_ar_matches_neural_prior(self):
        expected = pad_extrapolate(torch.from_numpy(self.history)).numpy()
        np.testing.assert_allclose(ar_predict(self.history, bounded=True), expected, rtol=5e-4, atol=2e-4)

    def test_single_complex_exponential_extrapolates(self):
        time = np.arange(15)
        signal = np.exp(.2j * time)
        full = np.broadcast_to(np.stack([signal.real, signal.imag], -1)[None, :, None, None], (1, 15, 8, 4, 2)).astype(np.float32)
        for method in [ar_predict, kalman_predict]:
            np.testing.assert_allclose(method(full[:, :10]), full[:, 10:], atol=.02)

    def test_batch_order_and_coefficient_permutation_are_independent(self):
        for method in [ar_predict, kalman_predict]:
            expected = method(self.history)
            np.testing.assert_allclose(method(self.history[::-1]), expected[::-1])
            np.testing.assert_allclose(method(self.history[:, :, ::-1, ::-1]), expected[:, :, ::-1, ::-1])

    def test_nonfinite_and_wrong_shapes_rejected(self):
        for method in [hold_last, ar_predict, kalman_predict]:
            with self.assertRaises(ValueError):
                method(np.full_like(self.history, np.nan))
            with self.assertRaises(ValueError):
                method(self.history[:, :9])


if __name__ == '__main__':
    unittest.main()
