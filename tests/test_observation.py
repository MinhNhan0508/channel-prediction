"""Controlled arrays check observation causality, not radio performance."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.observation import make_observation


class ObservationTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(41)
        self.channel=rng.normal(size=(15,16,4))+1j*rng.normal(size=(15,16,4))

    def observe(self,channel=None,**kwargs):
        options=dict(history_frames=10,future_frames=5,n_rf=8,snr_db=20,
                     rng=np.random.default_rng(7))
        options.update(kwargs)
        return make_observation(self.channel if channel is None else channel,**options)

    def test_future_perturbation_cannot_change_combiner_or_noisy_history(self):
        changed=self.channel.copy();changed[10:]=changed[10:]*100+42j
        original,modified=self.observe(),self.observe(changed)
        for key in ['D','He_history','He_history_clean']:
            np.testing.assert_array_equal(original[key],modified[key])
        self.assertFalse(np.array_equal(original['He_future'],modified['He_future']))

    def test_same_seed_reproduces_sample(self):
        for key,value in self.observe().items():
            np.testing.assert_array_equal(value,self.observe()[key])

    def test_shapes_and_finite_float32(self):
        expected={'D':(8,16,2),'He_history':(10,8,4,2),
                  'He_history_clean':(10,8,4,2),'He_future':(5,8,4,2)}
        for key,value in self.observe().items():
            self.assertEqual(value.shape,expected[key])
            self.assertEqual(value.dtype,np.float32)
            self.assertTrue(np.isfinite(value).all())

    def test_noise_changes_history_but_not_target(self):
        low,high=self.observe(snr_db=0),self.observe(snr_db=30)
        np.testing.assert_array_equal(low['D'],high['D'])
        np.testing.assert_array_equal(low['He_future'],high['He_future'])
        self.assertGreater(np.square(low['He_history']-low['He_history_clean']).sum(),
                           np.square(high['He_history']-high['He_history_clean']).sum())

    def test_target_is_clean_channel_under_fixed_combiner(self):
        s=self.observe();d=s['D'][...,0]+1j*s['D'][...,1]
        expected=np.einsum('rb,tbm->trm',d,self.channel[10:])
        actual=s['He_future'][...,0]+1j*s['He_future'][...,1]
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)

    def test_zero_channel_is_finite(self):
        for value in self.observe(np.zeros_like(self.channel)).values():
            self.assertTrue(np.isfinite(value).all())

    def test_invalid_inputs_are_rejected(self):
        cases=[dict(history_frames=0),dict(future_frames=0),dict(n_rf=17),
               dict(n_rf=0),dict(snr_db=float('nan')),dict(history_frames=9)]
        for options in cases:
            with self.subTest(options=options),self.assertRaises(ValueError):self.observe(**options)
        with self.assertRaises(ValueError):self.observe(self.channel.real)
        changed=self.channel.copy();changed[0,0,0]=np.nan
        with self.assertRaises(ValueError):self.observe(changed)


if __name__ == '__main__':
    unittest.main()
