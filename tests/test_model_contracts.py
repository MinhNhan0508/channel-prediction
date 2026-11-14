"""Shape, parameter and gradient checks for maintained configurations."""
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.utils import build_model, is_spatial_model, count_trainable_params
from src.utils.config import load_config


class ModelsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_registered_models_match_configured_dimensions(self):
        for name in ['rnn','lstm','gru','cnn','transformer','pi_dast']:
            with self.subTest(model=name):
                cfg = load_config(ROOT/f'configs/model/{name}.yaml')
                model = build_model(name,cfg).eval()
                p,q = cfg['p_frames'],cfg['q_frames']
                shape = (2,p,8,4,2) if is_spatial_model(name) else (2,p,64)
                expected = (2,q,8,4,2) if is_spatial_model(name) else (2,q,64)
                with torch.no_grad(): result = model(torch.randn(*shape))
                self.assertEqual(tuple(result.shape),expected)
                self.assertTrue(torch.isfinite(result).all())
                self.assertTrue(330_000 <= count_trainable_params(model) <= 410_000)

    def test_full_model_parameter_count(self):
        cfg = load_config(ROOT/'configs/model/pi_dast.yaml')
        self.assertEqual(count_trainable_params(build_model('pi_dast',cfg)),406_677)

    def test_full_model_backpropagates_through_neural_branch_and_gate(self):
        model = build_model('pi_dast',load_config(ROOT/'configs/model/pi_dast.yaml')).train()
        model(torch.randn(2,10,8,4,2)).square().mean().backward()
        for parameter in [model.encoder_spatial.weight,model.gate[-1].weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(),0)

    def test_no_encoder_ablation_keeps_prior_and_gate(self):
        full = load_config(ROOT/'configs/model/pi_dast.yaml')
        ablation = load_config(ROOT/'configs/model/pi_dast_no_doppler_encoder.yaml')
        model = build_model('pi_dast',ablation)
        self.assertFalse(model.use_doppler_enc)
        self.assertTrue(model.use_pad_prior)
        self.assertTrue(model.use_gate)
        self.assertFalse(hasattr(model,'doppler'))
        keys=['n_rf','n_m','p_frames','q_frames','d_model','n_heads',
              'num_encoder_layers','num_decoder_layers','dim_feedforward','dropout',
              'use_pad_prior','use_gate']
        self.assertEqual({k:full[k] for k in keys},{k:ablation[k] for k in keys})


if __name__ == '__main__':
    unittest.main()
