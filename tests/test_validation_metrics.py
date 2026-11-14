"""Validation statistics must weight the final partial batch correctly."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.training.trainer import Trainer


class MeanTarget(torch.nn.Module):
    def forward(self,pred,true):
        value=true.mean()
        return {key:value for key in ['total','nmse','phase','temporal']}


class ValidationMetricsTest(unittest.TestCase):
    def test_partial_batch_is_weighted_by_sample_count(self):
        trainer=Trainer.__new__(Trainer)
        trainer.model=torch.nn.Identity()
        trainer.device=torch.device('cpu')
        trainer.device_type='cpu'
        trainer.amp_enabled=False
        trainer.config={}
        trainer.flatten_input=False
        trainer.loss_fn=MeanTarget()
        trainer.val_loader=[{'history':torch.zeros(2,1,1),'future':torch.ones(2,1,1)},
                            {'history':torch.zeros(1,1,1),'future':torch.full((1,1,1),10.)}]
        result=trainer._validate(0)
        for value in result.values():self.assertAlmostEqual(value,4.0)
