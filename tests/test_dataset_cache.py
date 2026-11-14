from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from src.data.dataset import ChannelDataset
from src.data.normalize import ChannelNormalizer


class DatasetCacheTest(unittest.TestCase):
    def test_cached_and_lazy_reads_match(self):
        with tempfile.TemporaryDirectory(prefix='mimo-cache-') as tmp:
            path=Path(tmp)
            np.savez(path/'sample_00000.npz',He_history=np.ones((10,8,4,2),dtype=np.float32),
                     He_future=np.full((5,8,4,2),2,dtype=np.float32),velocity=30.,snr=20.,delay_spread=50.)
            lazy=ChannelDataset(path,ChannelNormalizer(2),flatten_spatial=True)
            cached=ChannelDataset(path,ChannelNormalizer(2),flatten_spatial=True,cache_in_memory=True)
            for key in ['history','future']:self.assertTrue(torch.equal(lazy[0][key],cached[0][key]))
            self.assertEqual(lazy[0]['metadata'],cached[0]['metadata'])
