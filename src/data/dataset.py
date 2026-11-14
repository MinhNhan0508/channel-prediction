"""PyTorch Dataset for channel-prediction samples stored as .npz files."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.normalize import ChannelNormalizer


class ChannelDataset(Dataset):
    """Lazy-loading dataset.

    Each sample is one .npz file written by scripts/01b_generate_sionna.py.
    For baselines that flatten (B, P, N_RF, N_M, 2) -> (B, P, N_RF*N_M*2), set
    `flatten_spatial=True`. For spatial models (pi_dast, cnn) leave it False.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        normalizer: Optional[ChannelNormalizer] = None,
        flatten_spatial: bool = False,
        cache_in_memory: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("sample_*.npz"))
        if len(self.files) == 0:
            raise RuntimeError(f"No samples found in {self.data_dir}")
        self.normalizer = normalizer
        self.flatten_spatial = flatten_spatial
        self._cache = [self._read(i) for i in range(len(self.files))] if cache_in_memory else None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._read(idx) if self._cache is None else self._cache[idx]

    def _read(self, idx: int) -> Dict[str, torch.Tensor]:
        with np.load(self.files[idx], allow_pickle=False) as data:
            He_history = data["He_history"].astype(np.float32)
            He_future = data["He_future"].astype(np.float32)
            metadata = {key: float(data[key]) for key in ['velocity', 'snr', 'delay_spread']}

        if self.normalizer is not None:
            He_history = self.normalizer.transform(He_history)
            He_future = self.normalizer.transform(He_future)

        if self.flatten_spatial:
            P = He_history.shape[0]
            Q = He_future.shape[0]
            He_history = He_history.reshape(P, -1)
            He_future = He_future.reshape(Q, -1)

        return {
            "history": torch.from_numpy(He_history),
            "future": torch.from_numpy(He_future),
            "metadata": metadata,
        }


if __name__ == "__main__":
    print("dataset module loaded OK")
