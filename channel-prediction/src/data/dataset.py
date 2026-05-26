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

    Each sample is one .npz file written by scripts/01_generate_data.py.
    For baselines that flatten (B, P, N_RF, N_M, 2) -> (B, P, N_RF*N_M*2), set
    `flatten_spatial=True`. For ST-Mamba leave it False to preserve spatial dims.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        normalizer: Optional[ChannelNormalizer] = None,
        flatten_spatial: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("sample_*.npz"))
        if len(self.files) == 0:
            raise RuntimeError(f"No samples found in {self.data_dir}")
        self.normalizer = normalizer
        self.flatten_spatial = flatten_spatial

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.files[idx])
        He_history = data["He_history"].astype(np.float32)
        He_future = data["He_future"].astype(np.float32)

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
            "metadata": {
                "velocity": float(data["velocity"]),
                "snr": float(data["snr"]),
                "delay_spread": float(data["delay_spread"]),
            },
        }


if __name__ == "__main__":
    print("dataset module loaded OK")
