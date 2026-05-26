"""Curriculum learning: train on increasingly difficult velocity ranges.

Each stage filters the training set by velocity. To avoid touching every sample
just to read its metadata, this implementation reads metadata once at __init__
and caches the velocity per sample index.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from torch.utils.data import Subset

from src.data.dataset import ChannelDataset


class CurriculumScheduler:
    """Filter ChannelDataset by velocity for each stage.

    Args:
        full_dataset: a ChannelDataset
        stages: list of (end_epoch, (v_min, v_max)) tuples. The first stage with
                end_epoch >= current_epoch is used.
    """

    def __init__(
        self,
        full_dataset: ChannelDataset,
        stages: List[Tuple[int, Tuple[float, float]]],
    ):
        self.full_dataset = full_dataset
        self.stages = sorted(stages, key=lambda s: s[0])

        # Build a quick lookup of velocity per sample index by peeking only metadata
        velocities = np.empty(len(full_dataset), dtype=np.float32)
        for i, path in enumerate(full_dataset.files):
            data = np.load(path)
            velocities[i] = float(data["velocity"])
        self._velocities = velocities

        self.stage_indices = {}
        for end_epoch, (v_min, v_max) in self.stages:
            mask = (velocities >= v_min) & (velocities <= v_max)
            idx = np.nonzero(mask)[0].tolist()
            self.stage_indices[end_epoch] = idx
            print(
                f"[curriculum] stage until epoch {end_epoch}, "
                f"v=[{v_min},{v_max}]: {len(idx)} samples",
                flush=True,
            )

    def get_subset(self, epoch: int) -> Subset:
        for end_epoch, _ in self.stages:
            if epoch <= end_epoch:
                return Subset(self.full_dataset, self.stage_indices[end_epoch])
        return Subset(self.full_dataset, self.stage_indices[self.stages[-1][0]])
