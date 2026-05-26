"""Max-abs normalization for complex channels stored as (..., 2)."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


class ChannelNormalizer:
    """Single-scalar normalizer: divides by the max |value| of the training set."""

    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)

    @classmethod
    def fit(cls, train_data: np.ndarray) -> "ChannelNormalizer":
        scale = float(np.max(np.abs(train_data)))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        return cls(scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x / self.scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale

    def save(self, path: Union[str, Path]) -> None:
        np.savez(path, scale=self.scale)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ChannelNormalizer":
        data = np.load(path)
        return cls(scale=float(data["scale"]))


if __name__ == "__main__":
    x = np.random.randn(10, 5).astype(np.float32) * 7.3
    norm = ChannelNormalizer.fit(x)
    y = norm.transform(x)
    z = norm.inverse_transform(y)
    assert np.allclose(x, z, atol=1e-5)
    print(f"normalize OK | scale={norm.scale:.4f}")
