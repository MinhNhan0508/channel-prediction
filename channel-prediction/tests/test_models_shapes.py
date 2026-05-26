"""Shape + parameter-budget tests for all five models.

Run with:
    PYTHONPATH=. python tests/test_models_shapes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.utils import build_model, count_trainable_params  # noqa: E402
from src.utils.config import load_config  # noqa: E402


PARAM_BUDGET = (280_000, 420_000)


def _shape_check(name: str, model, x_shape, y_shape_expected):
    x = torch.randn(*x_shape)
    y = model(x)
    assert tuple(y.shape) == y_shape_expected, (
        f"{name}: expected {y_shape_expected}, got {tuple(y.shape)}"
    )
    return y


def test_all_models():
    failures = []
    for name in ["rnn", "lstm", "gru", "transformer", "st_mamba"]:
        cfg = load_config(f"configs/model/{name}.yaml")
        model = build_model(name, cfg)
        n = count_trainable_params(model)
        in_budget = PARAM_BUDGET[0] <= n <= PARAM_BUDGET[1]

        if name == "st_mamba":
            x_shape = (2, 30, 8, 4, 2)
            y_expected = (2, 5, 8, 4, 2)
        else:
            x_shape = (2, 30, 64)
            y_expected = (2, 5, 64)

        _shape_check(name, model, x_shape, y_expected)
        status = "OK" if in_budget else "OUT_OF_BUDGET"
        print(f"  [{status:13s}] {name:12s} params={n:>8,}")
        if not in_budget:
            failures.append((name, n))

    if failures:
        print(f"\nFAIL: {len(failures)} model(s) outside budget {PARAM_BUDGET}:")
        for n, p in failures:
            print(f"  - {n}: {p:,}")
        raise SystemExit(1)
    print("\nAll models PASS shape + budget tests.")


if __name__ == "__main__":
    test_all_models()
