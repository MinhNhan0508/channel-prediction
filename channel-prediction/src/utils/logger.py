"""TensorBoard + console logger."""
from pathlib import Path
from typing import Mapping

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: Path):
        self.writer = SummaryWriter(str(log_dir))

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        for k, v in metrics.items():
            self.writer.add_scalar(k, float(v), step)
        line = f"Step {step}: " + " | ".join(f"{k}={float(v):.4f}" for k, v in metrics.items())
        print(line, flush=True)

    def close(self) -> None:
        self.writer.close()
