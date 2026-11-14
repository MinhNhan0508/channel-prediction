"""TensorBoard + console logger."""
from pathlib import Path
import json
import time
from typing import Mapping

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: Path):
        self.writer = SummaryWriter(str(log_dir))
        self.metrics_path = Path(log_dir) / 'metrics.jsonl'
        self.started = time.perf_counter()

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        record = {'epoch': step + 1, 'elapsed_s': time.perf_counter() - self.started,
                  **{k: float(v) for k, v in metrics.items()}}
        with self.metrics_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        for k, v in metrics.items():
            self.writer.add_scalar(k, float(v), step)
        line = f"Step {step}: " + " | ".join(f"{k}={float(v):.4f}" for k, v in metrics.items())
        print(line, flush=True)

    def close(self) -> None:
        self.writer.close()
