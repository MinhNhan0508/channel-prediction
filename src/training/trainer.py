"""Generic Trainer for channel-prediction models.

Supports:
  - AdamW + warmup/cosine schedule
  - Mixed precision (FP16) via torch.amp (autocast + GradScaler) -- important for
    the GTX 1650 4GB
  - Gradient clipping
  - Early stopping on validation NMSE
  - Checkpoint saving (best + last)
  - TensorBoard logging
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.amp import GradScaler, autocast  # PyTorch >= 2.x
    def _autocast(device_type: str, enabled: bool):
        return autocast(device_type=device_type, enabled=enabled)
    def _grad_scaler(enabled: bool, device_type: str):
        return GradScaler(device=device_type, enabled=enabled)
except Exception:  # legacy fallback
    from torch.cuda.amp import GradScaler, autocast  # type: ignore
    def _autocast(device_type: str, enabled: bool):
        return autocast(enabled=enabled)
    def _grad_scaler(enabled: bool, device_type: str):
        return GradScaler(enabled=enabled)

from tqdm import tqdm

from src.training.optimizer import build_optimizer_and_scheduler
from src.utils.logger import Logger
from src.training.checkpoint import atomic_save, load_for_resume, random_state, restore_random_state


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        config: dict,
        log_dir: Path,
        ckpt_dir: Path,
        flatten_input: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.log_dir = Path(log_dir)
        self.ckpt_dir = Path(ckpt_dir)
        self.flatten_input = flatten_input

        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            model, config, steps_per_epoch=len(train_loader)
        )

        use_amp_cfg = bool(config.get("use_amp", False))
        self.amp_enabled = use_amp_cfg and device.type == "cuda"
        self.device_type = device.type
        self.scaler = _grad_scaler(self.amp_enabled, device.type)

        self.logger = Logger(log_dir)
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.start_epoch = 0
        self.training_complete = False
        self.stopped = False

    def load_checkpoint(self, ckpt_path: Path) -> None:
        """Resume at an epoch boundary with matching data, model and recipe.

        CPU reproducibility is tested. GPU bitwise equivalence is not promised
        across driver, library or hardware changes.
        """
        ck = load_for_resume(ckpt_path, self.config)
        self.model.load_state_dict(ck["model_state_dict"])
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        self.scheduler.load_state_dict(ck["scheduler_state_dict"])
        if self.amp_enabled:
            self.scaler.load_state_dict(ck["scaler_state_dict"])
        self.best_val_loss = float(ck["best_val_loss"])
        self.patience_counter = int(ck["patience_counter"])
        self.start_epoch = int(ck["epoch"]) + 1
        self.training_complete = bool(ck["training_complete"])
        restore_random_state(ck["random_state"])
        print(f"[trainer] resumed after epoch {self.start_epoch}; complete={self.training_complete}", flush=True)

    def train(self) -> float:
        n_epochs = int(self.config["n_epochs"])
        patience = int(self.config.get("patience", 15))
        try:
            if self.training_complete:
                return self.best_val_loss
            for epoch in range(self.start_epoch, n_epochs):
                if hasattr(self.loss_fn, "set_epoch"):
                    self.loss_fn.set_epoch(epoch)
                train_metrics = self._train_epoch(epoch)
                val_metrics = self._validate(epoch)
                self.logger.log_metrics({**train_metrics, **val_metrics}, step=epoch)
                val_loss = val_metrics["val/nmse_loss"]
                improved = val_loss < self.best_val_loss
                if improved:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                self.training_complete = (epoch + 1 >= n_epochs or self.patience_counter >= patience)
                if improved:
                    self._save_checkpoint(epoch, "best")
                self._save_checkpoint(epoch, "last")
                stop_file = self.config.get("stop_file")
                if stop_file and Path(stop_file).exists() and not self.training_complete:
                    self.stopped = True
                    print("[trainer] stop requested; epoch checkpoint saved.", flush=True)
                    break
                if self.training_complete:
                    print(f"[trainer] finished after epoch {epoch + 1}", flush=True)
                    break
            return self.best_val_loss
        finally:
            self.logger.close()

    def _prepare_batch(self, batch: Dict) -> tuple:
        x = batch["history"].to(self.device)
        y = batch["future"].to(self.device)
        if self.flatten_input and x.dim() == 5:
            B = x.shape[0]
            x_in = x.view(B, x.shape[1], -1)
        else:
            x_in = x
        return x_in, y

    def _restore_pred_shape(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.flatten_input and y.dim() == 5:
            B = y.shape[0]
            return pred.view(B, y.shape[1], *y.shape[2:])
        return pred

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        sums: Dict[str, float] = {"total": 0.0, "nmse": 0.0, "phase": 0.0, "temporal": 0.0}
        n_samples = 0

        pbar = tqdm(self.train_loader, desc=f"epoch {epoch} [train]", disable=self.config.get('quiet_progress', False))
        for batch in pbar:
            x_in, y = self._prepare_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with _autocast(self.device_type, enabled=self.amp_enabled):
                pred = self.model(x_in)
                pred_for_loss = self._restore_pred_shape(pred, y)
                loss_dict = self.loss_fn(pred_for_loss, y)
            if not torch.isfinite(loss_dict["total"]):
                raise FloatingPointError("Nonfinite loss; experiment stopped without skipping this run.")

            if self.amp_enabled:
                self.scaler.scale(loss_dict["total"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                self.optimizer.step()

            self.scheduler.step()

            for k in sums:
                v = loss_dict[k]
                sums[k] += float(v.item() if isinstance(v, torch.Tensor) else v) * y.shape[0]
            n_samples += y.shape[0]
            pbar.set_postfix({"loss": f"{loss_dict['total'].item():.4f}"})

        return {f"train/{k}_loss": (sums[k] / max(1, n_samples)) for k in sums}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        sums: Dict[str, float] = {"total": 0.0, "nmse": 0.0, "phase": 0.0, "temporal": 0.0}
        n_samples = 0

        for batch in tqdm(self.val_loader, desc=f"epoch {epoch} [val]", disable=self.config.get('quiet_progress', False)):
            x_in, y = self._prepare_batch(batch)
            with _autocast(self.device_type, enabled=self.amp_enabled):
                pred = self.model(x_in)
                pred_for_loss = self._restore_pred_shape(pred, y)
                loss_dict = self.loss_fn(pred_for_loss, y)
            if not torch.isfinite(loss_dict["total"]):
                raise FloatingPointError("Nonfinite loss; experiment stopped without skipping this run.")
            for k in sums:
                v = loss_dict[k]
                sums[k] += float(v.item() if isinstance(v, torch.Tensor) else v) * y.shape[0]
            n_samples += y.shape[0]

        return {f"val/{k}_loss": (sums[k] / max(1, n_samples)) for k in sums}

    def _save_checkpoint(self, epoch: int, tag: str) -> None:
        atomic_save({
            "schema_version": 2,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.amp_enabled else None,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "config": self.config,
            "training_complete": self.training_complete,
            "random_state": random_state(),
        }, self.ckpt_dir / f"checkpoint_{tag}.pt")
