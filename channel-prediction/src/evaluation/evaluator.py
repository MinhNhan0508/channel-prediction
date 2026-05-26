"""End-to-end evaluation of one trained model on a test set."""
from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_nmse_per_frame, compute_sum_rate, nmse_to_db


class Evaluator:
    def __init__(self, model, device, normalizer, flatten_input: bool = False):
        self.model = model.to(device).eval()
        self.device = device
        self.normalizer = normalizer
        self.flatten_input = flatten_input

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, snr_db_list: List[float]) -> Dict:
        all_preds = []
        all_trues = []

        for batch in tqdm(loader, desc="evaluating"):
            x = batch["history"].to(self.device)
            y = batch["future"].to(self.device)

            if self.flatten_input and x.dim() == 5:
                B = x.shape[0]
                x_in = x.view(B, x.shape[1], -1)
            else:
                x_in = x

            pred = self.model(x_in)
            if self.flatten_input:
                B = y.shape[0]
                pred = pred.view(B, y.shape[1], *y.shape[2:])
            all_preds.append(pred.cpu())
            all_trues.append(y.cpu())

        preds = torch.cat(all_preds, dim=0)
        trues = torch.cat(all_trues, dim=0)

        # NMSE
        nmse_per_frame = compute_nmse_per_frame(preds, trues)
        nmse_db_per_frame = nmse_to_db(nmse_per_frame).tolist()
        nmse_avg_db = float(nmse_to_db(nmse_per_frame.mean()).item())

        # Denormalize for sum-rate
        preds_denorm = preds * self.normalizer.scale
        trues_denorm = trues * self.normalizer.scale

        sum_rates = {}
        for snr in snr_db_list:
            sr = compute_sum_rate(preds_denorm, trues_denorm, snr_db=snr)
            sum_rates[str(snr)] = {
                "predicted": float(sr["sum_rate_predicted"].item()),
                "ideal": float(sr["sum_rate_ideal"].item()),
            }

        return {
            "nmse_per_frame_db": nmse_db_per_frame,
            "nmse_avg_db": nmse_avg_db,
            "sum_rates": sum_rates,
        }
