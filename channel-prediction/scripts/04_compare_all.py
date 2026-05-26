"""Compare all 5 models on both test scenarios. Make plots + summary table.

Usage:
    python scripts/04_compare_all.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


MODELS = ["st_mamba", "rnn", "lstm", "gru", "transformer"]
MODEL_DISPLAY = {
    "st_mamba": "ST-Mamba (Proposed)",
    "rnn": "RNN",
    "lstm": "LSTM",
    "gru": "GRU",
    "transformer": "Transformer",
}
COLORS = {
    "st_mamba": "red", "rnn": "blue", "lstm": "green",
    "gru": "orange", "transformer": "purple",
}
SCENARIOS = ["v30", "v50"]


def load_results():
    out = {}
    for m in MODELS:
        out[m] = {}
        for sc in SCENARIOS:
            f = Path(f"results/tables/{m}_full_seed42/results_{sc}.json")
            if f.exists():
                with open(f, "r", encoding="utf-8") as fp:
                    out[m][sc] = json.load(fp)
    return out


def plot_nmse_per_frame(results, scenario, save_path: Path):
    plt.figure(figsize=(8, 5))
    for m in MODELS:
        if scenario in results.get(m, {}):
            nmse = results[m][scenario]["nmse_per_frame_db"]
            frames = list(range(1, len(nmse) + 1))
            plt.plot(
                frames, nmse, marker="o", label=MODEL_DISPLAY[m],
                color=COLORS[m], linewidth=2 if m == "st_mamba" else 1,
            )
    plt.xlabel("Predicted Frame Index (x 0.625 ms)")
    plt.ylabel("NMSE (dB)")
    plt.title(f"NMSE vs Predicted Frame -- v = {scenario[1:]} km/h, SNR = 20 dB")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sum_rate(results, scenario, save_path: Path):
    # Find an existing model with this scenario for the SNR list
    snrs = None
    for m in MODELS:
        if scenario in results.get(m, {}):
            snrs = sorted(float(s) for s in results[m][scenario]["sum_rates"].keys())
            break
    if snrs is None:
        return

    plt.figure(figsize=(8, 5))

    # Ideal as reference (any model's "ideal" should be the same up to noise)
    for m in MODELS:
        if scenario in results.get(m, {}):
            ideal = [results[m][scenario]["sum_rates"][str(s)]["ideal"] for s in snrs]
            plt.plot(snrs, ideal, "k--", label="Ideal CSI", linewidth=2)
            break

    for m in MODELS:
        if scenario in results.get(m, {}):
            sr = [results[m][scenario]["sum_rates"][str(s)]["predicted"] for s in snrs]
            plt.plot(
                snrs, sr, marker="o", label=MODEL_DISPLAY[m],
                color=COLORS[m], linewidth=2 if m == "st_mamba" else 1,
            )
    plt.xlabel("SNR (dB)")
    plt.ylabel("Sum-rate (bps/Hz)")
    plt.title(f"Sum-rate vs SNR -- v = {scenario[1:]} km/h")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    results = load_results()
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    for sc in SCENARIOS:
        plot_nmse_per_frame(results, sc, out_dir / f"nmse_per_frame_{sc}.png")
        plot_sum_rate(results, sc, out_dir / f"sum_rate_{sc}.png")

    # Summary table
    print()
    print("=== NMSE Average (dB) ===")
    print(f"{'Model':25s} | {'v=30 km/h':>10s} | {'v=50 km/h':>10s}")
    print("-" * 55)
    for m in MODELS:
        v30 = results.get(m, {}).get("v30", {}).get("nmse_avg_db", float("nan"))
        v50 = results.get(m, {}).get("v50", {}).get("nmse_avg_db", float("nan"))
        print(f"{MODEL_DISPLAY[m]:25s} | {v30:>10.2f} | {v50:>10.2f}")


if __name__ == "__main__":
    main()
