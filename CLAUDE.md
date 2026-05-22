# CLAUDE.md — Channel Prediction Project

## Project Overview
**Thesis:** Channel prediction in high-mobility wireless environments using deep learning (CNN + LSTM + Attention).

**Goal:** Predict the Rayleigh fading channel coefficient h(t+1) given the past 50 time steps, evaluated by NMSE vs SNR across 3 mobility scenarios.

---

## Environment

- **Python:** `C:/Users/lequo/anaconda3/envs/channel_prediction/python.exe` (PyTorch 2.6, CUDA 12.4)
- **GPU:** NVIDIA GeForce GTX 1650, CUDA 12.4
- **Run scripts from:** `channel-prediction/` subfolder (vì code dùng path tương đối `data/`, `results/`)

---

## Project Structure

```
FinalProject/
├── channel-prediction/
│   ├── data/
│   │   └── generate_channel.py     ✅ DONE
│   ├── models/
│   │   ├── baselines.py            ✅ DONE (RNN, LSTM, GRU, Transformer)
│   │   └── proposed.py             ✅ DONE (CNN-LSTM-Attention v2)
│   ├── train.py                    ✅ DONE
│   ├── evaluate.py                 ✅ DONE
│   └── CONTEXT.md                  (outdated — refer to this file instead)
├── data/                           (generated .npz files)
├── results/
│   ├── checkpoints/                (model .pth files)
│   ├── nmse_vs_snr.png
│   ├── attention_weights.png
│   └── training_history.npy
└── CLAUDE.md                       (this file)
```

---

## Dataset

| Parameter | Value |
|-----------|-------|
| Channel model | Rayleigh fading (Jakes model) |
| Carrier frequency | 2.4 GHz |
| Sampling frequency | 1000 Hz |
| Total samples (N) | 10,000 |
| Sequence length | 50 |
| Train / Test split | 80% / 20% (7960 / 1990) |
| Velocities | 30 km/h, 120 km/h, 300 km/h |
| Input shape | (batch, 50, 2) — real + imaginary |
| Output shape | (batch, 2) |

---

## Models

### Baselines (`models/baselines.py`)
- RNN, LSTM, GRU — hidden_size=64, num_layers=2, dropout=0.1
- Transformer — d_model=64, nhead=4, num_layers=2, dropout=0.1

### Proposed Model (`models/proposed.py`) — CNNLSTMAttention v2
Architecture: **CNN (Residual) → LSTM → Multi-Head Attention → FC**

| Component | Detail |
|-----------|--------|
| CNN | 2-layer Conv1d, cnn_channels=32, kernel=3, BatchNorm, Residual |
| LSTM | hidden_size=64, num_layers=2, dropout=0.1 |
| Attention | Multi-Head (num_heads=2), LayerNorm |
| FC | hidden*2 → hidden → 2, GELU activation |
| Dropout | 0.1 throughout |

---

## Training Config (current)

```python
CONFIG = {
    'epochs'     : 150,
    'batch_size' : 64,
    'lr'         : 5e-4,
    'patience'   : 20,      # early stopping
    'hidden_size': 64,
    'num_layers' : 2,
    'data_dir'   : 'data',
    'save_dir'   : 'results/checkpoints',
    'velocities' : [30, 120, 300],
}
```

Optimizer: Adam | Loss: MSELoss | LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=5) | Grad clip: 1.0

---

## Evaluation Metric
**NMSE (dB)** = 10 × log10(||ŷ - y||² / ||y||²) — lower is better (more negative = better)

Evaluated across SNR range: 0, 5, 10, 15, 20, 25, 30 dB

---

## Training Results Summary

### Training 1 (epochs=50, lr=1e-3, patience=10)
| Model | 30 km/h | 120 km/h | 300 km/h |
|-------|---------|----------|----------|
| RNN | -6.66 | +2.35 | -1.96 |
| LSTM | -9.93 | -2.99 | +4.70 |
| GRU | -8.78 | +5.87 | +6.72 |
| Transformer | **-17.73** | **-15.37** | **-13.09** |
| CNN-LSTM-Attention ★ | -14.92 | -11.65 | -6.00 |

### Training 2 (epochs=150, lr=5e-4, patience=20) — current checkpoints
| Model | 30 km/h | 120 km/h | 300 km/h |
|-------|---------|----------|----------|
| Transformer | **-17.76** | **-15.18** | **-12.84** |
| CNN-LSTM-Attention ★ | -15.44 | -12.24 | -6.64 |

> Transformer still outperforms the proposed model. Need to improve CNN-LSTM-Attention at 300 km/h.

---

## Git Tags
| Tag | Description |
|-----|-------------|
| `v1-training-baseline` | Training 1 — all 5 models, epochs=50 |
| `v2-training-lr5e4-ep150` | Training 2 — CNN-LSTM-Attention only, epochs=150 |

---

## How to Run

```bash
# 1. Generate data
C:/Anaconda/envs/tf_env/python.exe channel-prediction/data/generate_channel.py

# 2. Train (only CNN-LSTM-Attention by default)
C:/Anaconda/envs/tf_env/python.exe channel-prediction/train.py

# 3. Evaluate all models
PYTHONIOENCODING=utf-8 C:/Anaconda/envs/tf_env/python.exe channel-prediction/evaluate.py
```

---

## Rules for Claude
- Do NOT run `git push` or `git commit` unless user explicitly says "làm giúp tôi" or equivalent
- Do NOT modify function logic unless explicitly asked
- Do NOT add unnecessary abstractions or new files
- Always use `C:/Anaconda/envs/tf_env/python.exe` when running Python
- Run scripts from `FinalProject/` root, not from inside `channel-prediction/`
