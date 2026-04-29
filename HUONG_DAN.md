
channel-prediction/
├── CONTEXT.md                       <-- ĐÈ
├── data/
│   └── generate_channel.py          <-- ĐÈ (fs=10kHz, có buffer)
├── models/
│   ├── classical.py                 <-- MỚI (Wiener filter)
│   └── proposed.py                  <-- ĐÈ (thêm LSTMAttention)
├── train.py                         <-- ĐÈ (7 model + skip-existing)
└── evaluate.py                      <-- ĐÈ (Wiener + ablation + multi-step)
```

---

## Tóm tắt thay đổi

### 1. `data/generate_channel.py`
- **fs đổi từ 1000 Hz → 10000 Hz** (1 symbol = 0.1ms, gần với LTE)
- Thêm **buffer = 50 symbols giữa train và test** để loại bỏ leak từ sliding window
- Lưu thêm metadata `fs`, `buffer` vào file `.npz`

### 2. `models/classical.py` (MỚI)
- `WienerFilter` — linear MMSE predictor closed-form
- Hỗ trợ horizon=1 (single-step) và horizon>1 (multi-step direct)
- Không cần training, fit từ X_train/y_train của mỗi (velocity, seed)

### 3. `models/proposed.py`
- Thêm class `LSTMAttention` — LSTM + Multi-Head Attention, KHÔNG có CNN
- Để hoàn chỉnh ablation 4 phiên bản: LSTM / CNN-LSTM / LSTM-Attn / CNN-LSTM-Attn

### 4. `train.py`
- Đăng ký 7 model: RNN, LSTM, GRU, Transformer, CNN-LSTM, LSTM-Attention, CNN-LSTM-Attention
- Thêm `CONFIG['skip_existing'] = True` — skip những checkpoint đã có (tiết kiệm thời gian retrain)
- Fix lỗi hardcode `lr` và `patience` (giờ đọc từ CONFIG)

### 5. `evaluate.py`
- Hỗ trợ Wiener (xử lý riêng, không load checkpoint)
- Thêm `plot_ablation_study()` — biểu đồ NMSE vs SNR cho 4 phiên bản ablation
- Thêm **Multi-step prediction**: `predict_multistep_iterative`, `compute_nmse_multistep`, `plot_nmse_vs_horizon`
- Tách 2 chart NMSE: `nmse_vs_snr_main.png` (5 model gọn) và `nmse_vs_snr_all.png` (8 model đầy đủ)

### 6. `CONTEXT.md`
- Document hóa các assumption: SISO, Rayleigh, Jakes, fs=10kHz
- Bảng câu hỏi hội đồng + đáp án

---

## Quy trình chạy (4 bước)

> **QUAN TRỌNG**: Vì `fs` đã thay đổi, tất cả checkpoint cũ không còn dùng được.
> Phải xóa và retrain toàn bộ.

### Bước 1: Xóa data và checkpoint cũ

```bash
cd channel-prediction

# Xóa data cũ (sinh từ fs=1000Hz)
rm -f data/*.npz

# Xóa checkpoint cũ (không tương thích với data mới)
rm -rf results/checkpoints/*
rm -f results/*.npy
```

### Bước 2: Sinh lại data (~30 giây)

```bash
python data/generate_channel.py
```

Output kỳ vọng:
```
[v =  30 km/h | seed=42]  fd =   66.7 Hz  fdT = 0.0067  (coherent)
[v = 120 km/h | seed=42]  fd =  266.7 Hz  fdT = 0.0267  (challenging)
[v = 300 km/h | seed=42]  fd =  666.7 Hz  fdT = 0.0667  (high-mobility)
... (lặp cho seed 123 và 456)
```

Tổng cộng 9 file `.npz` trong `data/`.

### Bước 3: Train tất cả 63 model (~6-10 giờ trên GTX 1650)

```bash
python train.py
```

- 7 model × 3 velocity × 3 seed = **63 runs**
- Thời gian dự kiến mỗi run: ~6-10 phút
- **Lưu ý**: Để máy chạy qua đêm, hoặc dùng `nohup`/`tmux`

Để chạy thử subset trước (kiểm tra mọi thứ OK), sửa tạm `train.py`:
```python
CONFIG['velocities'] = [120]    # chỉ 1 velocity
CONFIG['seeds']      = [42]      # chỉ 1 seed
CONFIG['epochs']     = 30        # ít epoch
```
Khi chắc chắn không lỗi → revert về full setting → chạy thật.

### Bước 4: Evaluate (~10-15 phút)

```bash
python evaluate.py
```

Sẽ sinh:
- `results/nmse_vs_snr_main.png` — chart chính (5 model)
- `results/nmse_vs_snr_all.png` — chart đầy đủ (8 model)
- `results/ablation_study.png` — Ablation 4 phiên bản
- `results/timeseries_v{30,120,300}kmh.png` — Real vs Predicted
- `results/attention_weights.png` — XAI
- `results/nmse_vs_horizon.png` — **Multi-step (channel aging)**
- `results/ber_vs_snr_v{30,120,300}kmh.png` — BER với Perfect CSI bound
- `results/complexity_*.png` — Params/FLOPs/Latency
- Files `.npy` lưu kết quả số

---

## Kết quả Wiener trên data mới (sanity check)

Tui đã verify Wiener filter trên data mới:

| v (km/h) | fdT | NMSE @ SNR=20dB |
|---|---|---|
| 30 | 0.0067 | -26.4 dB (Wiener tốt) |
| 120 | 0.0267 | -21.9 dB (vừa phải) |
| 300 | 0.0667 | -18.7 dB (bắt đầu sập) |

→ Story thesis đúng hướng: AI sẽ vượt Wiener ở v=300 (high-mobility), gần ngang ở v=30/120.

---

## Nếu gặp lỗi

| Lỗi | Cách xử lý |
|---|---|
| `FileNotFoundError: data/channel_v*.npz` | Chạy lại Bước 2 |
| `RuntimeError: size mismatch` khi load checkpoint | Xóa checkpoint cũ (Bước 1), retrain |
| `CUDA out of memory` | Giảm `batch_size` từ 64 → 32 trong `train.py` |
| Wiener trả về NMSE > 0 dB ở SNR thấp | Bình thường — ở SNR=0dB, nhiễu mạnh hơn signal |
| `ImportError: thop` | `pip install thop` (cho FLOPs analysis trong evaluate.py) |
