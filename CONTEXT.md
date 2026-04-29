# Đồ án tốt nghiệp - Channel Prediction

## Tên đề tài
Dự đoán kênh truyền trong môi trường di động tốc độ cao
sử dụng mạng học sâu kết hợp CNN, LSTM và Attention Mechanism

---

## 1. Phạm vi và giả định (Assumptions)

| Khía cạnh | Lựa chọn | Lý do |
|---|---|---|
| Cấu hình anten | **SISO** (1 phát, 1 thu) | Đề tài tập trung vào temporal prediction (Doppler, channel aging). MIMO thêm chiều spatial — là đề tài khác lớn hơn. Kiến trúc đề xuất có thể mở rộng sang MIMO bằng cách áp dụng cho từng sub-channel (future work). |
| Mô hình kênh | **Rayleigh fading** (NLOS) | Worst-case scenario — nếu model hoạt động tốt trên Rayleigh thì trên Rician (LOS) sẽ tốt hơn. Đây là chọn lựa chuẩn của Liu et al. cho urban OFDM. |
| Phổ Doppler | **Jakes (isotropic 2D)** | Mô hình kinh điển, được dùng rộng rãi. Các mô hình thực tế hơn (3GPP TDL, Aulin) là hướng phát triển. |
| Băng thông | **Flat-fading single-tap** | Tập trung vào time-domain prediction. Frequency-selective fading có thể xử lý độc lập từng subcarrier bằng cùng kiến trúc. |
| Pilot estimation | **Idealized** (giả định channel tại pilot là perfect) | Tập trung vào bài toán prediction từ chuỗi ước lượng. Kết hợp với ước lượng có nhiễu tại pilot là future work. |
| Tần số sóng mang | **fc = 2.4 GHz** | Wi-Fi 6E lower / 5G n7 — band chính cho high-mobility. mmWave (28 GHz) chưa triển khai cho tàu cao tốc do path-loss cao. |
| Tần số lấy mẫu | **fs = 10 kHz** (T_symbol = 0.1ms) | Gần với LTE (T_symbol ~ 70μs ↔ fs ~ 14kHz). Đảm bảo normalized Doppler fdT nằm trong dải nghiên cứu (0.007 - 0.07). |

### Bảng thông số scenarios

| Scenario | v (km/h) | fd (Hz) | fdT = fd/fs | Đánh giá |
|---|---|---|---|---|
| Đô thị / pedestrian | 30 | 66.7 | 0.0067 | Coherent — Wiener đủ tốt |
| Cao tốc / vehicular | 120 | 266.7 | 0.0267 | Challenging — AI bắt đầu vượt Wiener |
| Tàu cao tốc / UAV | 300 | 666.7 | 0.0667 | High-mobility — AI vượt rõ rệt |

---

## 2. Model đề xuất

**CNN + LSTM + Multi-Head Attention** (hybrid):
- **CNN**: trích xuất đặc trưng cục bộ, lọc nhiễu cấp local
- **LSTM**: học temporal dependency dài hạn
- **Multi-Head Attention**: đánh trọng số thời điểm quan trọng + cung cấp giải thích (XAI)

Tổng số tham số: ~89k (gọn hơn Transformer ~100k).

---

## 3. So sánh

### Baseline cổ điển
- **Wiener filter** (linear MMSE predictor) — closed-form, không cần training

### Baseline AI
- RNN, LSTM, GRU, Transformer (vanilla)

### Ablation Study (chứng minh vai trò từng thành phần)
1. **LSTM** thuần
2. **CNN-LSTM** (bỏ Attention)
3. **LSTM-Attention** (bỏ CNN)
4. **CNN-LSTM-Attention** * (đầy đủ — đề xuất)

---

## 4. Metrics đánh giá

| Metric | Mục đích |
|---|---|
| **NMSE vs SNR** | Chính — đo độ chính xác prediction |
| **NMSE vs Horizon** | Multi-step prediction → chứng minh channel aging |
| **BER** (QPSK + ZF eq) | Hiệu năng end-to-end so với Perfect CSI |
| **Train/Val Loss** | Chứng minh hội tụ, không overfit |
| **Params + FLOPs** | Complexity cho edge inference |
| **CPU/GPU latency** | Tính thực tế cho deployment |
| **Attention weights** | XAI — giải thích model nhìn vào đâu |

### Mean ± Std
Tất cả NMSE được tính qua **3 random seeds** (42, 123, 456) để báo cáo có thống kê.

---

## 5. Cấu trúc thư mục

```
channel-prediction/
├── data/
│   ├── generate_channel.py    Sinh data Rayleigh-Jakes (3 seed × 3 velocity)
│   └── channel_v*.npz         (generated, có buffer chống leak)
├── models/
│   ├── baselines.py           RNN, LSTM, GRU, Transformer
│   ├── classical.py           Wiener filter (closed-form)
│   └── proposed.py            CNN-LSTM-Attn + 2 ablation
├── train.py                   Multi-seed (skip-existing supported)
├── evaluate.py                NMSE/BER/Ablation/Multi-step/Complexity
├── app.py                     Gradio demo
└── results/
    ├── checkpoints/           (63 file = 7 model × 3 v × 3 seed)
    └── *.png                  (~10 charts)
```

---

## 6. Cài đặt train/test

- **N = 50,000 samples** mỗi (velocity, seed) = 5 giây thực tế
- **Train/Test split = 80/20** + **buffer = SEQ_LEN (50)** giữa train/test
  - Buffer loại bỏ overlap do sliding window → không có leak
- **seq_len = 50** symbols = 5ms thực tế
- **batch_size = 64**, **lr = 5e-4**, **epochs = 150** (early stop patience=20)
- **Data augmentation**: thêm AWGN ngẫu nhiên trong khi training (SNR ~ U(0, 30) dB)
  - Giúp model robust với nhiễu thực tế

---

## 7. Tiến độ

- [x] `generate_channel.py` (fs=10kHz, có buffer)
- [x] `models/baselines.py` (RNN, LSTM, GRU, Transformer)
- [x] `models/classical.py` (Wiener filter)
- [x] `models/proposed.py` (CNN-LSTM-Attn + 2 ablation: CNN-LSTM, LSTM-Attn)
- [x] `train.py` (7 model × 3 v × 3 seed, skip-existing)
- [x] `evaluate.py` (NMSE/BER/Ablation/Multi-step/Complexity)
- [ ] **Regenerate data + retrain toàn bộ 63 model** ← TODO

---

## 8. Câu hỏi hội đồng — đáp án nhanh

| Câu hỏi | Đáp án |
|---|---|
| Vì sao SISO mà không MIMO? | Đề tài tập trung temporal prediction; MIMO là extension future work. |
| Vì sao Rayleigh mà không Rician? | Rayleigh là worst-case NLOS; nếu work trên Rayleigh thì work trên Rician. |
| Vì sao không Kalman filter? | Wiener đại diện linear MMSE optimal; Kalman cần state-space model mà ở high-mobility không stationary nên cũng degrade. |
| BER tại sao chỉ QPSK? | QPSK là điều chế chuẩn, đơn giản; mở rộng sang 16-QAM là trivial. |
| Window 50 sao không 100? | seq_len=50 = 5ms = đủ bắt 1-2 chu kỳ Doppler ở fd=666Hz. Window dài hơn không thêm thông tin do channel decorrelation. |
| Sao chỉ 3 seed? | Cân bằng giữa độ tin cậy thống kê và compute budget; std qua 3 seed đã đủ ổn định (<1 dB). |
| Train/test có leak không? | Không — có buffer 50 symbol giữa train và test loại bỏ overlap do sliding window. |
| Sao không dùng Real Channel Data? | Dataset thực tế (như 3GPP CDL) cần subscription. Jakes là chuẩn academic phổ biến. |
