# Đồ án tốt nghiệp - Channel Prediction

## Tên đề tài
Dự đoán kênh truyền trong môi trường di động tốc độ cao
sử dụng mạng học sâu kết hợp CNN, LSTM và Attention Mechanism

## Model đề xuất
CNN + LSTM + Attention (hybrid)
- CNN: trích xuất đặc trưng cục bộ từ chuỗi kênh
- LSTM: học temporal dependency dài hạn  
- Attention: đánh trọng số thời điểm quan trọng

## Baselines để so sánh
RNN, LSTM, GRU, Transformer (vanilla)

## Metric đánh giá
NMSE (Normalized Mean Square Error) vs SNR

## Stack kỹ thuật
- Python + PyTorch
- Kênh: Rayleigh fading, mô hình Jakes
- Tốc độ mô phỏng: 30, 120, 300 km/h
- seq_len = 50, N = 10000 samples

## Cấu trúc thư mục
channel-prediction/
├── data/
│   ├── generate_channel.py   ✅ DONE
│   └── channel_v*.npz        (generated)
├── models/
│   ├── baselines.py          (RNN, LSTM, GRU, Transformer)
│   └── proposed.py           (CNN-LSTM-Attention)
├── train.py
├── evaluate.py
└── results/

## Tiến độ
- [x] generate_channel.py: hoàn thành
- [ ] models/baselines.py
- [ ] models/proposed.py
- [ ] train.py
- [ ] evaluate.py