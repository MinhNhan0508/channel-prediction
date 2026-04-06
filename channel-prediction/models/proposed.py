import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# KHỐI 1: CNN - trích xuất đặc trưng cục bộ
# ============================================================
class CNNBlock(nn.Module):
    """
    CNN 1 chiều (Conv1d) trích xuất pattern cục bộ từ chuỗi kênh.

    Tại sao cần CNN?
    - LSTM xử lý từng bước thời gian một, không giỏi nhận
      pattern ngắn như "kênh tăng đột biến 3 bước liên tiếp"
    - CNN dùng kernel trượt qua chuỗi, phát hiện pattern cục bộ
      rất tốt — giống như phát hiện cạnh trong ảnh

    Cấu trúc:
      input (batch, seq_len, 2)
        → Conv1d lớp 1 → BatchNorm → ReLU
        → Conv1d lớp 2 → BatchNorm → ReLU
      output (batch, seq_len, cnn_channels)

    Args:
        input_size   : số features đầu vào (2: real + imag)
        cnn_channels : số filter CNN (càng nhiều → học được
                       nhiều pattern hơn, nhưng nặng hơn)
        kernel_size  : kích thước cửa sổ trượt
                       kernel=3 → nhìn 3 bước liên tiếp 1 lúc
    """
    def __init__(self, input_size=2, cnn_channels=32, kernel_size=3):
        super(CNNBlock, self).__init__()

        # padding = kernel_size // 2 để giữ nguyên độ dài chuỗi
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels  = input_size,
            out_channels = cnn_channels,
            kernel_size  = kernel_size,
            padding      = pad
        )
        self.bn1 = nn.BatchNorm1d(cnn_channels)

        self.conv2 = nn.Conv1d(
            in_channels  = cnn_channels,
            out_channels = cnn_channels,
            kernel_size  = kernel_size,
            padding      = pad
        )
        self.bn2 = nn.BatchNorm1d(cnn_channels)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x vào: (batch, seq_len, input_size)
        # Conv1d cần: (batch, channels, seq_len) → transpose
        x = x.transpose(1, 2)               # (batch, 2, seq_len)

        x = F.relu(self.bn1(self.conv1(x))) # (batch, cnn_ch, seq_len)
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x))) # (batch, cnn_ch, seq_len)

        # Trả về (batch, seq_len, cnn_channels) cho LSTM
        x = x.transpose(1, 2)
        return x


# ============================================================
# KHỐI 2: Attention - đánh trọng số bước thời gian quan trọng
# ============================================================
class AttentionBlock(nn.Module):
    """
    Additive Attention (Bahdanau-style) trên output của LSTM.

    Tại sao cần Attention?
    - LSTM lấy hidden state ở bước CUỐI để dự đoán
      → thông tin ở bước ĐẦU có thể bị "quên" dần
    - Attention nhìn lại TẤT CẢ hidden states và tự học
      bước nào quan trọng hơn cho dự đoán hiện tại
    - Kết quả là vector context = tổng có trọng số của
      tất cả hidden states

    Cơ chế:
      score(t) = tanh(W * h_t + b)    ← tính điểm mỗi bước
      weight(t) = softmax(score(t))   ← chuẩn hóa thành xác suất
      context  = Σ weight(t) * h_t   ← tổng có trọng số

    Args:
        hidden_size : kích thước hidden state của LSTM
    """
    def __init__(self, hidden_size):
        super(AttentionBlock, self).__init__()

        # Lớp tính attention score
        self.attn  = nn.Linear(hidden_size, hidden_size)
        self.v     = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_size)

        # Bước 1: tính score cho mỗi bước thời gian
        score = torch.tanh(self.attn(lstm_out))  # (batch, seq_len, hidden)
        score = self.v(score)                     # (batch, seq_len, 1)

        # Bước 2: softmax → trọng số tổng = 1
        weight = F.softmax(score, dim=1)          # (batch, seq_len, 1)

        # Bước 3: tổng có trọng số
        # weight: (batch, seq_len, 1)
        # lstm_out: (batch, seq_len, hidden)
        context = (weight * lstm_out).sum(dim=1)  # (batch, hidden)

        return context, weight.squeeze(-1)        # trả thêm weight để visualize


# ============================================================
# MODEL CHÍNH: CNN + LSTM + Attention
# ============================================================
class CNNLSTMAttention(nn.Module):
    """
    Kiến trúc đề xuất: CNN → LSTM → Attention → FC

    Luồng xử lý:
      1. CNN trích xuất pattern cục bộ từ chuỗi kênh thô
      2. LSTM học temporal dependency dài hạn từ features của CNN
      3. Attention đánh trọng số hidden states của LSTM,
         tập trung vào bước thời gian quan trọng nhất
      4. FC layer map ra dự đoán cuối cùng

    So với LSTM thuần (baseline):
      + CNN giúp trích xuất đặc trưng trước khi vào LSTM
        → LSTM nhận input chất lượng cao hơn
      + Attention giúp không bỏ sót thông tin ở đầu chuỗi
        → dự đoán chính xác hơn đặc biệt với chuỗi dài

    So với Transformer:
      + Nhẹ hơn (ít params hơn) → phù hợp edge/embedded
      + Training ổn định hơn với dataset nhỏ
      + Latency thấp hơn khi inference

    Args:
        input_size   : features đầu vào (mặc định 2: real+imag)
        cnn_channels : số filter CNN
        hidden_size  : số neurons LSTM
        num_layers   : số lớp LSTM
        output_size  : số output (mặc định 2: real+imag)
        kernel_size  : kernel size CNN
        dropout      : tỷ lệ dropout
    """
    def __init__(self,
                 input_size   = 2,
                 cnn_channels = 32,
                 hidden_size  = 64,
                 num_layers   = 2,
                 output_size  = 2,
                 kernel_size  = 3,
                 dropout      = 0.1):
        super(CNNLSTMAttention, self).__init__()
        self.name = 'CNN-LSTM-Attention'

        # Tầng 1: CNN
        self.cnn = CNNBlock(
            input_size   = input_size,
            cnn_channels = cnn_channels,
            kernel_size  = kernel_size
        )

        # Tầng 2: LSTM
        # input_size của LSTM = cnn_channels (output của CNN)
        self.lstm = nn.LSTM(
            input_size  = cnn_channels,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0
        )

        # Tầng 3: Attention
        self.attention = AttentionBlock(hidden_size)

        # Tầng 4: FC output
        # Nhận context từ Attention + hidden cuối của LSTM
        # → concat lại → FC → dự đoán
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

        # Khởi tạo weights tốt hơn mặc định
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization cho FC layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x : input tensor (batch, seq_len, 2)

        Returns:
            out     : dự đoán (batch, 2)
            weights : attention weights (batch, seq_len)
                      dùng để visualize model chú ý vào đâu
        """
        # Tầng 1: CNN
        # (batch, seq_len, 2) → (batch, seq_len, cnn_channels)
        cnn_out = self.cnn(x)

        # Tầng 2: LSTM
        # (batch, seq_len, cnn_channels) → (batch, seq_len, hidden)
        lstm_out, (h_n, _) = self.lstm(cnn_out)
        # h_n: (num_layers, batch, hidden) → lấy lớp cuối
        last_hidden = h_n[-1]                    # (batch, hidden)

        # Tầng 3: Attention
        # context: (batch, hidden) — vector tổng hợp có trọng số
        context, weights = self.attention(lstm_out)

        # Tầng 4: Kết hợp context + last_hidden → FC
        # Concat 2 vector: (batch, hidden*2)
        combined = torch.cat([context, last_hidden], dim=1)
        out = self.fc(combined)                  # (batch, 2)

        return out, weights

    def predict(self, x):
        """
        Interface đơn giản hơn khi inference — chỉ trả output.
        Dùng trong evaluate.py để đồng nhất với baselines.
        """
        out, _ = self.forward(x)
        return out


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    batch   = 32
    seq_len = 50
    x = torch.randn(batch, seq_len, 2)

    model = CNNLSTMAttention()
    out, weights = model(x)

    total_params = sum(p.numel() for p in model.parameters())

    print("=" * 55)
    print("  Kiểm tra CNN-LSTM-Attention (model đề xuất)")
    print("=" * 55)
    print(f"  Input shape   : {x.shape}")
    print(f"  Output shape  : {out.shape}")
    print(f"  Weights shape : {weights.shape}  ← attention weights")
    print(f"  Total params  : {total_params:,}")
    print()

    # Kiểm tra attention weights tổng = 1
    w_sum = weights[0].sum().item()
    print(f"  Attention weights sum (sample 0): {w_sum:.6f}  "
          f"← phải ≈ 1.0")

    # Kiểm tra không có NaN
    has_nan = torch.isnan(out).any().item()
    print(f"  NaN in output : {has_nan}  ← phải False")

    print()
    print("  Kiến trúc:")
    print(f"    CNN      : input(2) → {32} channels, kernel=3")
    print(f"    LSTM     : {32} → hidden({64}), layers=2")
    print(f"    Attention: học trọng số {seq_len} bước thời gian")
    print(f"    FC       : {64*2} → {64} → 2")
    print()
    print("  Model OK!")