import torch
import torch.nn as nn
import math


# ============================================================
# MODEL 1: RNN (đơn giản nhất, dùng làm lower bound)
# ============================================================
class RNNModel(nn.Module):
    """
    Vanilla RNN - model đơn giản nhất.
    Thường cho kết quả kém nhất do vanishing gradient.
    Dùng làm lower bound để so sánh.

    Args:
        input_size  : số features mỗi bước thời gian (2: real + imag)
        hidden_size : số neurons trong hidden layer
        num_layers  : số lớp RNN chồng nhau
        output_size : số output (2: real + imag)
    """
    def __init__(self, input_size=2, hidden_size=64,
                 num_layers=2, output_size=2):
        super(RNNModel, self).__init__()
        self.name = 'RNN'

        self.rnn = nn.RNN(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,      # input shape: (batch, seq, feature)
            dropout     = 0.1 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, 2)
        out, _ = self.rnn(x)         # out: (batch, seq_len, hidden)
        out = out[:, -1, :]          # lấy bước cuối: (batch, hidden)
        out = self.fc(out)           # (batch, 2)
        return out


# ============================================================
# MODEL 2: LSTM (baseline chính)
# ============================================================
class LSTMModel(nn.Module):
    """
    LSTM - cải thiện RNN nhờ cơ chế forget/input/output gate.
    Đây là baseline quan trọng nhất để so sánh với model đề xuất.

    Điểm khác RNN: có cell state riêng giúp nhớ thông tin dài hạn tốt hơn.
    """
    def __init__(self, input_size=2, hidden_size=64,
                 num_layers=2, output_size=2):
        super(LSTMModel, self).__init__()
        self.name = 'LSTM'

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.1 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, 2)
        out, _ = self.lstm(x)        # out: (batch, seq_len, hidden)
        out = out[:, -1, :]          # lấy bước cuối: (batch, hidden)
        out = self.fc(out)           # (batch, 2)
        return out


# ============================================================
# MODEL 3: GRU (nhẹ hơn LSTM, thường nhanh hơn)
# ============================================================
class GRUModel(nn.Module):
    """
    GRU - phiên bản đơn giản hóa của LSTM.
    Ít parameters hơn LSTM, training nhanh hơn,
    nhưng thường accuracy tương đương hoặc chỉ kém một chút.

    Phù hợp khi cần deploy lên thiết bị có tài nguyên hạn chế (edge).
    """
    def __init__(self, input_size=2, hidden_size=64,
                 num_layers=2, output_size=2):
        super(GRUModel, self).__init__()
        self.name = 'GRU'

        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.1 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, 2)
        out, _ = self.gru(x)         # out: (batch, seq_len, hidden)
        out = out[:, -1, :]          # lấy bước cuối: (batch, hidden)
        out = self.fc(out)           # (batch, 2)
        return out


# ============================================================
# MODEL 4: Transformer (state-of-the-art baseline)
# ============================================================
class PositionalEncoding(nn.Module):
    """
    Positional Encoding cho Transformer.
    Vì Transformer không có khái niệm thứ tự thời gian như RNN,
    cần thêm thông tin vị trí vào input.

    Dùng công thức sin/cos chuẩn từ paper "Attention Is All You Need".
    """
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # chiều chẵn
        pe[:, 1::2] = torch.cos(position * div_term)   # chiều lẻ
        pe = pe.unsqueeze(0)                            # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer Encoder cho time-series channel prediction.
    State-of-the-art trong nhiều bài toán sequence.

    Cơ chế Multi-Head Self-Attention cho phép model nhìn
    toàn bộ chuỗi input và tự học mối quan hệ giữa các bước.

    Điểm mạnh: xử lý song song, không bị vanishing gradient.
    Điểm yếu: cần nhiều data hơn, nặng hơn LSTM/GRU.
    """
    def __init__(self, input_size=2, d_model=64, nhead=4,
                 num_layers=2, output_size=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.name = 'Transformer'

        # Project input_size -> d_model
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers
        )

        # Output layer
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        # x: (batch, seq_len, 2)
        x = self.input_proj(x)      # (batch, seq_len, d_model)
        x = self.pos_enc(x)         # thêm positional encoding
        x = self.transformer(x)     # (batch, seq_len, d_model)
        x = x[:, -1, :]            # lấy bước cuối: (batch, d_model)
        x = self.fc(x)             # (batch, 2)
        return x


# ============================================================
# FACTORY FUNCTION: tạo model theo tên
# ============================================================
def get_baseline(name, hidden_size=64, num_layers=2):
    """
    Tạo baseline model theo tên.

    Args:
        name        : 'RNN', 'LSTM', 'GRU', 'Transformer'
        hidden_size : số neurons ẩn
        num_layers  : số lớp

    Returns:
        model : PyTorch model tương ứng

    Ví dụ:
        model = get_baseline('LSTM')
        model = get_baseline('Transformer', hidden_size=128)
    """
    models = {
        'RNN':         RNNModel(hidden_size=hidden_size,
                                num_layers=num_layers),
        'LSTM':        LSTMModel(hidden_size=hidden_size,
                                 num_layers=num_layers),
        'GRU':         GRUModel(hidden_size=hidden_size,
                                num_layers=num_layers),
        'Transformer': TransformerModel(d_model=hidden_size,
                                        num_layers=num_layers),
    }
    if name not in models:
        raise ValueError(f"Model '{name}' không tồn tại. "
                         f"Chọn trong: {list(models.keys())}")
    return models[name]


# ============================================================
# QUICK TEST: chạy file này trực tiếp để kiểm tra shape
# ============================================================
if __name__ == "__main__":
    batch    = 32
    seq_len  = 50
    x = torch.randn(batch, seq_len, 2)

    print("=" * 50)
    print("  Kiểm tra shape output các baseline model")
    print(f"  Input: {x.shape}")
    print("=" * 50)

    for name in ['RNN', 'LSTM', 'GRU', 'Transformer']:
        model  = get_baseline(name)
        out    = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name:<12} output: {out.shape}  "
              f"params: {params:,}")

    print("\nTất cả model OK!")