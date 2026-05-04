import torch
import torch.nn as nn
import math

# ============================================================
# THAM SỐ MASSIVE MIMO
# ============================================================
N_TX         = 64
N_RX         = 2
FEATURE_SIZE = 2 * N_TX * N_RX  # = 256


# ============================================================
# MODEL 1: RNN Baseline
# ============================================================
class RNNModel(nn.Module):
    """
    Vanilla RNN cho Massive MIMO channel prediction.

    SISO → MIMO thay đổi:
        input_size : 2 → 128  (real+imag của 64 anten)
        output_size: 2 → 128
    """
    def __init__(self, input_size=FEATURE_SIZE, hidden_size=128,
                 num_layers=2, output_size=FEATURE_SIZE):
        super().__init__()
        self.name = 'RNN'
        self.rnn  = nn.RNN(input_size, hidden_size, num_layers,
                           batch_first=True,
                           dropout=0.1 if num_layers > 1 else 0)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


# ============================================================
# MODEL 2: LSTM Baseline
# ============================================================
class LSTMModel(nn.Module):
    """
    LSTM cho Massive MIMO channel prediction.
    Baseline quan trọng nhất để so sánh với CNN-LSTM-Attention.

    Thách thức MIMO: output 128 chiều thay vì 2 chiều như SISO
    → LSTM phải học đồng thời 64 hệ số kênh phức
    """
    def __init__(self, input_size=FEATURE_SIZE, hidden_size=128,
                 num_layers=2, output_size=FEATURE_SIZE):
        super().__init__()
        self.name  = 'LSTM'
        self.lstm  = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True,
                             dropout=0.1 if num_layers > 1 else 0)
        self.fc    = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ============================================================
# MODEL 3: GRU Baseline
# ============================================================
class GRUModel(nn.Module):
    """
    GRU cho Massive MIMO.
    Nhẹ hơn LSTM, phù hợp deploy thiết bị hạn chế tài nguyên.
    """
    def __init__(self, input_size=FEATURE_SIZE, hidden_size=128,
                 num_layers=2, output_size=FEATURE_SIZE):
        super().__init__()
        self.name = 'GRU'
        self.gru  = nn.GRU(input_size, hidden_size, num_layers,
                           batch_first=True,
                           dropout=0.1 if num_layers > 1 else 0)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# ============================================================
# MODEL 4: Transformer Baseline
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe       = torch.zeros(max_len, d_model)
        pos      = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModel(nn.Module):
    """
    Transformer cho Massive MIMO channel prediction.

    Ưu điểm với MIMO: Self-Attention có thể học tương quan
    không gian giữa các anten (spatial correlation) qua thời gian.

    Với input 128 chiều, d_model phải đủ lớn để biểu diễn
    toàn bộ thông tin không gian-thời gian.
    """
    def __init__(self, input_size=FEATURE_SIZE, d_model=128,
                 nhead=8, num_layers=2, output_size=FEATURE_SIZE,
                 dropout=0.1):
        super().__init__()
        self.name       = 'Transformer'
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc          = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.pos_enc(self.input_proj(x))
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


# ============================================================
# FACTORY FUNCTION
# ============================================================
def get_baseline(name, hidden_size=128, num_layers=2,
                 feature_size=FEATURE_SIZE):
    """
    Tạo baseline model theo tên.

    Args:
        name         : 'RNN', 'LSTM', 'GRU', 'Transformer'
        hidden_size  : số hidden units
        num_layers   : số tầng
        feature_size : 2*N_TX*N_RX (mặc định 128 cho 64×1 MIMO)
    """
    models = {
        'RNN': RNNModel(
            input_size=feature_size, hidden_size=hidden_size,
            num_layers=num_layers, output_size=feature_size),
        'LSTM': LSTMModel(
            input_size=feature_size, hidden_size=hidden_size,
            num_layers=num_layers, output_size=feature_size),
        'GRU': GRUModel(
            input_size=feature_size, hidden_size=hidden_size,
            num_layers=num_layers, output_size=feature_size),
        'Transformer': TransformerModel(
            input_size=feature_size, d_model=hidden_size,
            num_layers=num_layers, output_size=feature_size),
    }
    if name not in models:
        raise ValueError(f"Không có model '{name}'. "
                         f"Chọn: {list(models.keys())}")
    return models[name]


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    batch   = 16
    seq_len = 50
    feat    = FEATURE_SIZE  # 128
    x = torch.randn(batch, seq_len, feat)

    print("=" * 55)
    print(f"  Massive MIMO Baselines — Feature size = {feat}")
    print(f"  (N_TX={N_TX}, N_RX={N_RX}, 2×{N_TX}×{N_RX} = {feat})")
    print("=" * 55)

    for name in ['RNN', 'LSTM', 'GRU', 'Transformer']:
        model  = get_baseline(name)
        out    = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name:<12} in:{x.shape} → out:{out.shape} "
              f"| params: {params:,}")

    print("\nAll baselines OK!")