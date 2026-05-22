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
# CẤU HÌNH MẶC ĐỊNH — Fair comparison (~400K params mỗi model)
# ============================================================
DEFAULT_CONFIGS = {
    'RNN': {
        'hidden_size': 290,
        'num_layers' : 2,
    },
    'LSTM': {
        'hidden_size': 135,
        'num_layers' : 2,
    },
    'GRU': {
        'hidden_size': 160,
        'num_layers' : 2,
    },
    'Transformer': {
        'd_model'       : 120,
        'nhead'         : 4,
        'num_layers'    : 2,
        'dim_feedforward': 480,
    },
}


# ============================================================
# MODEL 1: RNN Baseline
# ============================================================
class RNNModel(nn.Module):
    """
    Vanilla RNN cho Massive MIMO channel prediction.

    Fair comparison config: hidden_size=290, num_layers=2 → ~402K params
    """
    def __init__(self, input_size=FEATURE_SIZE, hidden_size=290,
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

    Fair comparison config: hidden_size=135, num_layers=2 → ~394K params
    """
    def __init__(self, input_size=FEATURE_SIZE, hidden_size=135,
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
    GRU cho Massive MIMO channel prediction.

    Fair comparison config: hidden_size=160, num_layers=2 → ~396K params
    """
    def __init__(self, input_size=FEATURE_SIZE, hidden_size=160,
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
# MODEL 4: Transformer (Model đề xuất)
# ============================================================
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""
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
    Transformer cho Massive MIMO channel prediction. (Model đề xuất)

    Fair comparison config:
        d_model=120, nhead=4, num_layers=2, dim_feedforward=480 → ~411K params
    """
    def __init__(self, input_size=FEATURE_SIZE, d_model=120,
                 nhead=4, num_layers=2, output_size=FEATURE_SIZE,
                 dim_feedforward=480, dropout=0.1):
        super().__init__()
        self.name       = 'Transformer'
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
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
def get_baseline(name, hidden_size=None, num_layers=None,
                 feature_size=FEATURE_SIZE, **kwargs):
    """
    Tạo baseline model theo tên.

    Nếu không truyền hidden_size / num_layers, tự lấy từ DEFAULT_CONFIGS.

    Args:
        name         : 'RNN', 'LSTM', 'GRU', 'Transformer'
        hidden_size  : override hidden_size (hoặc d_model cho Transformer)
        num_layers   : override num_layers
        feature_size : 2*N_TX*N_RX (mặc định 256)
        **kwargs     : tham số riêng cho Transformer (nhead, dim_feedforward)

    Returns:
        Instance của model tương ứng.
    """
    if name not in DEFAULT_CONFIGS:
        raise ValueError(f"Không có model '{name}'. "
                         f"Chọn: {list(DEFAULT_CONFIGS.keys())}")

    cfg = DEFAULT_CONFIGS[name].copy()

    if name == 'Transformer':
        d_model        = hidden_size  if hidden_size is not None else cfg['d_model']
        nl             = num_layers   if num_layers  is not None else cfg['num_layers']
        nhead          = kwargs.get('nhead',          cfg['nhead'])
        dim_feedforward = kwargs.get('dim_feedforward', cfg['dim_feedforward'])
        return TransformerModel(
            input_size      = feature_size,
            d_model         = d_model,
            nhead           = nhead,
            num_layers      = nl,
            output_size     = feature_size,
            dim_feedforward = dim_feedforward,
        )

    h  = hidden_size if hidden_size is not None else cfg['hidden_size']
    nl = num_layers  if num_layers  is not None else cfg['num_layers']

    models = {
        'RNN' : RNNModel( input_size=feature_size, hidden_size=h,
                          num_layers=nl, output_size=feature_size),
        'LSTM': LSTMModel(input_size=feature_size, hidden_size=h,
                          num_layers=nl, output_size=feature_size),
        'GRU' : GRUModel( input_size=feature_size, hidden_size=h,
                          num_layers=nl, output_size=feature_size),
    }
    return models[name]


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    batch   = 16
    seq_len = 50
    feat    = FEATURE_SIZE  # 256
    x = torch.randn(batch, seq_len, feat)

    print("=" * 65)
    print(f"  Fair Comparison — Massive MIMO Baselines + Transformer")
    print(f"  N_TX={N_TX}, N_RX={N_RX} | Feature size = {feat}")
    print(f"  Target: ~400K params per model")
    print("=" * 65)
    print(f"  {'Model':<14} {'Params':>10}  {'Output shape'}")
    print("  " + "-" * 50)

    for name in ['RNN', 'LSTM', 'GRU', 'Transformer']:
        model  = get_baseline(name, feature_size=feat)
        out    = model(x)
        params = sum(p.numel() for p in model.parameters())
        diff   = (params - 400_000) / 400_000 * 100
        mark   = " ✓" if abs(diff) < 10 else " ✗"
        print(f"  {name:<14} {params:>10,}  {str(tuple(out.shape)):<16}"
              f"  ({diff:+.1f}%){mark}")

    print("\nAll models OK!")
