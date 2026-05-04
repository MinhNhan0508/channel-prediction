import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# THAM SỐ MASSIVE MIMO
# ============================================================
N_TX         = 64
N_RX         = 2
FEATURE_SIZE = 2 * N_TX * N_RX  # = 256


# ============================================================
# KHỐI 1: CNN — trích xuất đặc trưng không gian-thời gian
# ============================================================
class CNNBlock(nn.Module):
    """
    CNN 1D cho Massive MIMO.

    Trong SISO, CNN học pattern theo thời gian (temporal pattern).
    Trong Massive MIMO, CNN còn học thêm tương quan không gian
    (spatial correlation) giữa các anten vì feature vector
    chứa thông tin của 64 anten được flatten liên tiếp.

    Anten liền kề nhau trong ULA có tương quan không gian cao
    → CNN kernel=3 có thể phát hiện tương quan này.

    Residual connection giúp gradient chạy thẳng qua,
    đặc biệt quan trọng khi input có 128 chiều.
    """
    def __init__(self, input_size=FEATURE_SIZE,
                 cnn_channels=64, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(input_size, cnn_channels,
                               kernel_size, padding=pad)
        self.ln1   = nn.LayerNorm(cnn_channels)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels,
                               kernel_size, padding=pad)
        self.ln2   = nn.LayerNorm(cnn_channels)

        self.res_proj = nn.Conv1d(input_size, cnn_channels,
                                  kernel_size=1)
        self.dropout  = nn.Dropout(0.1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        xt = x.transpose(1, 2)  # (batch, input_size, seq_len)

        # Main path
        out = self.conv1(xt).transpose(1, 2)
        out = F.relu(self.ln1(out)).transpose(1, 2)
        out = self.dropout(out)
        out = self.conv2(out).transpose(1, 2)
        out = self.ln2(out).transpose(1, 2)

        # Residual
        res = self.res_proj(xt)
        out = F.relu(out + res)
        return out.transpose(1, 2)  # (batch, seq_len, cnn_channels)


# ============================================================
# KHỐI 2: Spatial-Temporal Attention
# ============================================================
class SpatialTemporalAttention(nn.Module):
    """
    Multi-Head Attention cho Massive MIMO.

    Điểm khác biệt quan trọng so với SISO:
    Trong MIMO, Attention cần học 2 loại tương quan:
      1. Temporal correlation : bước thời gian nào quan trọng?
         (giống SISO attention)
      2. Spatial correlation  : tương quan anten được học qua
         LSTM và được Attention tổng hợp

    Với hidden_size lớn hơn (256 thay vì 64 ở SISO),
    num_heads cũng tăng (8 thay vì 2) để học đa dạng hơn.

    Dùng Global Average Pooling (mean) thay vì chỉ lấy bước cuối
    → tránh attention collapse (vấn đề đã phát hiện ở SISO).
    """
    def __init__(self, hidden_size=256, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.head_dim    = hidden_size // num_heads
        self.scale       = self.head_dim ** -0.5

        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size)

        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, lstm_out):
        """
        Args:
            lstm_out : (batch, seq_len, hidden_size)
        Returns:
            context     : (batch, hidden_size) — global mean pooling
            vis_weights : (batch, seq_len) — để visualize
        """
        B, T, _ = lstm_out.shape

        Q = self.W_q(lstm_out).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        K = self.W_k(lstm_out).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        V = self.W_v(lstm_out).view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        scores  = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        attn_out = torch.matmul(weights, V)
        attn_out = attn_out.transpose(1,2).contiguous().view(B, T, self.hidden_size)
        attn_out = self.W_o(attn_out)

        out = self.layer_norm(lstm_out + attn_out)

        # ✅ Global Average Pooling: tổng hợp toàn bộ chuỗi
        # Tránh attention collapse (không chỉ lấy bước cuối)
        context = out.mean(dim=1)  # (batch, hidden_size)

        # Weights để visualize (trung bình qua tất cả head và query)
        vis_weights = weights.mean(dim=1).mean(dim=1)  # (batch, seq_len)

        return context, vis_weights


# ============================================================
# MODEL CHÍNH: CNN + LSTM + Spatial-Temporal Attention
# ============================================================
class CNNLSTMAttention(nn.Module):
    """
    Kiến trúc đề xuất cho Massive MIMO Channel Prediction.

    Thay đổi so với phiên bản SISO:

    [1] input_size: 2 → 128 (64 anten × 2)
        Mỗi bước thời gian chứa thông tin của tất cả 64 anten

    [2] CNN: học spatial correlation giữa các anten
        + temporal pattern theo thời gian

    [3] hidden_size: 64 → 256
        Cần nhiều hơn để biểu diễn không gian 64 anten

    [4] num_heads: 2 → 8
        Nhiều head hơn để học đa chiều spatial-temporal

    [5] output_size: 2 → 128
        Dự đoán toàn bộ vector kênh 64 anten cùng lúc

    [6] Attention: Global Average Pooling (sửa lỗi collapse SISO)

    [7] FC output: tách thành real và imag riêng
        → ổn định hơn khi output có 128 chiều

    Kiến trúc tổng thể:
        Input (batch, 50, 128)
            ↓ CNN Block (Spatial + Local Temporal)
        (batch, 50, 64)
            ↓ LSTM (Long-range Temporal)
        (batch, 50, 256)
            ↓ Spatial-Temporal Attention (Global pooling)
        (batch, 256)  +  CNN residual (batch, 64→256)
            ↓ Concat → (batch, 512)
            ↓ FC
        Output (batch, 128)
    """
    def __init__(self,
                 input_size   = FEATURE_SIZE,   # 128
                 cnn_channels = 64,
                 hidden_size  = 256,
                 num_layers   = 2,
                 num_heads    = 8,
                 output_size  = FEATURE_SIZE,   # 128
                 kernel_size  = 3,
                 dropout      = 0.1):
        super().__init__()
        self.name = 'CNN-LSTM-Attention'

        # Tầng 1: CNN
        self.cnn = CNNBlock(
            input_size   = input_size,
            cnn_channels = cnn_channels,
            kernel_size  = kernel_size
        )

        # Tầng 2: LSTM
        self.lstm = nn.LSTM(
            input_size  = cnn_channels,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0
        )

        # Tầng 3: Spatial-Temporal Attention
        self.attention = SpatialTemporalAttention(
            hidden_size = hidden_size,
            num_heads   = num_heads,
            dropout     = dropout
        )

        # Residual từ CNN (bước cuối) vào FC
        self.cnn_proj = nn.Linear(cnn_channels, hidden_size)

        # Tầng 4: FC — output 128 chiều (64 anten × 2)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        """
        Args:
            x : (batch, seq_len, 128)  ← 64 anten × 2 (real+imag)

        Returns:
            out     : (batch, 128)     ← dự đoán kênh bước tiếp theo
            weights : (batch, seq_len) ← attention weights để visualize
        """
        # CNN: trích xuất spatial + local temporal features
        cnn_out = self.cnn(x)                     # (B, seq_len, 64)

        # LSTM: học long-range temporal dependency
        lstm_out, _ = self.lstm(cnn_out)           # (B, seq_len, 256)

        # Spatial-Temporal Attention: global pooling toàn chuỗi
        context, weights = self.attention(lstm_out) # (B, 256)

        # CNN residual: thông tin cục bộ không bị mất sau LSTM
        cnn_last = cnn_out[:, -1, :]               # (B, 64)
        cnn_res  = self.cnn_proj(cnn_last)         # (B, 256)

        # Concat và dự đoán
        combined = torch.cat([context, cnn_res], dim=1)  # (B, 512)
        out      = self.fc(combined)                      # (B, 128)

        return out, weights

    def predict(self, x):
        """Interface đơn giản cho train/evaluate."""
        out, _ = self.forward(x)
        return out


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    B, T, F = 16, 50, FEATURE_SIZE  # F=128
    x = torch.randn(B, T, F)

    model        = CNNLSTMAttention()
    out, weights = model(x)
    params       = sum(p.numel() for p in model.parameters())

    print("=" * 65)
    print("  CNN-LSTM-Attention — Massive MIMO Version")
    print(f"  N_TX={N_TX}, N_RX={N_RX}, Feature={F}")
    print("=" * 65)
    print(f"  Input   : {x.shape}")
    print(f"  Output  : {out.shape}   ← 64 anten × 2 (real+imag)")
    print(f"  Weights : {weights.shape}")
    print(f"  Params  : {params:,}")
    print()

    w_max = weights[0].max().item()
    w_std = weights[0].std().item()
    print(f"  Weight max : {w_max:.4f}  (phải < 0.15 để không collapse)")
    print(f"  Weight std : {w_std:.4f}  (càng cao càng phân biệt tốt)")
    print(f"  NaN output : {torch.isnan(out).any().item()}")
    print()
    print("  Kiến trúc:")
    print(f"    CNN Block          : {F} → 64 channels, kernel=3")
    print(f"    LSTM               : 64 → 256, layers=2")
    print(f"    Spatial-Temp Attn  : 8 heads × 32 dim (global pooling)")
    print(f"    CNN Residual       : 64 → 256")
    print(f"    FC                 : 512 → 256 → {F}")
    print()
    print("  Model OK!")