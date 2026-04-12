import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# KHỐI 1: CNN — trích xuất đặc trưng cục bộ (giữ nguyên)
# ============================================================
class CNNBlock(nn.Module):
    """
    CNN 1 chiều trích xuất pattern cục bộ từ chuỗi kênh.
    Giữ nguyên từ v1 — phần này hoạt động tốt.

    Thêm residual connection bên trong CNN:
    output = F.relu(conv2(conv1(x))) + project(x)
    Giúp gradient chạy thẳng qua, tránh vanishing.
    """
    def __init__(self, input_size=2, cnn_channels=64, kernel_size=3):
        super(CNNBlock, self).__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(input_size,   cnn_channels,
                               kernel_size, padding=pad)
        self.ln1 = nn.LayerNorm(cnn_channels)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels,
                               kernel_size, padding=pad)
        self.ln2 = nn.LayerNorm(cnn_channels)

        # Project input_size → cnn_channels cho residual
        self.residual_proj = nn.Conv1d(input_size, cnn_channels,
                                       kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x_t = x.transpose(1, 2)             

        # Main path
        out = self.conv1(x_t)             
        out = out.transpose(1, 2)          
        out = self.ln1(out)                  
        out = out.transpose(1, 2)            
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)                
        out = out.transpose(1, 2)             
        out = self.ln2(out)                  
        out = out.transpose(1, 2)
        # Residual path
        res = self.residual_proj(x_t)

        out = F.relu(out + res)
        return out.transpose(1, 2)             # (batch, seq_len, cnn_channels)


# ============================================================
# KHỐI 2: Multi-Head Attention — thay thế Single-Head
# ============================================================
class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Attention đặt SAU LSTM.

    Tại sao Multi-Head tốt hơn Single-Head (v1)?
    - Single-Head v1 bị attention collapse: 1 thanh ~0.85, còn lại ~0
      → model chỉ nhìn 1 bước thời gian, bỏ qua lịch sử
    - Multi-Head chia hidden state thành nhiều head độc lập:
      * Head 1: học pattern ngắn hạn (2-3 bước gần)
      * Head 2: học trend dài hạn (10-20 bước trước)
      * Head 3: học điểm đột biến bất thường
      * Head 4: học correlation theo chu kỳ Doppler
    - Mỗi head tự do học pattern riêng → không bị collapse
    - Concat output tất cả head → thông tin phong phú hơn

    Khác Transformer ở chỗ:
    - Transformer: Self-Attention thay thế RNN hoàn toàn
    - Model này: Multi-Head Attention bổ sung SAU LSTM
      → tận dụng temporal memory của LSTM
      → AND multi-perspective attention của Transformer

    Args:
        hidden_size : kích thước hidden state LSTM
        num_heads   : số attention head (phải chia hết hidden_size)
        dropout     : tỷ lệ dropout
    """
    def __init__(self, hidden_size=128, num_heads=4, dropout=0.1):
        super(MultiHeadAttentionBlock, self).__init__()

        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) phải chia hết num_heads ({num_heads})"

        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.head_dim    = hidden_size // num_heads

        # Query, Key, Value projections
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output projection
        self.W_o = nn.Linear(hidden_size, hidden_size)

        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.scale      = self.head_dim ** -0.5  # 1/sqrt(d_k)

    def forward(self, lstm_out):
        """
        Args:
            lstm_out : (batch, seq_len, hidden_size)

        Returns:
            context  : (batch, hidden_size) — vector tổng hợp
            weights  : (batch, seq_len) — để visualize
        """
        batch, seq_len, _ = lstm_out.shape

        # Tính Q, K, V
        Q = self.W_q(lstm_out)
        K = self.W_k(lstm_out)
        V = self.W_v(lstm_out)

        # Reshape thành nhiều head
        # (batch, seq_len, hidden) → (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores  = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Weighted sum
        attn_out = torch.matmul(weights, V)

        # Ghép tất cả head lại
        attn_out = attn_out.transpose(1, 2).contiguous() \
                           .view(batch, seq_len, self.hidden_size)

        # Output projection
        attn_out = self.W_o(attn_out)

        # Residual + LayerNorm
        out = self.layer_norm(lstm_out + attn_out)

        # Lấy bước cuối làm context vector
        context = out[:, -1, :]                          # (batch, hidden)

        # Trả weights trung bình các head để visualize
        vis_weights = weights.mean(dim=1)[:, -1, :]      # (batch, seq_len)

        return context, vis_weights


# ============================================================
# MODEL CHÍNH v2: CNN + LSTM + Multi-Head Attention + Residual
# ============================================================
class CNNLSTMAttention(nn.Module):
    """
    Kiến trúc đề xuất v2.

    Thay đổi so với v1:
    [1] CNN có thêm Residual Connection bên trong
        → gradient chạy thẳng, tránh vanishing
    [2] Single-Head → Multi-Head Attention (4 heads)
        → tránh attention collapse
    [3] LayerNorm sau Attention
        → training ổn định hơn
    [4] hidden_size tăng từ 64 → 128, cnn_channels 32 → 64
        → capacity lớn hơn cho kênh tốc độ cao
    [5] Thêm CNN Residual Connection vào FC
        → thông tin cục bộ từ CNN không bị mất sau LSTM
    [6] GELU thay ReLU trong FC
        → tốt hơn cho regression

    So với Transformer:
    - Giữ LSTM để học temporal ordering
    - Multi-Head Attention bổ sung, không thay thế LSTM
    - Phù hợp hơn với dataset nhỏ (< 10k mẫu)
    """
    def __init__(self,
                 input_size   = 2,
                 cnn_channels = 32,
                 hidden_size  = 64,
                 num_layers   = 2,
                 num_heads    = 2,
                 output_size  = 2,
                 kernel_size  = 3,
                 dropout      = 0.1):
        super(CNNLSTMAttention, self).__init__()
        self.name = 'CNN-LSTM-Attention'

        # Tầng 1: CNN với Residual
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

        # Tầng 3: Multi-Head Attention
        self.attention = MultiHeadAttentionBlock(
            hidden_size = hidden_size,
            num_heads   = num_heads,
            dropout     = dropout
        )

        # Residual từ CNN vào FC
        self.cnn_residual_proj = nn.Linear(cnn_channels, hidden_size)

        # Tầng 4: FC
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
            x : (batch, seq_len, 2)

        Returns:
            out     : (batch, 2)
            weights : (batch, seq_len)
        """
        # Tầng 1: CNN
        cnn_out = self.cnn(x)                        # (batch, seq_len, 64)

        # Tầng 2: LSTM
        lstm_out, _ = self.lstm(cnn_out)             # (batch, seq_len, 128)

        # Tầng 3: Multi-Head Attention
        context, weights = self.attention(lstm_out)  # (batch, 128)

        # Residual từ CNN (bước cuối)
        cnn_last = cnn_out[:, -1, :]                 # (batch, 64)
        cnn_res  = self.cnn_residual_proj(cnn_last)  # (batch, 128)

        # FC
        combined = torch.cat([context, cnn_res], dim=1)  # (batch, 256)
        out      = self.fc(combined)                      # (batch, 2)

        return out, weights

    def predict(self, x):
        """Interface cho train.py và evaluate.py."""
        out, _ = self.forward(x)
        return out


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    batch   = 32
    seq_len = 50
    x       = torch.randn(batch, seq_len, 2)

    model        = CNNLSTMAttention()
    out, weights = model(x)
    total_params = sum(p.numel() for p in model.parameters())

    print("=" * 60)
    print("  CNN-LSTM-Attention v2 (Multi-Head + Residual)")
    print("=" * 60)
    print(f"  Input        : {x.shape}")
    print(f"  Output       : {out.shape}")
    print(f"  Weights      : {weights.shape}")
    print(f"  Total params : {total_params:,}")
    print()

    w_max = weights[0].max().item()
    w_std = weights[0].std().item()
    w_sum = weights[0].sum().item()
    print(f"  Weight sum : {w_sum:.4f}  ")
    print(f"  Weight max : {w_max:.4f}  ")
    print(f"  Weight std : {w_std:.4f}  ")
    print()

    has_nan = torch.isnan(out).any().item()
    print(f"  NaN output : {has_nan} ")
    print()
