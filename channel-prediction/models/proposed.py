import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# KHỐI 1: CNN — trích xuất đặc trưng cục bộ (giữ nguyên)
# ============================================================
class CNNBlock(nn.Module):
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
# MODEL ABLATION: CNN + LSTM
# ============================================================
class CNNLSTMNoAttention(nn.Module):
    def __init__(self,
                 input_size   = 2,
                 cnn_channels = 32,
                 hidden_size  = 64,
                 num_layers   = 2,
                 output_size  = 2,
                 kernel_size  = 3,
                 dropout      = 0.1):
        super(CNNLSTMNoAttention, self).__init__()
        self.name = 'CNN-LSTM'

        # Tầng 1: CNN (giống CNNLSTMAttention)
        self.cnn = CNNBlock(
            input_size   = input_size,
            cnn_channels = cnn_channels,
            kernel_size  = kernel_size
        )

        # Tầng 2: LSTM (giống CNNLSTMAttention)
        self.lstm = nn.LSTM(
            input_size  = cnn_channels,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0
        )

        # Không có Attention — LayerNorm sau LSTM để ổn định
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Residual từ CNN vào FC (giữ nguyên như CNNLSTMAttention)
        self.cnn_residual_proj = nn.Linear(cnn_channels, hidden_size)

        # FC giống CNNLSTMAttention để so sánh công bằng
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
            out : (batch, 2)
        """
        # Tầng 1: CNN
        cnn_out = self.cnn(x)                     # (batch, seq_len, cnn_channels)

        # Tầng 2: LSTM
        lstm_out, _ = self.lstm(cnn_out)           # (batch, seq_len, hidden_size)

        # Lấy bước cuối (không qua Attention)
        context = self.layer_norm(lstm_out[:, -1, :])  # (batch, hidden_size)

        # Residual từ CNN
        cnn_last = cnn_out[:, -1, :]               # (batch, cnn_channels)
        cnn_res  = self.cnn_residual_proj(cnn_last) # (batch, hidden_size)

        # FC
        combined = torch.cat([context, cnn_res], dim=1)  # (batch, hidden_size*2)
        out      = self.fc(combined)                      # (batch, 2)

        return out

    def predict(self, x):
        """Interface cho train.py và evaluate.py."""
        return self.forward(x)


# ============================================================
# MODEL BASELINE: LSTM thuần (không CNN, không Attention)
# ============================================================
class LSTMModel(nn.Module):
    """Baseline LSTM thuần để so sánh trong ablation study."""
    def __init__(self,
                 input_size  = 2,
                 hidden_size = 64,
                 num_layers  = 2,
                 output_size = 2,
                 dropout     = 0.1):
        super(LSTMModel, self).__init__()
        self.name = 'LSTM'
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ============================================================
# MODEL ABLATION 2: LSTM + Attention (KHÔNG có CNN)
# ============================================================
class LSTMAttention(nn.Module):
    """
    Phiên bản ablation: bỏ CNN block, chỉ giữ LSTM + Multi-Head Attention.
    Mục đích: chứng minh CNN có vai trò gì trong model đầy đủ.

    So với CNNLSTMAttention:
      - Input đi thẳng vào LSTM (không qua CNN)
      - Không có residual từ CNN
      - FC chỉ nhận context từ Attention (hidden_size, không phải hidden*2)
    """
    def __init__(self,
                 input_size  = 2,
                 hidden_size = 64,
                 num_layers  = 2,
                 num_heads   = 2,
                 output_size = 2,
                 dropout     = 0.1):
        super(LSTMAttention, self).__init__()
        self.name = 'LSTM-Attention'

        # Tầng 1: LSTM (input thẳng, không qua CNN)
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0
        )

        # Tầng 2: Multi-Head Attention (giống CNNLSTMAttention)
        self.attention = MultiHeadAttentionBlock(
            hidden_size = hidden_size,
            num_heads   = num_heads,
            dropout     = dropout
        )

        # Tầng 3: FC — input chỉ là context (hidden_size), không có CNN residual
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
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

    def forward(self, x):
        """
        Args:
            x : (batch, seq_len, 2)

        Returns:
            out     : (batch, 2)
            weights : (batch, seq_len) — attention weights để visualize
        """
        # Tầng 1: LSTM
        lstm_out, _ = self.lstm(x)                       # (B, seq_len, hidden)

        # Tầng 2: Multi-Head Attention
        context, weights = self.attention(lstm_out)      # (B, hidden)

        # Tầng 3: FC
        out = self.fc(context)                           # (B, 2)

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

    print("=" * 60)
    print("  Ablation Study — Kiểm tra 4 variant")
    print("=" * 60)

    # Variant 1: LSTM thuần
    m_lstm     = LSTMModel(hidden_size=64, num_layers=2)
    out_lstm   = m_lstm(x)
    p_lstm     = sum(p.numel() for p in m_lstm.parameters())
    print(f"  [1] LSTM only             | output: {out_lstm.shape} "
          f"| params: {p_lstm:,}")

    # Variant 2: CNN-LSTM (không Attention)
    m_no_attn  = CNNLSTMNoAttention()
    out2       = m_no_attn(x)
    p_no_attn  = sum(p.numel() for p in m_no_attn.parameters())
    print(f"  [2] CNN-LSTM (no Attn)    | output: {out2.shape} "
          f"| params: {p_no_attn:,}")

    # Variant 3: LSTM-Attention (không CNN)
    m_no_cnn   = LSTMAttention()
    out3, _    = m_no_cnn(x)
    p_no_cnn   = sum(p.numel() for p in m_no_cnn.parameters())
    print(f"  [3] LSTM-Attention (noCNN)| output: {out3.shape} "
          f"| params: {p_no_cnn:,}")

    # Variant 4: CNN-LSTM-Attention (đầy đủ)
    m_full     = CNNLSTMAttention()
    out4, w    = m_full(x)
    p_full     = sum(p.numel() for p in m_full.parameters())
    print(f"  [4] CNN-LSTM-Attention *  | output: {out4.shape} "
          f"| params: {p_full:,}")

    print(f"\n  Delta params do CNN     : +{p_no_attn - p_lstm:>6,}  "
          f"(LSTM -> CNN-LSTM)")
    print(f"  Delta params do Attn    : +{p_no_cnn - p_lstm:>6,}  "
          f"(LSTM -> LSTM-Attn)")
    print(f"  Delta params do CNN+Attn: +{p_full   - p_lstm:>6,}  "
          f"(LSTM -> Full)")

    # Sanity check NaN
    for name, t in [('LSTM',out_lstm),('CNN-LSTM',out2),
                    ('LSTM-Attn',out3),('Full',out4)]:
        nan = torch.isnan(t).any().item()
        print(f"  NaN check ({name:<10}): {nan}")