import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.baselines import get_baseline
from models.proposed  import CNNLSTMAttention


# ============================================================
# CẤU HÌNH TRAINING
# ============================================================

CONFIG = {
    'epochs'     : 150,     # tăng từ 50
    'batch_size' : 64,
    'lr'         : 5e-4,    # giảm từ 1e-3
    'patience'   : 20,      # tăng từ 10
    'hidden_size': 64,
    'num_layers' : 2,
    'data_dir'   : 'data',
    'save_dir'   : 'results/checkpoints',
    'velocities' : [30, 120, 300]
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# HÀM 1: Load data từ file .npz
# ============================================================
def load_data(velocity, data_dir='data', batch_size=64):
    """
    Load dataset đã sinh từ generate_channel.py.
    Chuyển numpy arrays → PyTorch tensors → DataLoader.

    DataLoader tự động:
      - Chia thành mini-batches
      - Shuffle data train mỗi epoch (tránh overfit thứ tự)

    Args:
        velocity : tốc độ (30, 120, 300)
        data_dir : thư mục chứa file .npz
        batch_size: số mẫu mỗi batch

    Returns:
        train_loader, test_loader, số mẫu train, số mẫu test
    """
    path = os.path.join(data_dir, f'channel_v{velocity}kmh.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy {path}. "
            f"Chạy generate_channel.py trước!"
        )

    d = np.load(path)
    X_train = torch.FloatTensor(d['X_train'])
    y_train = torch.FloatTensor(d['y_train'])
    X_test  = torch.FloatTensor(d['X_test'])
    y_test  = torch.FloatTensor(d['y_test'])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size = batch_size,
        shuffle    = True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size = batch_size,
        shuffle    = False      # test không cần shuffle
    )

    return train_loader, test_loader, len(X_train), len(X_test)


# ============================================================
# HÀM 2: Train 1 epoch
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Chạy 1 vòng train qua toàn bộ training data.

    Mỗi batch:
      1. Forward pass  : model tính output
      2. Tính loss     : so sánh output với target
      3. Backward pass : tính gradient
      4. Update weights: optimizer cập nhật params

    Returns:
        loss trung bình của epoch này
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        # Forward: proposed model trả (out, weights), baselines trả out
        if hasattr(model, 'predict'):
            out = model.predict(X_batch)
        else:
            out = model(X_batch)

        loss = criterion(out, y_batch)
        loss.backward()

        # Gradient clipping: tránh gradient bùng nổ
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# HÀM 3: Đánh giá trên test set
# ============================================================
def evaluate(model, loader, criterion, device):
    """
    Chạy model trên test set, không cập nhật weights.
    model.eval() + torch.no_grad() để tắt dropout và không
    tính gradient → nhanh hơn và kết quả ổn định hơn.

    Returns:
        loss trung bình trên test set
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if hasattr(model, 'predict'):
                out = model.predict(X_batch)
            else:
                out = model(X_batch)

            total_loss += criterion(out, y_batch).item()

    return total_loss / len(loader)


# ============================================================
# HÀM 4: Train 1 model hoàn chỉnh với early stopping
# ============================================================
def train_model(model, train_loader, test_loader,
                config, model_name, velocity):
    """
    Train model qua nhiều epochs với early stopping.

    Early stopping:
      - Theo dõi val_loss sau mỗi epoch
      - Nếu val_loss không cải thiện sau 'patience' epochs → dừng
      - Lưu lại weights tốt nhất (best checkpoint)
      - Tránh overfit và tiết kiệm thời gian

    Args:
        model        : PyTorch model cần train
        train_loader : DataLoader training data
        test_loader  : DataLoader test data
        config       : dict cấu hình
        model_name   : tên model (để lưu file)
        velocity     : tốc độ (để lưu file)

    Returns:
        history: dict chứa train_loss và val_loss theo epoch
    """
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # Learning rate scheduler: giảm lr khi val_loss không giảm
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    os.makedirs(config['save_dir'], exist_ok=True)
    save_path = os.path.join(
        config['save_dir'],
        f'{model_name}_v{velocity}kmh.pth'
    )

    best_val_loss = float('inf')
    patience_count = 0
    history = {'train_loss': [], 'val_loss': []}

    print(f"\n  Training {model_name} | v={velocity}km/h | "
          f"device={DEVICE}")
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | "
          f"{'Val Loss':>10} | {'Time':>6}")
    print("  " + "-" * 42)

    for epoch in range(1, config['epochs'] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        elapsed = time.time() - t0

        # In mỗi 5 epoch để không spam quá nhiều
        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>5} | {train_loss:>10.6f} | "
                  f"{val_loss:>10.6f} | {elapsed:>5.1f}s")

        # Lưu checkpoint tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save({
                'epoch'     : epoch,
                'model_state': model.state_dict(),
                'val_loss'  : best_val_loss,
                'config'    : config
            }, save_path)
        else:
            patience_count += 1

        # Early stopping
        if patience_count >= config['patience']:
            print(f"  Early stopping tại epoch {epoch} "
                  f"(best val_loss={best_val_loss:.6f})")
            break

    print(f"  Best val_loss: {best_val_loss:.6f} → saved: {save_path}")
    return history


# ============================================================
# HÀM 5: Tạo tất cả model
# ============================================================
def get_all_models(config):
    """
    Tạo dict chứa tất cả 5 model.
    Mỗi lần gọi hàm này tạo model mới hoàn toàn (random weights).
    """
    h = config['hidden_size']
    num_layers = config['num_layers']

    return {
        'RNN'              : get_baseline('RNN',         h, num_layers),
        'LSTM'             : get_baseline('LSTM',        h, num_layers),
        'GRU'              : get_baseline('GRU',         h, num_layers),
        'Transformer'      : get_baseline('Transformer', h, num_layers),
        'CNN-LSTM-Attention': CNNLSTMAttention(
            hidden_size  = h,
            num_layers   = num_layers,
            cnn_channels = 32
        ),
    }


# ============================================================
# CHẠY CHÍNH
# ============================================================
if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    all_histories = {}

    print("=" * 55)
    print("  Channel Prediction - Training All Models")
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {CONFIG['epochs']}  |  "
          f"Batch  : {CONFIG['batch_size']}  |  "
          f"LR     : {CONFIG['lr']}")
    print("=" * 55)

    for velocity in CONFIG['velocities']:
        print(f"\n{'='*55}")
        print(f"  Velocity = {velocity} km/h")
        print(f"{'='*55}")

        # Load data
        train_loader, test_loader, n_train, n_test = load_data(
            velocity,
            CONFIG['data_dir'],
            CONFIG['batch_size']
        )
        print(f"  Data loaded: train={n_train}, test={n_test}")

        # Train từng model
        for model_name, model in get_all_models(CONFIG).items():
            if model_name != 'CNN-LSTM-Attention':
                continue   
            history = train_model(
                model, train_loader, test_loader,
                CONFIG, model_name, velocity
            )
            all_histories[f"{model_name}_v{velocity}"] = history

    # Lưu toàn bộ history để vẽ đồ thị sau
    np.save('results/training_history.npy', all_histories)

    print("\n" + "=" * 55)
    print("  Training hoàn tất!")
    print(f"  Checkpoints: {CONFIG['save_dir']}/")
    print("  History    : results/training_history.npy")
    print("=" * 55)