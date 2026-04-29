import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.baselines import get_baseline
from models.proposed  import (
    CNNLSTMAttention,
    CNNLSTMNoAttention,
    LSTMAttention,
)


# ============================================================
# CẤU HÌNH
# ============================================================
CONFIG = {
    'epochs'         : 150,
    'batch_size'     : 64,
    'lr'             : 5e-4,
    'patience'       : 20,
    'hidden_size'    : 64,
    'num_layers'     : 2,
    'data_dir'       : 'data',
    'save_dir'       : 'results/checkpoints',
    'velocities'     : [30, 120, 300],
    'seeds'          : [42, 123, 456],
    'skip_existing'  : True,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# HÀM 1: Load data theo velocity + seed
# ============================================================
def load_data(velocity, seed, data_dir='data', batch_size=64):
    path = os.path.join(data_dir, f'channel_v{velocity}kmh_s{seed}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Chạy generate_channel.py trước!"
        )
    d = np.load(path)
    X_train = torch.FloatTensor(d['X_train'])
    y_train = torch.FloatTensor(d['y_train'])
    X_test  = torch.FloatTensor(d['X_test'])
    y_test  = torch.FloatTensor(d['y_test'])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader, len(X_train), len(X_test)


# ============================================================
# HÀM 2: Train 1 epoch có noise augmentation
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 70% có nhiễu, 30% data sạch
        if np.random.rand() < 0.7:
            snr_train    = np.random.uniform(10, 30)
            signal_power = torch.mean(X_batch ** 2).item()
            noise_std    = np.sqrt(signal_power / (10 ** (snr_train / 10)))
            X_input      = X_batch + torch.randn_like(X_batch) * noise_std
        else:
            X_input = X_batch

        optimizer.zero_grad()

        if hasattr(model, 'predict'):
            out = model.predict(X_input)
        else:
            out = model(X_input)

        loss = criterion(out, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# HÀM 3: Evaluate trên test set (data sạch)
# ============================================================
def evaluate(model, loader, criterion, device):
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
# HÀM 4: Train 1 model với early stopping
# ============================================================
def train_model(model, train_loader, test_loader,
                config, model_name, velocity, seed):

    # Lấy từ config (trước đây bị hardcode trong hàm)
    patience = config['patience']
    lr       = config['lr']

    model     = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    os.makedirs(config['save_dir'], exist_ok=True)
    save_path = os.path.join(
        config['save_dir'],
        f'{model_name}_v{velocity}kmh_s{seed}.pth'
    )

    best_val = float('inf')
    wait     = 0
    history  = {'train_loss': [], 'val_loss': []}

    print(f"\n  {model_name} | v={velocity}km/h | seed={seed} | "
          f"lr={lr} | patience={patience}")
    print(f"  {'Ep':>4} | {'Train':>10} | {'Val':>10} | {'Time':>5}")
    print("  " + "-" * 38)

    for epoch in range(1, config['epochs'] + 1):
        t0         = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE)
        val_loss   = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:>4} | {train_loss:>10.6f} | "
                  f"{val_loss:>10.6f} | {time.time()-t0:>4.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            wait     = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_loss'   : best_val,
            }, save_path)
        else:
            wait += 1

        if wait >= patience:
            print(f"  Early stop ep {epoch} | best={best_val:.6f}")
            break

    print(f"  Best val: {best_val:.6f} → {save_path}")
    return history, best_val


# ============================================================
# HÀM 5: Tạo tất cả model
# ============================================================
def get_all_models(config):
    h          = config['hidden_size']
    num_layers = config['num_layers']
    return {
        # --- Baseline AI ---
        'RNN'              : get_baseline('RNN',         h, num_layers),
        'LSTM'             : get_baseline('LSTM',        h, num_layers),
        'GRU'              : get_baseline('GRU',         h, num_layers),
        'Transformer'      : get_baseline('Transformer', h, num_layers),

        # --- Ablation variants (mới) ---
        'CNN-LSTM'         : CNNLSTMNoAttention(
            hidden_size  = h,
            num_layers   = num_layers,
            cnn_channels = 32,
        ),
        'LSTM-Attention'   : LSTMAttention(
            hidden_size  = h,
            num_layers   = num_layers,
            num_heads    = 2,
        ),

        # --- Model đề xuất ---
        'CNN-LSTM-Attention': CNNLSTMAttention(
            hidden_size  = h,
            num_layers   = num_layers,
            cnn_channels = 32,
            num_heads    = 2,
        ),
    }
    # Lưu ý: Wiener filter KHÔNG cần train (closed-form), được fit
    # trực tiếp trong evaluate.py.


# ============================================================
# CHẠY CHÍNH
# ============================================================
if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    all_histories  = {}
    best_val_table = {}

    print("=" * 60)
    print("  Channel Prediction — Multi-Seed Training")
    print(f"  Device  : {DEVICE}")
    print(f"  Seeds   : {CONFIG['seeds']}")
    print(f"  Epochs  : {CONFIG['epochs']}")
    n_models = len(get_all_models(CONFIG))
    n_runs   = n_models * len(CONFIG['velocities']) * len(CONFIG['seeds'])
    print(f"  Total   : {n_models} models × "
          f"{len(CONFIG['velocities'])} velocities × "
          f"{len(CONFIG['seeds'])} seeds = {n_runs} runs")
    print("=" * 60)

    for seed in CONFIG['seeds']:
        print(f"\n{'='*60}")
        print(f"  SEED = {seed}")
        print(f"{'='*60}")

        for velocity in CONFIG['velocities']:
            print(f"\n  --- v = {velocity} km/h ---")
            train_loader, test_loader, n_train, n_test = load_data(
                velocity, seed,
                CONFIG['data_dir'], CONFIG['batch_size']
            )
            print(f"  Data: train={n_train}, test={n_test}")

            for model_name, model in get_all_models(CONFIG).items():
                ckpt_path = os.path.join(
                    CONFIG['save_dir'],
                    f'{model_name}_v{velocity}kmh_s{seed}.pth'
                )

                # Skip nếu đã có checkpoint và skip_existing=True
                if CONFIG['skip_existing'] and os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location='cpu',
                                      weights_only=True)
                    best_val = float(ckpt.get('val_loss', float('nan')))
                    print(f"  [SKIP] {model_name} | v={velocity} | s={seed} | "
                          f"val={best_val:.6f}  (đã có checkpoint)")
                    history = {'train_loss': [], 'val_loss': [best_val]}
                else:
                    history, best_val = train_model(
                        model, train_loader, test_loader,
                        CONFIG, model_name, velocity, seed
                    )

                key = f"{model_name}_v{velocity}_s{seed}"
                all_histories[key] = history

                if model_name not in best_val_table:
                    best_val_table[model_name] = {}
                if velocity not in best_val_table[model_name]:
                    best_val_table[model_name][velocity] = []
                best_val_table[model_name][velocity].append(best_val)

    # In bảng tổng hợp
    print("\n" + "=" * 65)
    print("  TỔNG HỢP — Mean Best Val Loss qua 3 seeds")
    print("=" * 65)
    print(f"  {'Model':<22} | {'30 km/h':>14} | "
          f"{'120 km/h':>14} | {'300 km/h':>14}")
    print("  " + "-" * 67)

    for name in get_all_models(CONFIG).keys():
        row = f"  {name:<22} |"
        for v in [30, 120, 300]:
            vals = best_val_table[name][v]
            mean = np.mean(vals)
            std  = np.std(vals)
            row += f" {mean:.2e}±{std:.1e} |"
        print(row)
    print("=" * 65)

    np.save('results/training_history_multiseed.npy', all_histories)
    np.save('results/best_val_table.npy', best_val_table)

    print("\nXong!")
    print(f"Checkpoints: {CONFIG['save_dir']}/")