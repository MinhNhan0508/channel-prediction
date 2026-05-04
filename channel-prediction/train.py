import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.baselines import get_baseline
from models.proposed  import CNNLSTMAttention

# ============================================================
# CẤU HÌNH
# ============================================================
FEATURE_SIZE = 256   # 2 × N_TX × N_RX = 2 × 64 × 2

CONFIG = {
    'epochs'      : 150,
    'batch_size'  : 64,
    'lr'          : 5e-4,
    'patience'    : 20,
    'hidden_size' : 128,   # tăng so với SISO (64 → 128)
    'num_layers'  : 2,
    'data_dir'    : 'data',
    'save_dir'    : 'results/checkpoints',
    'velocities'  : [30, 120, 300],
    'seeds'       : [42, 123, 456],
    'feature_size': FEATURE_SIZE,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# HÀM 1: Load data MIMO
# ============================================================
def load_data(velocity, seed, data_dir='data', batch_size=64):
    """
    Load dataset Massive MIMO từ file .npz.
    Tên file: mimo_v{velocity}kmh_s{seed}.npz
    """
    path = os.path.join(
        data_dir, f'mimo_v{velocity}kmh_s{seed}.npz'
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy: {path}\n"
            f"→ Chạy generate_data.py trước!"
        )

    d = np.load(path)
    X_train = torch.FloatTensor(d['X_train'])
    y_train = torch.FloatTensor(d['y_train'])
    X_test  = torch.FloatTensor(d['X_test'])
    y_test  = torch.FloatTensor(d['y_test'])

    print(f"  MIMO data: X={X_train.shape} "
          f"(batch × seq × {FEATURE_SIZE} features)")

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
# HÀM 2: Tính NMSE (dB)
# ============================================================
def compute_nmse_db(pred, target):
    """
    NMSE cho Massive MIMO:
    Tính trên toàn bộ vector kênh 128 chiều (64 anten × 2).
    """
    err  = pred - target
    nmse = np.sum(err**2) / (np.sum(target**2) + 1e-10)
    return 10 * np.log10(nmse + 1e-10)


# ============================================================
# HÀM 3: Train 1 epoch (có noise augmentation)
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Noise augmentation: 70% có nhiễu → model học robust hơn
        if np.random.rand() < 0.7:
            snr      = np.random.uniform(10, 30)
            sig_pow  = torch.mean(X_batch**2).item()
            noise_std = np.sqrt(sig_pow / (10**(snr/10)))
            X_input   = X_batch + torch.randn_like(X_batch) * noise_std
        else:
            X_input = X_batch

        optimizer.zero_grad()

        out = model.predict(X_input) \
              if hasattr(model, 'predict') \
              else model(X_input)

        loss = criterion(out, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# HÀM 4: Evaluate trên test set
# ============================================================
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            out = model.predict(X_batch) \
                  if hasattr(model, 'predict') \
                  else model(X_batch)
            total_loss += criterion(out, y_batch.to(device)).item()
            all_preds.append(out.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    nmse_db = compute_nmse_db(preds, targets)

    return total_loss / len(loader), nmse_db


# ============================================================
# HÀM 5: Train 1 model hoàn chỉnh
# ============================================================
def train_model(model, train_loader, test_loader,
                config, model_name, velocity, seed):

    model     = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    os.makedirs(config['save_dir'], exist_ok=True)
    save_path = os.path.join(
        config['save_dir'],
        f'{model_name}_v{velocity}kmh_s{seed}.pth'
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"\n  {model_name} | v={velocity}km/h | seed={seed} | "
          f"params={params:,}")
    print(f"  {'Ep':>4} | {'Train':>10} | {'Val':>10} | "
          f"{'NMSE':>9} | {'Time':>5}")
    print("  " + "-" * 50)

    best_val, wait = float('inf'), 0
    history = {'train_loss': [], 'val_loss': [], 'nmse_db': []}

    for epoch in range(1, config['epochs'] + 1):
        t0         = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE)
        val_loss, nmse_db = evaluate(
            model, test_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['nmse_db'].append(nmse_db)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:>4} | {train_loss:>10.6f} | "
                  f"{val_loss:>10.6f} | {nmse_db:>7.2f}dB | "
                  f"{time.time()-t0:>4.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            wait     = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_loss'   : best_val,
                'nmse_db'    : nmse_db,
                'config'     : config,
            }, save_path)
        else:
            wait += 1
            if wait >= config['patience']:
                print(f"  Early stop ep {epoch} | "
                      f"best_val={best_val:.6f} | "
                      f"NMSE={history['nmse_db'][np.argmin(history['val_loss'])]:.2f}dB")
                break

    print(f"  Saved: {save_path}")
    return history, best_val


# ============================================================
# HÀM 6: Tạo tất cả model MIMO
# ============================================================
def get_all_models(config):
    h   = config['hidden_size']
    nl  = config['num_layers']
    fs  = config['feature_size']
    return {
        'RNN'              : get_baseline('RNN',         h, nl, fs),
        'LSTM'             : get_baseline('LSTM',        h, nl, fs),
        'GRU'              : get_baseline('GRU',         h, nl, fs),
        'Transformer'      : get_baseline('Transformer', h, nl, fs),
        'CNN-LSTM-Attention': CNNLSTMAttention(
            input_size   = fs,
            cnn_channels = 64,
            hidden_size  = 256,
            num_layers   = nl,
            num_heads    = 8,
            output_size  = fs,
        ),
    }


# ============================================================
# CHẠY CHÍNH
# ============================================================
if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)

    total_runs = (len(CONFIG['velocities']) *
                  len(CONFIG['seeds']) * 5)

    print("=" * 65)
    print("  Massive MIMO Channel Prediction — Training")
    print(f"  Device      : {DEVICE}")
    print(f"  Feature size: {FEATURE_SIZE} (N_TX=64 × 2)")
    print(f"  Seeds       : {CONFIG['seeds']}")
    print(f"  Total runs  : {total_runs} "
          f"(5 models × {len(CONFIG['velocities'])} speeds × "
          f"{len(CONFIG['seeds'])} seeds)")
    print("=" * 65)

    all_results = {}

    for seed in CONFIG['seeds']:
        print(f"\n{'='*65}")
        print(f"  SEED = {seed}")
        print(f"{'='*65}")

        for velocity in CONFIG['velocities']:
            print(f"\n  ── v = {velocity} km/h ──")

            train_loader, test_loader, n_tr, n_te = load_data(
                velocity, seed,
                CONFIG['data_dir'], CONFIG['batch_size']
            )
            print(f"  Data: train={n_tr}, test={n_te}")

            for model_name, model in get_all_models(CONFIG).items():
                save_path = os.path.join(
                    CONFIG['save_dir'],
                    f'{model_name}_v{velocity}kmh_s{seed}.pth'
                )
                if os.path.exists(save_path):
                    print(f"  Skipping {model_name} v={velocity} s={seed} — đã có checkpoint")
                    continue

                history, best_val = train_model(
                    model, train_loader, test_loader,
                    CONFIG, model_name, velocity, seed
                )
                key = f"{model_name}_v{velocity}_s{seed}"
                all_results[key] = {
                    'history' : history,
                    'best_val': best_val
                }

    np.save('results/training_results_mimo.npy', all_results)

    print("\n" + "=" * 65)
    print("  Training hoàn tất!")
    print(f"  Checkpoints: {CONFIG['save_dir']}/")
    print("=" * 65)