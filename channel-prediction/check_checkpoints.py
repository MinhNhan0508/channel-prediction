"""
Kiểm tra tất cả checkpoint: hiển thị config, epoch, NMSE, timestamp.
Dùng để phát hiện file từ lần train sai (sai N_RX, FEATURE_SIZE, ...).
"""
import os
import glob
import torch
from datetime import datetime

CHECKPOINT_DIR = "results/checkpoints"
EXPECTED_FEATURE_SIZE = 256  # 2 * 64 * 2

files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")))
if not files:
    print("Không tìm thấy file .pth nào trong", CHECKPOINT_DIR)
    exit()

print(f"{'File':<45} {'Epoch':>5} {'NMSE':>8} {'FeatSize':>9} {'Time'}")
print("-" * 95)

bad_files = []
for f in files:
    name = os.path.basename(f)
    mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%m-%d %H:%M")
    try:
        ckpt = torch.load(f, map_location='cpu', weights_only=False)
        epoch   = ckpt.get('epoch', '?')
        nmse    = ckpt.get('nmse_db', float('nan'))
        cfg     = ckpt.get('config', {})
        feat    = cfg.get('feature_size', '?')
        ok      = "OK" if feat == EXPECTED_FEATURE_SIZE else "!! SAI !!"
        if feat != EXPECTED_FEATURE_SIZE:
            bad_files.append(name)
        print(f"{name:<45} {str(epoch):>5} {nmse:>7.2f}dB {str(feat):>9} {mtime}  {ok}")
    except Exception as e:
        print(f"{name:<45} ERROR: {e}")

print("-" * 95)
print(f"Tổng: {len(files)} files")
if bad_files:
    print(f"\n!! {len(bad_files)} file SAI config (feature_size != {EXPECTED_FEATURE_SIZE}):")
    for f in bad_files:
        print(f"   - {f}")
else:
    print(f"\nTất cả {len(files)} file đều đúng config (feature_size={EXPECTED_FEATURE_SIZE}).")
