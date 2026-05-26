# PLAN.md — Channel Prediction Project (Spatial-Temporal Mamba)

> **MỤC ĐÍCH FILE NÀY:** Đây là kế hoạch chi tiết và đặc tả kỹ thuật đầy đủ cho dự án đồ án tốt nghiệp về Channel Prediction. File này được viết để Claude Code (hoặc bất kỳ AI coding assistant nào) có thể đọc và **tự triển khai toàn bộ codebase** mà không cần hỏi lại. Mọi quyết định kiến trúc, hyperparameter, file structure, và code skeleton đều được ghi rõ. KHÔNG được tự ý đổi kiến trúc, đổi hyperparameter, hay đổi đề tài. Nếu có vấn đề → dừng và hỏi người dùng.

---

## 0. METADATA

| Trường | Giá trị |
|---|---|
| **Tên đề tài (CỐ ĐỊNH)** | Dự đoán Kênh truyền (Channel Prediction) trong môi trường di động tốc độ cao sử dụng mạng học sâu |
| **Người làm** | Minh Nhân — HCMUTE, năm cuối CNKT Máy tính |
| **Ngôn ngữ** | Python 3.10 + PyTorch |
| **Phần cứng** | GTX 1650 4GB VRAM, 16GB RAM, Windows 11 |
| **Thời gian** | 3 tuần |
| **Tên model đề xuất** | **Spatial-Temporal Mamba (ST-Mamba)** |
| **Đóng góp khoa học** | (1) First Mamba application to channel prediction, (2) Spatial-temporal decoupling, (3) Physics-informed multi-component loss |
| **Bài báo tham chiếu chính** | Li et al., "Channel Prediction via Hybrid-Attention Parallel Synchronous Transformer", IEICE Trans. Commun., 2026 |

---

## 1. NGUYÊN TẮC LÀM VIỆC CHO CLAUDE CODE

**Đọc kỹ phần này trước khi viết bất kỳ dòng code nào.**

### 1.1 Quy tắc tuyệt đối — KHÔNG được vi phạm

1. **KHÔNG đổi đề tài.** Tên đề tài đã chốt. Không gợi ý "đổi sang khảo sát các model".
2. **KHÔNG đổi kiến trúc đề xuất.** Model đề xuất LÀ Spatial-Temporal Mamba. Nếu kết quả không tốt → debug, không đổi sang kiến trúc khác.
3. **KHÔNG đổi hyperparameter khi chưa hỏi.** Tất cả hyperparameter trong file này đã được suy nghĩ kỹ. Nếu muốn đổi → dừng và hỏi.
4. **Fair comparison là BẮT BUỘC.** Tất cả baseline và proposed model phải có cùng parameter budget ±10%. Khi viết code phải verify số params và log ra.
5. **Reproducibility:** Mọi random operation phải set seed. Mọi config phải save vào file YAML.
6. **KHÔNG copy-paste từ bài báo IEICE.** Đây là đồ án có đóng góp riêng, không phải bản sao.
7. **CHECKPOINT VALIDATION:** Trước khi training full, phải overfit được 1 batch nhỏ (sanity check). Nếu không overfit được → có bug.
8. **KHÔNG TỰ XÓA FILE CỦA USER.** Các file cũ chỉ được MOVE vào `legacy/`, không bao giờ `rm`/`Remove-Item`. Người dùng sẽ tự xóa `legacy/` sau khi verify project mới chạy được. Xem mục 4.5 để biết quy trình migration chi tiết.

### 1.2 Quy tắc khi gặp vấn đề

- **Nếu code có lỗi runtime:** dừng, in stack trace, KHÔNG sửa workaround tạm thời.
- **Nếu hyperparameter trong file không phù hợp với phần cứng (OOM, etc.):** dừng và hỏi, không tự giảm batch size hay đổi gì.
- **Nếu kết quả NMSE rất tệ (>0 dB) sau khi train xong:** dừng, debug data pipeline trước.
- **Nếu Mamba install không được:** dừng, hỏi người dùng có muốn dùng `mamba.py` (pure PyTorch implementation) thay thế không.

### 1.3 Định nghĩa "Hoàn thành" cho mỗi file code

Một file code chỉ được coi là "hoàn thành" khi:
1. Chạy được không lỗi runtime
2. Có docstring đầy đủ
3. Có type hints
4. Có test cơ bản (ở `tests/` hoặc inline `if __name__ == "__main__":`)
5. Số params được verify và log

---

## 2. TỔNG QUAN ĐỀ TÀI

### 2.1 Bài toán

**Channel Prediction (Dự đoán kênh truyền):** Cho biết trạng thái kênh truyền MIMO ở P khung thời gian quá khứ `H(t-P+1), ..., H(t)`, dự đoán trạng thái kênh ở Q khung tương lai `H(t+1), ..., H(t+Q)`.

**Tại sao cần:** Trong massive MIMO 5G/6G, base station chỉ ước lượng CSI một lần mỗi frame. Trong môi trường di động tốc độ cao (user di chuyển nhanh), kênh thay đổi nhanh hơn tần suất ước lượng → CSI "outdated" → giảm sum-rate. Dự đoán kênh trước khi nó được ước lượng lại sẽ giúp duy trì chất lượng truyền tin.

### 2.2 Setup hệ thống (CỐ ĐỊNH theo IEICE 2026 paper)

| Tham số | Ký hiệu | Giá trị | Ý nghĩa |
|---|---|---|---|
| Số anten base station | N_BS | **64** | Massive MIMO |
| Số RF chain | N_RF | **8** | Hybrid beamforming, giảm complexity 8× |
| Số anten user | N_M | **4** | Mobile user equipment |
| Tần số carrier | f | **30 GHz** | mmWave |
| Sampling time | T_s | **0.625 ms** | Theo 3GPP standard |
| Channel model | — | **3GPP CDL-A** | NLOS, multi-path |
| Frames quá khứ | P | **30** | Input window |
| Frames dự đoán | Q | **5** | Output horizon |
| Start token length | L | **10** | Decoder init (cho seq2seq) |
| Pilot length | G | **4** | = N_M |
| User velocity range (train) | v | **U[30, 50] km/h** | Random per sample |
| Delay spread | — | **U[25, 250] ns** | Random per sample |
| SNR range (train) | — | **U[10, 20] dB** | Random per sample |
| Test velocity | v_test | **30, 50 km/h** | Fixed for evaluation |
| Test delay spread | — | **50 ns** | Fixed |
| Test SNR | — | **{10, 12, 14, 16, 18, 20} dB** | Curve points |

**Lưu ý quan trọng về dimension sau RF reduction:**
- Channel ma trận đầy đủ: `H ∈ ℂ^(N_BS × N_M) = ℂ^(64 × 4)` = 256 complex = 512 real
- Sau analog combiner D ∈ ℂ^(N_RF × N_BS): `He = D·H ∈ ℂ^(N_RF × N_M) = ℂ^(8 × 4)` = **32 complex = 64 real**
- **Đây là dimension thực sự đi vào model** → GTX 1650 gánh được

### 2.3 Đầu vào — Đầu ra của model

```
INPUT:  effective_channels_history ∈ ℝ^(P × 64)
        = [vec(Re(He(t-P+1))), vec(Im(He(t-P+1))); ...; vec(Re(He(t))), vec(Im(He(t)))]
        Shape: (batch, 30, 64)

OUTPUT: predicted_channels ∈ ℝ^(Q × 64)
        Shape: (batch, 5, 64)
```

---

## 3. ĐÓNG GÓP KHOA HỌC (cho hội đồng bảo vệ)

### Đóng góp 1: Ứng dụng đầu tiên của Mamba (Selective State Space Model) vào dự đoán kênh truyền không dây

**Tuyên bố:** Tính đến tháng 5/2026, chưa có công trình nào ứng dụng Mamba (Gu & Dao, 2024) hoặc Selective SSM vào bài toán channel prediction trong wireless. Đây là khoảng trống nghiên cứu rõ ràng.

**Lý lẽ kỹ thuật:**
- Channel dynamics có cấu trúc Markov (h(t+1) phụ thuộc h(t)) — phù hợp với recurrent structure của SSM
- Mamba có linear complexity O(n) thay vì O(n²) của Transformer
- Selective mechanism (S6) cho phép model "chú ý" có chọn lọc đến các time step quan trọng — tương tự attention nhưng efficient hơn
- Tham số ít hơn Transformer ~43% với cùng performance

### Đóng góp 2: Kiến trúc Spatial-Temporal Decoupling cho channel prediction

**Tuyên bố:** Khác với các phương pháp hiện tại (bao gồm IEICE 2026) — flatten channel thành vector rồi đưa vào model — chúng tôi tách rời xử lý spatial (anten array) và temporal (Doppler dynamics).

**Pipeline:**
```
Channel He ∈ ℝ^(N_RF × N_M × 2)  [spatial × complex]
    ↓ Spatial Encoder (CNN 2D)
spatial features ∈ ℝ^d_spatial
    ↓ Temporal Predictor (Mamba)
predicted spatial features ∈ ℝ^(Q × d_spatial)
    ↓ Spatial Decoder (Transposed CNN)
Predicted He ∈ ℝ^(N_RF × N_M × 2)
```

**Lý lẽ:** Anten kề nhau có channel tương quan (theo cấu trúc Kronecker). 2D CNN khai thác được tương quan này, flatten thì không.

### Đóng góp 3: Physics-Informed Multi-Component Loss

**Tuyên bố:** Khác với loss NMSE thuần (dùng trong tất cả paper baseline), chúng tôi đề xuất loss đa thành phần dựa trên vật lý kênh.

**Công thức:**
```
L_total = α · L_NMSE + β · L_phase + γ · L_temporal_consistency

L_NMSE = E[||Ĥ - H||²] / E[||H||²]
L_phase = E[|∠Ĥ - ∠H|]  (chuẩn hoá)
L_temporal_consistency = E[|||Ĥ(t+1) - Ĥ(t)|| - ||H(t+1) - H(t)|||]
```

**Lý lẽ:**
- `L_phase`: pha có ý nghĩa cho beamforming (precoder dùng phase để align beam)
- `L_temporal_consistency`: ép model học đúng rate of change của kênh (Doppler physics)

---

## 4. CẤU TRÚC THƯ MỤC

```
channel-prediction/
├── README.md                    # Hướng dẫn chạy
├── PLAN.md                       # File này
├── requirements.txt              # pip deps
├── environment.yml               # conda env
├── .gitignore
│
├── legacy/                       # Files cũ của project (được migrate qua script 00_migrate_legacy.py)
│   │                             # KHÔNG XÓA tự động. User sẽ tự xóa sau khi verify project mới hoạt động.
│   ├── evaluate.py
│   ├── train.py
│   ├── ... (xem mục 4.5 để biết chi tiết)
│   ├── data/                     # 9 file .npz cũ (format Rayleigh+Jakes 16×16)
│   └── results/
│       └── checkpoints/          # 63 file .pth cũ (không dùng được với code mới)
│
├── configs/                      # YAML configs (HYDRA-style)
│   ├── default.yaml              # Default hyperparameters
│   ├── data/
│   │   └── cdl_a.yaml            # CDL-A dataset config
│   ├── model/
│   │   ├── st_mamba.yaml         # Proposed model config
│   │   ├── rnn.yaml              # Baseline
│   │   ├── lstm.yaml             # Baseline
│   │   ├── gru.yaml              # Baseline
│   │   └── transformer.yaml      # Baseline (IEICE-style)
│   └── train/
│       ├── full.yaml             # Full training (100 epochs)
│       ├── curriculum.yaml       # Curriculum training stages
│       └── debug.yaml            # 1-batch overfit test
│
├── data/                         # Generated channel datasets
│   ├── train/                    # 8000 samples
│   ├── val/                      # 1000 samples
│   ├── test/                     # 1000 samples
│   └── README.md                 # Dataset description
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── channel_generator.py  # CDL-A channel generation
│   │   ├── dataset.py            # PyTorch Dataset class
│   │   └── normalize.py          # Normalization utilities
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py               # BaseModel interface
│   │   ├── st_mamba.py           # ⭐ Proposed: Spatial-Temporal Mamba
│   │   ├── spatial_encoder.py    # Reusable 2D CNN spatial encoder
│   │   ├── spatial_decoder.py    # Reusable 2D CNN spatial decoder
│   │   ├── mamba_block.py        # Mamba implementation wrapper
│   │   ├── baselines/
│   │   │   ├── __init__.py
│   │   │   ├── rnn.py
│   │   │   ├── lstm.py
│   │   │   ├── gru.py
│   │   │   └── transformer.py    # IEICE-style parallel synchronous
│   │   └── utils.py              # Param counting, model info
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── nmse.py               # NMSE loss
│   │   ├── phase.py              # Phase loss
│   │   ├── temporal.py           # Temporal consistency loss
│   │   └── combined.py           # Multi-component loss
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training loop
│   │   ├── curriculum.py         # Curriculum learning scheduler
│   │   ├── callbacks.py          # Checkpoint, early stop, lr scheduler
│   │   └── optimizer.py          # AdamW + warmup
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # NMSE, sum-rate, BER
│   │   ├── ber.py                # BER simulation (ZF precoder)
│   │   ├── sum_rate.py           # Sum-rate computation
│   │   └── evaluator.py          # End-to-end evaluation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # YAML loader
│       ├── seed.py               # Reproducibility
│       ├── logger.py             # TensorBoard + file logger
│       └── visualize.py          # Plot utilities
│
├── scripts/                      # Entry points
│   ├── 01_generate_data.py       # Generate dataset
│   ├── 02_train.py               # Train one model
│   ├── 03_evaluate.py            # Evaluate one model
│   ├── 04_compare_all.py         # Compare all models (final plots)
│   ├── 05_ablation.py            # Run ablation experiments
│   └── 06_sanity_check.py        # Overfit 1 batch
│
├── results/                      # Outputs
│   ├── checkpoints/              # Model weights (.pth)
│   │   ├── st_mamba/
│   │   ├── rnn/
│   │   ├── lstm/
│   │   ├── gru/
│   │   └── transformer/
│   ├── logs/                     # Training logs (TensorBoard)
│   ├── figures/                  # Final plots for thesis
│   │   ├── nmse_vs_snr.png
│   │   ├── nmse_vs_frame.png
│   │   ├── sum_rate.png
│   │   ├── ablation.png
│   │   └── complexity.png
│   └── tables/                   # CSV tables
│       ├── nmse_summary.csv
│       ├── sum_rate_summary.csv
│       └── complexity.csv
│
├── notebooks/                    # Exploratory analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_debugging.ipynb
│   └── 03_results_analysis.ipynb
│
├── tests/                        # Unit tests
│   ├── test_channel_generator.py
│   ├── test_dataset.py
│   ├── test_st_mamba.py
│   ├── test_baselines.py
│   ├── test_losses.py
│   └── test_metrics.py
│
└── app.py                        # Gradio demo (giữ lại từ project cũ)
```

---

### 4.5 MIGRATION: Xử lý các file cũ của project (BƯỚC ĐẦU TIÊN bắt buộc)

**Bối cảnh:** Người dùng có project cũ với cấu trúc khác hoàn toàn. Trước khi viết code mới, Claude Code PHẢI di chuyển các file cũ vào folder `legacy/` để không xung đột.

**QUY TẮC TUYỆT ĐỐI:**
- **KHÔNG XÓA** bất kỳ file nào của người dùng. Chỉ MOVE vào `legacy/`.
- **KHÔNG CHẠM** vào folder `.git/`, `.gitignore`, `README.md`, `requirements.txt` cũ — giữ nguyên vị trí.
- **KHÔNG MOVE** folder `venv/` hoặc `__pycache__/` — chỉ ignore.
- **Người dùng sẽ tự xóa `legacy/`** sau khi verify project mới chạy đúng. Claude Code KHÔNG được tự xóa.

#### 4.5.1 Bảng phân loại file cũ

| File/Folder cũ | Hành động | Lý do |
|---|---|---|
| `evaluate.py` | MOVE → `legacy/evaluate.py` | Sẽ viết lại trong `scripts/03_evaluate.py` |
| `train.py` | MOVE → `legacy/train.py` | Sẽ viết lại trong `scripts/02_train.py` |
| `channel_dataset.py` | MOVE → `legacy/channel_dataset.py` | Sẽ viết lại trong `src/data/dataset.py` |
| `debug_nmse.py` | MOVE → `legacy/debug_nmse.py` | Không dùng nữa, giữ làm tham khảo |
| `generate_channel.py` | MOVE → `legacy/generate_channel.py` | Sẽ viết lại trong `src/data/channel_generator.py` (CDL-A khác Rayleigh+Jakes cũ) |
| `models/baselines.py` | MOVE → `legacy/models/baselines.py` | Sẽ viết lại trong `src/models/baselines/` (mỗi model 1 file) |
| `models/classical.py` | MOVE → `legacy/models/classical.py` | Wiener không còn baseline trong PLAN mới |
| `models/proposed.py` | MOVE → `legacy/models/proposed.py` | Thay thế bằng `src/models/st_mamba.py` |
| `data/*.npz` (9 file) | MOVE → `legacy/data/*.npz` | Format cũ (Rayleigh+Jakes 16×16) không khớp format mới (CDL-A 64×4) |
| `results/checkpoints/*.pth` (63 file) | MOVE → `legacy/results/checkpoints/*.pth` | Checkpoint cũ train trên data cũ, không dùng được |
| `results/*.png`, `*.csv`, `*.npy` | MOVE → `legacy/results/` | Kết quả cũ |
| `HUONG_DAN.md` | MOVE → `legacy/HUONG_DAN.md` | Có thể đã outdated |
| `CONTEXT.md` | MOVE → `legacy/CONTEXT.md` | Context cũ, PLAN.md là context mới |
| `app.py` | **GIỮ NGUYÊN** | Sẽ update trong Tuần 3 để dùng model mới |
| `.git/`, `.gitignore` | **GIỮ NGUYÊN** | Version control |
| `requirements.txt` | **GIỮ NGUYÊN** | Sẽ append dependencies mới vào |
| `venv/` | **GIỮ NGUYÊN** | Virtual environment |
| `__pycache__/` (bất kỳ đâu) | Skip, không move | Cache, sẽ tự tái tạo |
| `.mcp.json`, `certificate.pem` | **GIỮ NGUYÊN** | Config khác, không liên quan |

#### 4.5.2 Script migration (CHẠY ĐẦU TIÊN trước khi viết code mới)

Claude Code phải tạo và chạy script này TRƯỚC KHI viết bất kỳ file code mới nào:

**File:** `scripts/00_migrate_legacy.py`

```python
"""
Migrate old project files to legacy/ folder.

Run this ONCE before implementing new code.

Rules:
  - MOVE (not delete) old files into legacy/
  - SKIP venv/, .git/, __pycache__/
  - KEEP app.py, requirements.txt, .gitignore in place
  - User will manually delete legacy/ after verifying new project works
"""
import shutil
from pathlib import Path

# Files/folders to MOVE into legacy/
MOVE_TO_LEGACY = [
    # Python scripts
    "evaluate.py",
    "train.py",
    "channel_dataset.py",
    "debug_nmse.py",
    "generate_channel.py",

    # Old models folder (entire)
    "models",

    # Old data folder (entire — different format, regenerate)
    "data",

    # Old results (checkpoints, plots, csv)
    "results",

    # Old documentation
    "HUONG_DAN.md",
    "CONTEXT.md",
]

# Files/folders to KEEP in place (do NOT touch)
KEEP_IN_PLACE = [
    "app.py",
    "requirements.txt",
    ".gitignore",
    ".git",
    "venv",
    ".mcp.json",
    "certificate.pem",
    "README.md",
    "PLAN.md",
    "legacy",            # itself
    "src",               # new code will go here
    "scripts",           # new code will go here
    "configs",           # new code will go here
    "tests",             # new code will go here
    "notebooks",         # new code will go here
]

# Patterns to SKIP entirely
SKIP_PATTERNS = ["__pycache__", ".pytest_cache", ".ipynb_checkpoints"]


def main():
    root = Path.cwd()
    legacy = root / "legacy"
    legacy.mkdir(exist_ok=True)
    print(f"Migration root: {root}")
    print(f"Legacy folder:  {legacy}")
    print()

    moved = []
    skipped = []
    not_found = []

    for name in MOVE_TO_LEGACY:
        src = root / name
        if not src.exists():
            not_found.append(name)
            continue

        # Skip if user has __pycache__ inside, etc — still move whole folder, OK
        dst = legacy / name
        if dst.exists():
            # Avoid overwriting existing legacy
            print(f"[SKIP] {name} already in legacy/, leaving original alone")
            skipped.append(name)
            continue

        print(f"[MOVE] {src.name} -> legacy/{name}")
        shutil.move(str(src), str(dst))
        moved.append(name)

    print()
    print(f"Moved:     {len(moved)} items")
    print(f"Skipped:   {len(skipped)} items (already in legacy)")
    print(f"Not found: {len(not_found)} items (nothing to migrate)")
    if not_found:
        print(f"  Not found list: {not_found}")
    print()
    print("Migration complete. The following are PRESERVED in project root:")
    for name in KEEP_IN_PLACE:
        if (root / name).exists():
            print(f"  - {name}")
    print()
    print("Next steps:")
    print("  1. Verify legacy/ has all old files")
    print("  2. Verify app.py, requirements.txt still in root")
    print("  3. Begin implementing new code per PLAN.md")
    print("  4. AFTER project new is verified, user can delete legacy/ manually")


if __name__ == "__main__":
    main()
```

#### 4.5.3 Quy trình migration

Claude Code phải chạy chính xác theo thứ tự:

```
Bước 1: Tạo file scripts/00_migrate_legacy.py (theo code trên)
Bước 2: cd vào root project (D:\...\channel-prediction\)
Bước 3: Chạy: python scripts/00_migrate_legacy.py
Bước 4: Verify console output: tất cả file cũ đã vào legacy/
Bước 5: Verify thư mục root chỉ còn: app.py, requirements.txt, .git/, venv/, legacy/, PLAN.md
Bước 6: Tạo cấu trúc thư mục mới (configs/, src/, scripts/, tests/, ...)
Bước 7: Bắt đầu implement theo PLAN.md
```

#### 4.5.4 Khi nào người dùng xóa `legacy/`?

KHÔNG phải Claude Code xóa. Người dùng tự xóa khi:
- Project mới đã chạy được end-to-end (sanity check pass, ít nhất 1 model train xong, evaluation chạy được)
- Đã commit toàn bộ code mới lên git
- Không còn cần tham khảo file cũ nào

Khi đó người dùng sẽ chạy: `Remove-Item legacy -Recurse -Force` (PowerShell)

---



### 4.1 MIGRATION TỪ PROJECT CŨ — Xóa file dư

**Trước khi implement, Claude Code PHẢI:**

1. **List ra tất cả file/folder hiện có** trong thư mục project (dùng `ls -la` hoặc tương đương)
2. **So sánh với cấu trúc mục 4** ở trên
3. **HỎI người dùng CONFIRM** danh sách file sẽ xóa trước khi xóa thật
4. **Sau khi confirm**, xóa các file cũ

**File/folder CỤ THỂ cần xóa (project cũ của Minh Nhân):**

| Đường dẫn | Hành động | Lý do |
|---|---|---|
| `app.py` | **GIỮ LẠI** | Gradio demo, sẽ update sau |
| `evaluate.py` | **XÓA** | File cũ ở root, code mới ở `scripts/03_evaluate.py` |
| `train.py` | **XÓA** | File cũ ở root, code mới ở `scripts/02_train.py` |
| `channel_dataset.py` | **XÓA** | File cũ ở root, code mới ở `src/data/dataset.py` |
| `debug_nmse.py` | **XÓA** | Sanity check mới ở `scripts/06_sanity_check.py` |
| `HUONG_DAN.md` | **XÓA** | File hướng dẫn cũ, README.md mới sẽ thay |
| `CONTEXT.md` | **XÓA** | File context cũ |
| `models/` folder | **XÓA TOÀN BỘ** | Cấu trúc mới ở `src/models/` |
| `data/*.npz` files | **XÓA** | Sẽ generate lại với CDL-A spec mới (data cũ là Rayleigh+Jakes) |
| `data/` folder | **GIỮ LẠI folder** | Folder sẽ chứa data mới |
| `results/checkpoints/*` | **XÓA TOÀN BỘ** | Checkpoint cũ train với kiến trúc khác, không dùng được |
| `results/` folder | **GIỮ LẠI folder** | Sẽ chứa output mới |
| `.gradio/` | **GIỮ LẠI** | Cache của Gradio |
| `__pycache__/` các nơi | **XÓA** | Cache Python |
| `venv/` | **GIỮ LẠI** | Virtual environment |
| `.gitignore`, `requirements.txt` | **GIỮ + UPDATE** | Update theo dependency mới ở mục 15 |
| `.mcp.json` | **GIỮ LẠI** | Config MCP |

**Quy trình xóa AN TOÀN:**

```bash
# Bước 1: Backup trước khi xóa (tạo branch git)
cd D:\Tailieutrentruong\Doantotnghiep\FinalProject\channel-prediction
git checkout -b before-migration
git add -A
git commit -m "Backup before migration to ST-Mamba architecture"
git push origin before-migration

# Bước 2: Xóa file cũ (sau khi user confirm)
rm evaluate.py train.py channel_dataset.py debug_nmse.py HUONG_DAN.md CONTEXT.md
rm -rf models/ __pycache__/
rm -f data/*.npz
rm -rf results/checkpoints/*

# Bước 3: Tạo cấu trúc mới
mkdir -p src/data src/models/baselines src/losses src/training src/evaluation src/utils
mkdir -p configs/data configs/model configs/train
mkdir -p data/train data/val data/test_v30 data/test_v50
mkdir -p results/checkpoints results/logs results/figures results/tables
mkdir -p scripts notebooks tests

# Bước 4: Tạo file __init__.py rỗng cho package Python
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/models/baselines/__init__.py
touch src/losses/__init__.py
touch src/training/__init__.py
touch src/evaluation/__init__.py
touch src/utils/__init__.py
```

**Lưu ý:** Trên Windows PowerShell các lệnh là:
```powershell
Remove-Item -Path evaluate.py, train.py, channel_dataset.py, debug_nmse.py, HUONG_DAN.md, CONTEXT.md
Remove-Item -Path models, __pycache__ -Recurse -Force
Remove-Item -Path data\*.npz
Remove-Item -Path results\checkpoints\* -Recurse -Force
New-Item -ItemType Directory -Force -Path src\data, src\models\baselines, src\losses, src\training, src\evaluation, src\utils
New-Item -ItemType Directory -Force -Path configs\data, configs\model, configs\train
New-Item -ItemType Directory -Force -Path data\train, data\val, data\test_v30, data\test_v50
New-Item -ItemType Directory -Force -Path results\checkpoints, results\logs, results\figures, results\tables
New-Item -ItemType Directory -Force -Path scripts, notebooks, tests
```

**QUAN TRỌNG:** KHÔNG xóa gì cho đến khi user confirm. Nếu user không confirm hoặc muốn giữ file nào → tôn trọng. KHÔNG bao giờ xóa folder `venv/`, `.git/`, `.gradio/` mà không hỏi.

---

## 5. DATASET — Đặc tả CHI TIẾT

### 5.1 Channel Model: 3GPP CDL-A

Theo bài báo IEICE 2026:

```
H(t) = Σ_{l=1}^{L} ξ_l · a_BS(φ_BS,l) · a_user^H(φ_user,l) · exp(-j2π f_l t T_s)
```

Trong đó:
- `L`: số path (CDL-A có **23 path** theo 3GPP TR 38.901)
- `ξ_l`: complex gain của path l, theo CDL-A power profile
- `f_l = v/λ · cos(θ_l)`: Doppler shift của path l
- `a_BS(φ)`, `a_user(φ)`: array response vector (ULA)
- `φ_BS,l ~ Laplacian PAS` với mean `μ`, spread `σ` (theo CDL-A)

### 5.2 Tham số CDL-A (chuẩn 3GPP TR 38.901)

CDL-A là profile NLOS với 23 cluster. Bảng power-delay-angle profile:

```python
# Mỗi row: (delay_ns_normalized, power_dB, AoD_deg, AoA_deg, ZoD_deg, ZoA_deg)
CDL_A_PROFILE = [
    (0.0000, -13.4, -178.1,  51.3,  50.2, 125.4),
    (0.3819, -15.2, -178.1,  51.3,  50.2, 125.4),
    (0.4025,  -7.5, -158.6,  -89.4, 93.2, 117.5),
    # ... 23 clusters total
    # Use full table from 3GPP TR 38.901 Table 7.7.1-1
]
```

**Implementation note:** Module `src/data/channel_generator.py` phải implement đầy đủ 23 cluster theo bảng 3GPP TR 38.901 Table 7.7.1-1. Nếu không có table sẵn → hardcode từ link: https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/

### 5.3 Quy trình generate 1 sample

```python
def generate_one_sample(velocity_kmh, delay_spread_ns, snr_db, seed):
    """
    Tạo 1 sample channel sequence.
    Output: dict {
        'He_history': (P, N_RF, N_M, 2),    # P=30 historical effective channels
        'He_future':  (Q, N_RF, N_M, 2),    # Q=5 future effective channels
        'D': (N_RF, N_BS, 2),               # analog combiner (fixed in sample)
        'velocity': float,
        'delay_spread': float,
        'snr': float,
    }

    Steps:
        1. Sample 23 cluster angles từ Laplacian PAS theo CDL-A
        2. Tính Doppler shift cho mỗi cluster: f_l = v/λ · cos(θ_l)
        3. Generate L=23 path gains ξ_l (complex Gaussian theo CDL-A power profile)
        4. For t in range(P + Q):
            H(t) = Σ_l ξ_l · a_BS(φ_BS,l) · a_user^H(φ_user,l) · exp(-j2π f_l t T_s)
        5. Beam training: select top-8 strongest beams → D ∈ ℂ^(8 × 64)
        6. He(t) = D · H(t) ∈ ℂ^(8 × 4)
        7. Add AWGN với SNR target
        8. Tách real/imag → output shape (P+Q, 8, 4, 2)
        9. Split P=30 historical, Q=5 future
    """
```

### 5.4 Số lượng samples

| Split | Số samples | Mục đích |
|---|---|---|
| Train | **8000** | Training với velocity/SNR/delay ngẫu nhiên |
| Val | **1000** | Validation trong khi train (early stopping) |
| Test (v=30km/h) | **500** | Test scenario 1 |
| Test (v=50km/h) | **500** | Test scenario 2 |
| **Tổng** | **10000** | Khớp với IEICE paper |

### 5.5 Normalization

**Quy tắc:** Compute `scale = max(|He_train|)` trên toàn train set, lưu vào file `data/normalize_stats.npz`. Apply: `He_normalized = He / scale`.

**Lý do:** Channel có dynamic range lớn (gain Rayleigh có thể vài chục dB). Normalize ổn định training.

**Code skeleton:**
```python
# src/data/normalize.py
import numpy as np
from pathlib import Path

class ChannelNormalizer:
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    @classmethod
    def fit(cls, train_data: np.ndarray) -> 'ChannelNormalizer':
        """train_data shape: (N_samples, P+Q, N_RF, N_M, 2)"""
        scale = float(np.max(np.abs(train_data)))
        return cls(scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x / self.scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale

    def save(self, path: Path):
        np.savez(path, scale=self.scale)

    @classmethod
    def load(cls, path: Path) -> 'ChannelNormalizer':
        data = np.load(path)
        return cls(scale=float(data['scale']))
```

### 5.6 PyTorch Dataset class

```python
# src/data/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict

class ChannelDataset(Dataset):
    """
    Dataset cho channel prediction.

    Mỗi sample là 1 file .npz chứa He_history, He_future.
    Lazy loading: chỉ load khi __getitem__ được gọi (giữ RAM thấp).
    """
    def __init__(
        self,
        data_dir: Path,
        normalizer: Optional['ChannelNormalizer'] = None,
        flatten_spatial: bool = False,  # True for baselines (RNN/LSTM/GRU/Transformer)
    ):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("sample_*.npz"))
        if len(self.files) == 0:
            raise RuntimeError(f"No samples found in {data_dir}")
        self.normalizer = normalizer
        self.flatten_spatial = flatten_spatial

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.files[idx])
        He_history = data['He_history'].astype(np.float32)  # (P, N_RF, N_M, 2)
        He_future  = data['He_future'].astype(np.float32)   # (Q, N_RF, N_M, 2)

        if self.normalizer is not None:
            He_history = self.normalizer.transform(He_history)
            He_future  = self.normalizer.transform(He_future)

        if self.flatten_spatial:
            # For baselines: (P, N_RF*N_M*2) = (30, 64)
            P = He_history.shape[0]
            Q = He_future.shape[0]
            He_history = He_history.reshape(P, -1)
            He_future  = He_future.reshape(Q, -1)

        return {
            'history': torch.from_numpy(He_history),
            'future':  torch.from_numpy(He_future),
            'metadata': {
                'velocity': float(data['velocity']),
                'snr': float(data['snr']),
                'delay_spread': float(data['delay_spread']),
            }
        }
```

### 5.7 Script generate data

`scripts/01_generate_data.py`:

```python
"""
Generate CDL-A channel dataset for training and testing.

Usage:
    python scripts/01_generate_data.py --config configs/data/cdl_a.yaml

Output:
    data/train/sample_00000.npz, sample_00001.npz, ..., sample_07999.npz
    data/val/sample_00000.npz, ...
    data/test_v30/sample_00000.npz, ...
    data/test_v50/sample_00000.npz, ...
    data/normalize_stats.npz
"""
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.data.channel_generator import generate_one_sample
from src.data.normalize import ChannelNormalizer
from src.utils.config import load_config
from src.utils.seed import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data/cdl_a.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    # ========== TRAIN ==========
    train_dir = Path(cfg.output_dir) / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    train_samples = []
    for i in tqdm(range(cfg.n_train), desc="Generating train"):
        v = np.random.uniform(*cfg.velocity_range_train)
        ds = np.random.uniform(*cfg.delay_spread_range)
        snr = np.random.uniform(*cfg.snr_range_train)
        sample = generate_one_sample(v, ds, snr, seed=cfg.seed + i)
        np.savez(train_dir / f"sample_{i:05d}.npz", **sample)
        # Collect for normalizer fitting
        train_samples.append(np.concatenate([sample['He_history'], sample['He_future']], axis=0))

    # ========== Fit normalizer ==========
    all_train = np.stack(train_samples)
    normalizer = ChannelNormalizer.fit(all_train)
    normalizer.save(Path(cfg.output_dir) / "normalize_stats.npz")
    print(f"Normalizer scale: {normalizer.scale:.4f}")

    # ========== VAL ==========
    val_dir = Path(cfg.output_dir) / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(cfg.n_val), desc="Generating val"):
        v = np.random.uniform(*cfg.velocity_range_train)
        ds = np.random.uniform(*cfg.delay_spread_range)
        snr = np.random.uniform(*cfg.snr_range_train)
        sample = generate_one_sample(v, ds, snr, seed=cfg.seed + 100000 + i)
        np.savez(val_dir / f"sample_{i:05d}.npz", **sample)

    # ========== TEST v=30 ==========
    test_v30_dir = Path(cfg.output_dir) / "test_v30"
    test_v30_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(cfg.n_test_per_scenario), desc="Generating test v=30"):
        sample = generate_one_sample(
            velocity_kmh=30.0,
            delay_spread_ns=50.0,
            snr_db=20.0,  # SNR adjusted per-test in evaluation
            seed=cfg.seed + 200000 + i
        )
        np.savez(test_v30_dir / f"sample_{i:05d}.npz", **sample)

    # ========== TEST v=50 ==========
    test_v50_dir = Path(cfg.output_dir) / "test_v50"
    test_v50_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(cfg.n_test_per_scenario), desc="Generating test v=50"):
        sample = generate_one_sample(
            velocity_kmh=50.0,
            delay_spread_ns=50.0,
            snr_db=20.0,
            seed=cfg.seed + 300000 + i
        )
        np.savez(test_v50_dir / f"sample_{i:05d}.npz", **sample)

    print("Done.")

if __name__ == "__main__":
    main()
```

`configs/data/cdl_a.yaml`:

```yaml
# CDL-A dataset configuration
seed: 42
output_dir: data

# System parameters (FIXED according to IEICE 2026 paper)
n_bs: 64
n_rf: 8
n_m: 4
f_carrier_ghz: 30.0
t_sample_ms: 0.625
n_paths: 23  # CDL-A has 23 clusters

# Frame structure
p_frames: 30   # input window
q_frames: 5    # prediction horizon
l_frames: 10   # decoder start token length

# Random ranges for training samples
velocity_range_train: [30.0, 50.0]   # km/h
delay_spread_range: [25.0, 250.0]    # ns
snr_range_train: [10.0, 20.0]        # dB

# Dataset sizes
n_train: 8000
n_val: 1000
n_test_per_scenario: 500
```

---

## 6. KIẾN TRÚC MODEL ĐỀ XUẤT: Spatial-Temporal Mamba

### 6.1 Tổng quan kiến trúc

```
INPUT: He_history ∈ ℝ^(B, P, N_RF, N_M, 2)
       Shape: (B, 30, 8, 4, 2)

  ↓ [STAGE 1: Spatial Encoder — CNN 2D, áp dụng PER timestep]
  Mỗi timestep: (8, 4, 2) → Conv2D layers → spatial feature d_spatial=64

  ↓ Shape sau Stage 1: (B, P, d_spatial) = (B, 30, 64)

  ↓ [STAGE 2: Temporal Predictor — Mamba blocks + parallel head]
  Học dynamics theo thời gian, output Q time steps

  ↓ Shape sau Stage 2: (B, Q, d_spatial) = (B, 5, 64)

  ↓ [STAGE 3: Spatial Decoder — Transposed CNN 2D, áp dụng PER timestep]
  Mỗi timestep: (64) → Linear + Reshape → ConvT2D layers → (8, 4, 2)

OUTPUT: He_predicted ∈ ℝ^(B, Q, N_RF, N_M, 2)
        Shape: (B, 5, 8, 4, 2)
```

### 6.2 Spatial Encoder — Chi tiết

**File:** `src/models/spatial_encoder.py`

```python
"""
Spatial Encoder: CNN 2D encode each timestep's channel into a d-dim feature vector.

Input shape:  (B, P, N_RF, N_M, 2)
Output shape: (B, P, d_spatial)

Design rationale:
  - Treat (real, imag) as 2 channels of a 2D image of size (N_RF, N_M)
  - Conv2D learns spatial correlations between adjacent antennas
  - GlobalAvgPool to fixed-size feature vector
  - Lightweight: ~10K params
"""
import torch
import torch.nn as nn

class SpatialEncoder(nn.Module):
    def __init__(
        self,
        n_rf: int = 8,
        n_m: int = 4,
        d_spatial: int = 64,
        cnn_channels: tuple = (16, 32, 64),
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.n_rf = n_rf
        self.n_m = n_m
        self.d_spatial = d_spatial

        layers = []
        in_ch = 2  # real, imag
        for out_ch in cnn_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.conv_net = nn.Sequential(*layers)

        # Global pooling → d_spatial
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cnn_channels[-1], d_spatial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, P, N_RF, N_M, 2)
        Returns:
            (B, P, d_spatial)
        """
        B, P, N_RF, N_M, C = x.shape
        assert C == 2, "Last dim must be 2 (real, imag)"

        # Reshape to (B*P, 2, N_RF, N_M) — treat as 2D image with 2 channels
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # (B, P, 2, N_RF, N_M)
        x = x.view(B * P, 2, N_RF, N_M)

        # CNN
        x = self.conv_net(x)  # (B*P, cnn_channels[-1], N_RF, N_M)

        # Global pool
        x = self.pool(x).flatten(1)  # (B*P, cnn_channels[-1])

        # FC
        x = self.fc(x)  # (B*P, d_spatial)

        # Reshape back
        x = x.view(B, P, self.d_spatial)
        return x

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Sanity check
    enc = SpatialEncoder(n_rf=8, n_m=4, d_spatial=64)
    x = torch.randn(2, 30, 8, 4, 2)
    y = enc(x)
    assert y.shape == (2, 30, 64), f"Expected (2,30,64), got {y.shape}"
    print(f"SpatialEncoder OK | params: {enc.count_params():,}")
```

### 6.3 Mamba Block — Chi tiết

**File:** `src/models/mamba_block.py`

**Quan trọng — Mamba implementation options:**

1. **Option A (Preferred):** `pip install mamba-ssm` — official, fast CUDA
2. **Option B (Fallback):** Pure PyTorch implementation từ https://github.com/alxndrTL/mamba.py
3. **Option C (Last resort):** Implement S4D đơn giản inline

Code phải support cả 3 với try/except gracefully:

```python
"""
Mamba block wrapper.

Tries to import official mamba-ssm first; falls back to pure PyTorch implementation
if not available (e.g., on Windows where CUDA build can be tricky).
"""
import torch
import torch.nn as nn
from typing import Optional

# Try to import official mamba-ssm
try:
    from mamba_ssm import Mamba
    MAMBA_BACKEND = "official"
    print("[INFO] Using official mamba-ssm package")
except ImportError:
    MAMBA_BACKEND = "fallback"
    print("[WARN] mamba-ssm not found, using pure PyTorch fallback")


class MambaBlockFallback(nn.Module):
    """
    Pure PyTorch implementation of Mamba's Selective SSM (S6).

    Reference: Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

    This is a simplified version. For best performance, install mamba-ssm.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # A, D parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # (d_inner, d_state)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        B, L, _ = x.shape

        # Project and split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # (B, L, d_inner) each

        # 1D conv
        x_in = x_in.transpose(1, 2)  # (B, d_inner, L)
        x_in = self.conv1d(x_in)[:, :, :L]  # (B, d_inner, L)
        x_in = x_in.transpose(1, 2)  # (B, L, d_inner)
        x_in = torch.nn.functional.silu(x_in)

        # SSM
        y = self._ssm_scan(x_in)

        # Gating
        y = y * torch.nn.functional.silu(z)

        # Output projection
        return self.out_proj(y)

    def _ssm_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective SSM scan.

        Args:
            x: (B, L, d_inner)
        Returns:
            (B, L, d_inner)
        """
        B, L, D = x.shape
        N = self.d_state

        # Compute Δ, B, C from x
        x_dbl = self.x_proj(x)  # (B, L, N + N + d_inner)
        delta, B_param, C_param = torch.split(x_dbl, [D, N, N], dim=-1)
        # Wait — split should be [d_inner, N, N], let me fix

        # Actually the projection outputs d_state*2 + d_inner for (delta_raw, B, C)
        # Rearrange
        x_proj_out = x_dbl  # placeholder, real impl differs

        # For simplicity in fallback, use a basic linear recurrence
        # This is NOT the true selective SSM but a placeholder
        A = -torch.exp(self.A_log)  # (d_inner, N)
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        outs = []
        delta = torch.nn.functional.softplus(self.dt_proj(x))  # (B, L, d_inner)
        for t in range(L):
            # h = exp(A * delta) * h + delta * B * x_t   (simplified)
            dt = delta[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            decay = torch.exp(dt * A.unsqueeze(0))  # (B, d_inner, N)
            h = decay * h + dt * x[:, t, :].unsqueeze(-1)  # (B, d_inner, N)
            y_t = h.sum(dim=-1) + self.D * x[:, t, :]  # (B, d_inner)
            outs.append(y_t)
        return torch.stack(outs, dim=1)  # (B, L, d_inner)


def make_mamba_block(d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> nn.Module:
    """Factory: use official mamba-ssm if available, else fallback."""
    if MAMBA_BACKEND == "official":
        return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    else:
        return MambaBlockFallback(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


class BidirectionalMamba(nn.Module):
    """
    Bidirectional Mamba: process sequence forward and backward, concatenate.

    Inspired by ViM (Vision Mamba) bidirectional design.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.forward_mamba = make_mamba_block(d_model, d_state, d_conv, expand)
        self.backward_mamba = make_mamba_block(d_model, d_state, d_conv, expand)
        self.merge = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        x_fwd = self.forward_mamba(x)
        x_bwd = self.backward_mamba(torch.flip(x, dims=[1]))
        x_bwd = torch.flip(x_bwd, dims=[1])
        merged = self.merge(torch.cat([x_fwd, x_bwd], dim=-1))
        return self.norm(x + merged)  # residual + norm


if __name__ == "__main__":
    block = BidirectionalMamba(d_model=64, d_state=16)
    x = torch.randn(2, 30, 64)
    y = block(x)
    assert y.shape == (2, 30, 64), f"Expected (2,30,64), got {y.shape}"
    n_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    print(f"BidirectionalMamba OK | params: {n_params:,} | backend: {MAMBA_BACKEND}")
```

### 6.4 Spatial Decoder — Chi tiết

**File:** `src/models/spatial_decoder.py`

```python
"""
Spatial Decoder: reconstruct channel matrix from spatial feature vector.

Input shape:  (B, Q, d_spatial)
Output shape: (B, Q, N_RF, N_M, 2)
"""
import torch
import torch.nn as nn

class SpatialDecoder(nn.Module):
    def __init__(
        self,
        n_rf: int = 8,
        n_m: int = 4,
        d_spatial: int = 64,
        hidden_channels: tuple = (64, 32, 16),
    ):
        super().__init__()
        self.n_rf = n_rf
        self.n_m = n_m
        self.d_spatial = d_spatial

        # FC to expand back to spatial features
        # d_spatial → hidden_channels[0] * N_RF * N_M
        self.fc = nn.Linear(d_spatial, hidden_channels[0] * n_rf * n_m)

        # Refining CNN layers
        layers = []
        in_ch = hidden_channels[0]
        for out_ch in hidden_channels[1:]:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        # Final conv to 2 channels (real, imag)
        layers.append(nn.Conv2d(in_ch, 2, kernel_size=3, padding=1))
        self.conv_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, Q, d_spatial)
        Returns:
            (B, Q, N_RF, N_M, 2)
        """
        B, Q, D = x.shape

        # Flatten time into batch
        x = x.view(B * Q, D)  # (B*Q, d_spatial)

        # FC then reshape to (B*Q, hidden_channels[0], N_RF, N_M)
        x = self.fc(x)
        x = x.view(B * Q, -1, self.n_rf, self.n_m)

        # CNN refinement to 2 output channels
        x = self.conv_net(x)  # (B*Q, 2, N_RF, N_M)

        # Reshape: (B, Q, N_RF, N_M, 2)
        x = x.view(B, Q, 2, self.n_rf, self.n_m)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        return x


if __name__ == "__main__":
    dec = SpatialDecoder(n_rf=8, n_m=4, d_spatial=64)
    x = torch.randn(2, 5, 64)
    y = dec(x)
    assert y.shape == (2, 5, 8, 4, 2), f"Expected (2,5,8,4,2), got {y.shape}"
    n_params = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    print(f"SpatialDecoder OK | params: {n_params:,}")
```

### 6.5 Spatial-Temporal Mamba — Model chính

**File:** `src/models/st_mamba.py`

```python
"""
Spatial-Temporal Mamba (ST-Mamba) for MIMO Channel Prediction.

Architecture:
    Input (B, P, N_RF, N_M, 2)
        ↓ SpatialEncoder (CNN 2D per timestep)
    (B, P, d_spatial)
        ↓ Mamba blocks (bidirectional, stacked)
    (B, P, d_spatial)
        ↓ Parallel prediction head (P → Q steps)
    (B, Q, d_spatial)
        ↓ SpatialDecoder (CNN 2D per timestep)
    Output (B, Q, N_RF, N_M, 2)

Novel contributions:
    1. First Mamba/SSM application to channel prediction
    2. Spatial-temporal decoupling (CNN2D + Mamba)
    3. Parallel prediction head (predict Q frames at once, no autoregression)
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.models.spatial_encoder import SpatialEncoder
from src.models.spatial_decoder import SpatialDecoder
from src.models.mamba_block import BidirectionalMamba


class ParallelPredictionHead(nn.Module):
    """
    Predict Q future frames from P historical features in a single forward pass.

    Inspired by IEICE 2026's parallel synchronous prediction, but using
    Mamba features instead of Transformer encoder-decoder.

    Mechanism:
        - Take encoded features (B, P, d_spatial)
        - Use learned Q query tokens (Q, d_spatial)
        - Cross-attention between queries and encoded features
        - Output (B, Q, d_spatial)
    """
    def __init__(
        self,
        d_spatial: int,
        p_frames: int,
        q_frames: int,
        n_heads: int = 4,
    ):
        super().__init__()
        self.d_spatial = d_spatial
        self.p_frames = p_frames
        self.q_frames = q_frames

        # Learnable query embeddings for Q future positions
        self.queries = nn.Parameter(torch.randn(q_frames, d_spatial) * 0.02)

        # Cross-attention from Q queries to P encoded features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_spatial,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.1,
        )

        # Feed-forward
        self.norm1 = nn.LayerNorm(d_spatial)
        self.norm2 = nn.LayerNorm(d_spatial)
        self.ffn = nn.Sequential(
            nn.Linear(d_spatial, d_spatial * 2),
            nn.GELU(),
            nn.Linear(d_spatial * 2, d_spatial),
        )

    def forward(self, encoded_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoded_features: (B, P, d_spatial) — output from Mamba encoder
        Returns:
            (B, Q, d_spatial) — features for Q future frames
        """
        B = encoded_features.shape[0]

        # Expand queries to batch
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, Q, d_spatial)

        # Cross-attention: queries attend to encoded P features
        attn_out, _ = self.cross_attn(
            query=q,
            key=encoded_features,
            value=encoded_features,
        )
        q = self.norm1(q + attn_out)

        # FFN
        ffn_out = self.ffn(q)
        q = self.norm2(q + ffn_out)

        return q  # (B, Q, d_spatial)


class STMamba(nn.Module):
    """
    Spatial-Temporal Mamba for MIMO channel prediction.
    """
    def __init__(
        self,
        n_rf: int = 8,
        n_m: int = 4,
        p_frames: int = 30,
        q_frames: int = 5,
        d_spatial: int = 64,
        n_mamba_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        cnn_encoder_channels: tuple = (16, 32, 64),
        cnn_decoder_channels: tuple = (64, 32, 16),
        n_pred_heads: int = 4,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.config = {
            'n_rf': n_rf, 'n_m': n_m,
            'p_frames': p_frames, 'q_frames': q_frames,
            'd_spatial': d_spatial,
            'n_mamba_layers': n_mamba_layers,
        }

        # Spatial Encoder
        self.spatial_encoder = SpatialEncoder(
            n_rf=n_rf, n_m=n_m, d_spatial=d_spatial,
            cnn_channels=cnn_encoder_channels,
            use_batchnorm=use_batchnorm,
        )

        # Stacked bidirectional Mamba blocks (temporal predictor)
        self.mamba_blocks = nn.ModuleList([
            BidirectionalMamba(
                d_model=d_spatial,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_mamba_layers)
        ])

        # Parallel prediction head
        self.pred_head = ParallelPredictionHead(
            d_spatial=d_spatial,
            p_frames=p_frames,
            q_frames=q_frames,
            n_heads=n_pred_heads,
        )

        # Spatial Decoder
        self.spatial_decoder = SpatialDecoder(
            n_rf=n_rf, n_m=n_m, d_spatial=d_spatial,
            hidden_channels=cnn_decoder_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, P, N_RF, N_M, 2) — historical effective channels
        Returns:
            (B, Q, N_RF, N_M, 2) — predicted future channels
        """
        # Stage 1: Spatial encoding per timestep
        spatial_feats = self.spatial_encoder(x)  # (B, P, d_spatial)

        # Stage 2: Temporal modeling with stacked Mamba
        h = spatial_feats
        for mamba_block in self.mamba_blocks:
            h = mamba_block(h)
        # h: (B, P, d_spatial)

        # Stage 3: Parallel prediction P → Q
        future_feats = self.pred_head(h)  # (B, Q, d_spatial)

        # Stage 4: Spatial decoding per future timestep
        pred = self.spatial_decoder(future_feats)  # (B, Q, N_RF, N_M, 2)

        return pred

    def count_params(self) -> dict:
        """Return parameter breakdown."""
        return {
            'spatial_encoder': sum(p.numel() for p in self.spatial_encoder.parameters()),
            'mamba_blocks':    sum(p.numel() for p in self.mamba_blocks.parameters()),
            'pred_head':       sum(p.numel() for p in self.pred_head.parameters()),
            'spatial_decoder': sum(p.numel() for p in self.spatial_decoder.parameters()),
            'total':           sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


if __name__ == "__main__":
    model = STMamba(
        n_rf=8, n_m=4, p_frames=30, q_frames=5,
        d_spatial=64, n_mamba_layers=4,
    )
    x = torch.randn(2, 30, 8, 4, 2)
    y = model(x)
    assert y.shape == (2, 5, 8, 4, 2), f"Expected (2,5,8,4,2), got {y.shape}"
    params = model.count_params()
    print(f"STMamba OK")
    for k, v in params.items():
        print(f"  {k}: {v:,}")
```

### 6.6 Target params cho fair comparison

**Quy tắc:** Tất cả model phải nằm trong khoảng **300K-400K parameters** để fair comparison.

Sau khi viết code, **VERIFY**:

```python
# Tất cả model phải in được params và rơi vào target range
ST-Mamba       : ~350K  (target)
RNN baseline   : ~340K  (tăng hidden_size để đạt)
LSTM baseline  : ~360K
GRU baseline   : ~355K
Transformer    : ~370K  (IEICE-style)
```

Cách điều chỉnh: thay đổi `d_spatial`, `n_mamba_layers`, `hidden_size` để equalize.

`configs/model/st_mamba.yaml`:

```yaml
# Spatial-Temporal Mamba configuration
model_name: st_mamba
n_rf: 8
n_m: 4
p_frames: 30
q_frames: 5
d_spatial: 64
n_mamba_layers: 4
d_state: 16
d_conv: 4
expand: 2
cnn_encoder_channels: [16, 32, 64]
cnn_decoder_channels: [64, 32, 16]
n_pred_heads: 4
use_batchnorm: true
```

---

## 7. BASELINE MODELS

Tất cả baseline phải:
1. Cùng input/output shape với ST-Mamba: input `(B, P, N_RF*N_M*2)`, output `(B, Q, N_RF*N_M*2)`
2. Tham số ~300K-400K (fair comparison)
3. **Cũng dùng parallel prediction head** (không autoregressive) — tránh confound "Mamba thắng vì parallel prediction"

### 7.1 RNN Baseline

**File:** `src/models/baselines/rnn.py`

```python
"""
RNN baseline with parallel prediction head.

Note: We use parallel prediction (not autoregressive) for fair comparison
with ST-Mamba and IEICE Transformer.
"""
import torch
import torch.nn as nn

class RNNBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,   # N_RF * N_M * 2 = 8*4*2
        hidden_size: int = 256,
        num_layers: int = 3,
        p_frames: int = 30,
        q_frames: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.p_frames = p_frames
        self.q_frames = q_frames

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity='tanh',
        )

        # Parallel prediction head
        self.queries = nn.Parameter(torch.randn(q_frames, hidden_size) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, P, input_dim) — flattened channels
        Returns:
            (B, Q, input_dim) — predicted channels
        """
        B = x.shape[0]

        # RNN encoding
        h, _ = self.rnn(x)  # (B, P, hidden_size)

        # Cross-attention parallel prediction
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, Q, hidden_size)
        attn_out, _ = self.cross_attn(query=q, key=h, value=h)
        q = self.norm(q + attn_out)

        # Output projection
        return self.output_proj(q)  # (B, Q, input_dim)


if __name__ == "__main__":
    model = RNNBaseline(input_dim=64, hidden_size=256, num_layers=3,
                        p_frames=30, q_frames=5)
    x = torch.randn(2, 30, 64)
    y = model(x)
    assert y.shape == (2, 5, 64), f"Got {y.shape}"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"RNN OK | params: {n_params:,}")
```

`configs/model/rnn.yaml`:

```yaml
model_name: rnn
input_dim: 64
hidden_size: 256
num_layers: 3
p_frames: 30
q_frames: 5
dropout: 0.1
```

### 7.2 LSTM Baseline

**File:** `src/models/baselines/lstm.py`

```python
"""LSTM baseline with parallel prediction head."""
import torch
import torch.nn as nn

class LSTMBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        hidden_size: int = 192,
        num_layers: int = 2,
        p_frames: int = 30,
        q_frames: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.queries = nn.Parameter(torch.randn(q_frames, hidden_size) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=4, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h, _ = self.lstm(x)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(query=q, key=h, value=h)
        q = self.norm(q + attn_out)
        return self.output_proj(q)


if __name__ == "__main__":
    model = LSTMBaseline(input_dim=64, hidden_size=192, num_layers=2,
                         p_frames=30, q_frames=5)
    x = torch.randn(2, 30, 64)
    y = model(x)
    assert y.shape == (2, 5, 64)
    print(f"LSTM OK | params: {sum(p.numel() for p in model.parameters()):,}")
```

`configs/model/lstm.yaml`:

```yaml
model_name: lstm
input_dim: 64
hidden_size: 192
num_layers: 2
p_frames: 30
q_frames: 5
dropout: 0.1
```

### 7.3 GRU Baseline

**File:** `src/models/baselines/gru.py`

```python
"""GRU baseline with parallel prediction head."""
import torch
import torch.nn as nn

class GRUBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        hidden_size: int = 224,
        num_layers: int = 2,
        p_frames: int = 30,
        q_frames: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.queries = nn.Parameter(torch.randn(q_frames, hidden_size) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=4, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h, _ = self.gru(x)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(query=q, key=h, value=h)
        q = self.norm(q + attn_out)
        return self.output_proj(q)


if __name__ == "__main__":
    model = GRUBaseline(input_dim=64, hidden_size=224, num_layers=2,
                        p_frames=30, q_frames=5)
    x = torch.randn(2, 30, 64)
    y = model(x)
    assert y.shape == (2, 5, 64)
    print(f"GRU OK | params: {sum(p.numel() for p in model.parameters()):,}")
```

`configs/model/gru.yaml`:

```yaml
model_name: gru
input_dim: 64
hidden_size: 224
num_layers: 2
p_frames: 30
q_frames: 5
dropout: 0.1
```

### 7.4 Transformer Baseline (IEICE-style)

**File:** `src/models/baselines/transformer.py`

```python
"""
Transformer baseline modeled after IEICE 2026 "Hybrid-Attention Parallel Synchronous Transformer".

Differences from our ST-Mamba (deliberate, so comparison is meaningful):
  - Uses self-attention instead of Mamba SSM
  - No spatial decoupling (flattens channel)
  - No physics-informed loss (in baseline training)
"""
import torch
import torch.nn as nn

class TransformerBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        p_frames: int = 30,
        q_frames: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.p_frames = p_frames
        self.q_frames = q_frames

        # Input embedding
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding (learnable)
        self.pos_enc_encoder = nn.Parameter(torch.randn(p_frames, d_model) * 0.02)
        self.pos_enc_decoder = nn.Parameter(torch.randn(q_frames, d_model) * 0.02)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder (parallel synchronous: Q learned query tokens)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Learnable Q queries
        self.queries = nn.Parameter(torch.randn(q_frames, d_model) * 0.02)

        # Output projection
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, P, input_dim)
        Returns:
            (B, Q, input_dim)
        """
        B = x.shape[0]

        # Project + positional
        x = self.input_proj(x)  # (B, P, d_model)
        x = x + self.pos_enc_encoder.unsqueeze(0)

        # Encode
        memory = self.encoder(x)  # (B, P, d_model)

        # Decode with learned queries
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, Q, d_model)
        q = q + self.pos_enc_decoder.unsqueeze(0)
        out = self.decoder(tgt=q, memory=memory)  # (B, Q, d_model)

        return self.output_proj(out)


if __name__ == "__main__":
    model = TransformerBaseline(
        input_dim=64, d_model=128, n_heads=4,
        num_encoder_layers=3, num_decoder_layers=2,
        dim_feedforward=256, p_frames=30, q_frames=5,
    )
    x = torch.randn(2, 30, 64)
    y = model(x)
    assert y.shape == (2, 5, 64)
    print(f"Transformer OK | params: {sum(p.numel() for p in model.parameters()):,}")
```

`configs/model/transformer.yaml`:

```yaml
model_name: transformer
input_dim: 64
d_model: 128
n_heads: 4
num_encoder_layers: 3
num_decoder_layers: 2
dim_feedforward: 256
p_frames: 30
q_frames: 5
dropout: 0.1
```

---

## 8. LOSS FUNCTIONS

### 8.1 NMSE Loss

**File:** `src/losses/nmse.py`

```python
"""
NMSE Loss: Normalized Mean Squared Error.

Formula:
    NMSE = E[||y_pred - y_true||²] / E[||y_true||²]

This is the standard loss for channel prediction.
"""
import torch
import torch.nn as nn

class NMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: (B, Q, ...) predicted channels
            y_true: (B, Q, ...) ground truth channels
        Returns:
            scalar NMSE loss
        """
        # Reduce to (B, Q, -1) for power computation
        y_pred_flat = y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1)
        y_true_flat = y_true.reshape(y_true.shape[0], y_true.shape[1], -1)

        numerator = torch.sum((y_pred_flat - y_true_flat) ** 2, dim=-1)  # (B, Q)
        denominator = torch.sum(y_true_flat ** 2, dim=-1) + self.eps     # (B, Q)
        nmse_per_sample = numerator / denominator  # (B, Q)
        return nmse_per_sample.mean()  # average over batch and Q


def nmse_db(nmse: torch.Tensor) -> torch.Tensor:
    """Convert NMSE to dB."""
    return 10 * torch.log10(nmse.clamp(min=1e-12))
```

### 8.2 Phase Loss

**File:** `src/losses/phase.py`

```python
"""
Phase Loss: penalize phase errors in complex channels.

For each (real, imag) pair, compute phase = atan2(imag, real).
Then minimize circular phase difference.
"""
import torch
import torch.nn as nn

class PhaseLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: (B, Q, N_RF, N_M, 2)
            y_true: (B, Q, N_RF, N_M, 2)
        Returns:
            scalar phase loss
        """
        # Compute phase
        phase_pred = torch.atan2(y_pred[..., 1], y_pred[..., 0] + self.eps)
        phase_true = torch.atan2(y_true[..., 1], y_true[..., 0] + self.eps)

        # Circular difference: handle wrap-around at ±π
        diff = phase_pred - phase_true
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))

        # Weight by magnitude (don't penalize phase of near-zero entries)
        magnitude = torch.sqrt(y_true[..., 0]**2 + y_true[..., 1]**2 + self.eps)
        weighted_diff = (diff.abs() * magnitude).mean()
        return weighted_diff
```

### 8.3 Temporal Consistency Loss

**File:** `src/losses/temporal.py`

```python
"""
Temporal Consistency Loss: enforce predicted channels evolve at the correct rate.

If real channel changes by Δ_real between t and t+1,
predicted channel should change by approximately the same Δ.

This encodes Doppler physics: channels at high velocity change faster
than at low velocity, and the model should learn this implicitly.
"""
import torch
import torch.nn as nn

class TemporalConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: (B, Q, ...) predicted channels
            y_true: (B, Q, ...) ground truth channels (Q ≥ 2)
        Returns:
            scalar temporal consistency loss
        """
        if y_pred.shape[1] < 2:
            return torch.tensor(0.0, device=y_pred.device)

        # Compute consecutive differences (rate of change)
        pred_diff = y_pred[:, 1:] - y_pred[:, :-1]  # (B, Q-1, ...)
        true_diff = y_true[:, 1:] - y_true[:, :-1]  # (B, Q-1, ...)

        # Compare magnitudes of rate-of-change
        pred_rate = pred_diff.reshape(pred_diff.shape[0], pred_diff.shape[1], -1).norm(dim=-1)  # (B, Q-1)
        true_rate = true_diff.reshape(true_diff.shape[0], true_diff.shape[1], -1).norm(dim=-1)  # (B, Q-1)

        return (pred_rate - true_rate).abs().mean()
```

### 8.4 Combined Loss

**File:** `src/losses/combined.py`

```python
"""
Multi-component physics-informed loss for channel prediction.

L_total = α · L_NMSE + β · L_phase + γ · L_temporal
"""
import torch
import torch.nn as nn
from src.losses.nmse import NMSELoss
from src.losses.phase import PhaseLoss
from src.losses.temporal import TemporalConsistencyLoss

class CombinedLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,    # NMSE weight
        beta: float = 0.1,     # Phase weight
        gamma: float = 0.05,   # Temporal weight
        eps: float = 1e-8,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.nmse_loss = NMSELoss(eps=eps)
        self.phase_loss = PhaseLoss(eps=eps)
        self.temporal_loss = TemporalConsistencyLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
        """
        Args:
            y_pred: (B, Q, ...) predicted
            y_true: (B, Q, ...) ground truth
        Returns:
            dict with 'total', 'nmse', 'phase', 'temporal' losses
        """
        l_nmse = self.nmse_loss(y_pred, y_true)
        l_phase = self.phase_loss(y_pred, y_true) if y_pred.dim() == 5 else torch.tensor(0.0, device=y_pred.device)
        l_temporal = self.temporal_loss(y_pred, y_true)

        total = self.alpha * l_nmse + self.beta * l_phase + self.gamma * l_temporal

        return {
            'total': total,
            'nmse': l_nmse.detach(),
            'phase': l_phase.detach() if isinstance(l_phase, torch.Tensor) else l_phase,
            'temporal': l_temporal.detach(),
        }
```

---

## 9. TRAINING

### 9.1 Trainer

**File:** `src/training/trainer.py`

```python
"""
Generic trainer for channel prediction models.

Supports:
    - AdamW optimizer with warmup + cosine annealing
    - Mixed precision training (FP16) — important for GTX 1650 4GB
    - Gradient clipping
    - Early stopping
    - Checkpoint saving
    - TensorBoard logging
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Optional, Callable
from tqdm import tqdm

from src.utils.logger import Logger
from src.training.optimizer import build_optimizer_and_scheduler


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        config: dict,
        log_dir: Path,
        ckpt_dir: Path,
        flatten_input: bool = False,  # True for baselines
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.flatten_input = flatten_input

        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            model, config, len(train_loader)
        )

        self.scaler = GradScaler() if config.get('use_amp', True) else None
        self.logger = Logger(log_dir)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train(self):
        n_epochs = self.config['n_epochs']
        for epoch in range(n_epochs):
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate(epoch)

            self.logger.log_metrics({**train_metrics, **val_metrics}, step=epoch)

            # Early stopping & checkpointing
            val_loss = val_metrics['val/total_loss']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, 'best')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.get('patience', 15):
                    print(f"Early stopping at epoch {epoch}")
                    break

            self._save_checkpoint(epoch, 'last')

        return self.best_val_loss

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        losses = {'total': [], 'nmse': [], 'phase': [], 'temporal': []}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]")
        for batch in pbar:
            x = batch['history'].to(self.device)
            y = batch['future'].to(self.device)

            # For baselines: flatten spatial dims
            if self.flatten_input and x.dim() == 5:
                B, P = x.shape[0], x.shape[1]
                x_in = x.view(B, P, -1)
                y_in = y.view(B, y.shape[1], -1)
            else:
                x_in = x
                y_in = y

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast():
                    pred = self.model(x_in)
                    if self.flatten_input:
                        # reshape pred back to (B, Q, N_RF, N_M, 2) for losses that need it
                        pred_for_loss = pred.view(B, -1, *y.shape[2:])
                    else:
                        pred_for_loss = pred
                    loss_dict = self.loss_fn(pred_for_loss, y)
                self.scaler.scale(loss_dict['total']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(x_in)
                if self.flatten_input:
                    pred_for_loss = pred.view(B, -1, *y.shape[2:])
                else:
                    pred_for_loss = pred
                loss_dict = self.loss_fn(pred_for_loss, y)
                loss_dict['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            for k in losses:
                losses[k].append(loss_dict[k].item() if isinstance(loss_dict[k], torch.Tensor) else loss_dict[k])
            pbar.set_postfix({'loss': loss_dict['total'].item()})

        return {f"train/{k}_loss": sum(v) / len(v) for k, v in losses.items() if len(v) > 0}

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        self.model.eval()
        losses = {'total': [], 'nmse': [], 'phase': [], 'temporal': []}

        for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} [val]"):
            x = batch['history'].to(self.device)
            y = batch['future'].to(self.device)

            if self.flatten_input and x.dim() == 5:
                B, P = x.shape[0], x.shape[1]
                x_in = x.view(B, P, -1)
            else:
                x_in = x

            pred = self.model(x_in)
            if self.flatten_input:
                pred_for_loss = pred.view(B, -1, *y.shape[2:])
            else:
                pred_for_loss = pred

            loss_dict = self.loss_fn(pred_for_loss, y)
            for k in losses:
                losses[k].append(loss_dict[k].item() if isinstance(loss_dict[k], torch.Tensor) else loss_dict[k])

        return {f"val/{k}_loss": sum(v) / len(v) for k, v in losses.items() if len(v) > 0}

    def _save_checkpoint(self, epoch: int, tag: str):
        path = self.ckpt_dir / f"checkpoint_{tag}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }, path)
```

### 9.2 Optimizer + Scheduler

**File:** `src/training/optimizer.py`

```python
"""
AdamW optimizer with warmup + cosine annealing schedule.
"""
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math


def build_optimizer_and_scheduler(model, config, steps_per_epoch):
    optimizer = AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95),
    )

    total_steps = config['n_epochs'] * steps_per_epoch
    warmup_steps = int(config.get('warmup_ratio', 0.05) * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler
```

### 9.3 Curriculum Learning

**File:** `src/training/curriculum.py`

```python
"""
Curriculum learning: train on increasingly difficult velocities.

Stage 1 (epochs 0-30):   only v ∈ [30, 35] km/h (easy, near-static)
Stage 2 (epochs 31-60):  v ∈ [30, 45] km/h (medium)
Stage 3 (epochs 61-100): v ∈ [30, 50] km/h (full range, hardest)

Requires regenerating data with controlled velocity, OR
filtering training samples per stage (cheaper, recommended).
"""
import numpy as np
from torch.utils.data import Subset, DataLoader
from typing import List, Tuple


class CurriculumScheduler:
    """
    Filter training samples by velocity for each curriculum stage.

    Usage:
        scheduler = CurriculumScheduler(dataset, stages=[...])
        for epoch in range(n_epochs):
            loader = scheduler.get_loader(epoch, batch_size=...)
            # train one epoch with loader
    """
    def __init__(
        self,
        full_dataset,
        stages: List[Tuple[int, Tuple[float, float]]],
        # e.g. [(30, (30, 35)), (60, (30, 45)), (100, (30, 50))]
        # meaning: until epoch 30, use v in [30,35]; until 60, [30,45]; etc.
    ):
        self.full_dataset = full_dataset
        self.stages = stages

        # Build per-stage subset indices
        self.stage_indices = {}
        for end_epoch, (v_min, v_max) in stages:
            indices = []
            for i in range(len(full_dataset)):
                # Read metadata without loading full sample (use file naming or metadata index)
                # For simplicity, load metadata only
                sample = full_dataset[i]
                v = sample['metadata']['velocity']
                if v_min <= v <= v_max:
                    indices.append(i)
            self.stage_indices[end_epoch] = indices
            print(f"[Curriculum] stage until epoch {end_epoch}, v=[{v_min},{v_max}]: {len(indices)} samples")

    def get_subset(self, epoch: int) -> Subset:
        for end_epoch, _ in self.stages:
            if epoch <= end_epoch:
                return Subset(self.full_dataset, self.stage_indices[end_epoch])
        return Subset(self.full_dataset, self.stage_indices[self.stages[-1][0]])
```

### 9.4 Training script

**File:** `scripts/02_train.py`

```python
"""
Train one model.

Usage:
    python scripts/02_train.py --model st_mamba --train_config full
    python scripts/02_train.py --model lstm --train_config full
    python scripts/02_train.py --model st_mamba --train_config curriculum

This trains exactly one model. To compare all, run scripts/04_compare_all.py
"""
import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.utils.config import load_config, merge_configs
from src.utils.seed import set_seed
from src.data.dataset import ChannelDataset
from src.data.normalize import ChannelNormalizer
from src.losses.combined import CombinedLoss
from src.training.trainer import Trainer
from src.models.utils import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["st_mamba", "rnn", "lstm", "gru", "transformer"])
    parser.add_argument("--train_config", type=str, default="full",
                        choices=["full", "curriculum", "debug"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load configs
    model_cfg = load_config(f"configs/model/{args.model}.yaml")
    train_cfg = load_config(f"configs/train/{args.train_config}.yaml")
    data_cfg = load_config("configs/data/cdl_a.yaml")
    cfg = merge_configs(model_cfg, train_cfg, data_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    normalizer = ChannelNormalizer.load(Path("data/normalize_stats.npz"))
    flatten = (args.model != "st_mamba")  # baselines use flat input
    train_ds = ChannelDataset(Path("data/train"), normalizer=normalizer, flatten_spatial=flatten)
    val_ds   = ChannelDataset(Path("data/val"),   normalizer=normalizer, flatten_spatial=flatten)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg.get('num_workers', 2), pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=cfg.get('num_workers', 2), pin_memory=True)

    # Build model
    model = build_model(args.model, model_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} | params: {n_params:,}")

    # Verify fair comparison parameter budget
    PARAM_BUDGET_MIN = 280_000
    PARAM_BUDGET_MAX = 420_000
    if not (PARAM_BUDGET_MIN <= n_params <= PARAM_BUDGET_MAX):
        print(f"[WARN] Model params {n_params:,} outside fair budget [{PARAM_BUDGET_MIN:,}, {PARAM_BUDGET_MAX:,}]")
        print(f"[WARN] Adjust hyperparameters in configs/model/{args.model}.yaml")

    # Loss
    loss_fn = CombinedLoss(
        alpha=cfg.get('alpha', 1.0),
        beta=cfg.get('beta', 0.1),
        gamma=cfg.get('gamma', 0.05),
    )

    # Paths
    exp_name = args.exp_name or f"{args.model}_{args.train_config}_seed{args.seed}"
    log_dir = Path("results/logs") / exp_name
    ckpt_dir = Path("results/checkpoints") / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        config=cfg,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        flatten_input=flatten,
    )
    best_val = trainer.train()
    print(f"Done. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
```

`configs/train/full.yaml`:

```yaml
# Full training config
n_epochs: 100
batch_size: 32
lr: 3.0e-4
weight_decay: 1.0e-4
warmup_ratio: 0.05
use_amp: true            # mixed precision (FP16) for GTX 1650 4GB
patience: 15             # early stopping patience
num_workers: 2

# Loss weights
alpha: 1.0    # NMSE
beta: 0.1     # Phase
gamma: 0.05   # Temporal consistency
```

`configs/train/debug.yaml`:

```yaml
# Sanity check: overfit a tiny subset
n_epochs: 50
batch_size: 4
lr: 1.0e-3
weight_decay: 0.0
warmup_ratio: 0.0
use_amp: false
patience: 9999
num_workers: 0

alpha: 1.0
beta: 0.0
gamma: 0.0

# Special: only use first N samples for overfit test
overfit_n_samples: 8
```

### 9.5 Sanity check

**File:** `scripts/06_sanity_check.py`

```python
"""
Sanity check: verify that the model can overfit a tiny dataset.

If a model can NOT achieve <1e-4 NMSE on 8 samples in 50 epochs,
there is a bug somewhere. Stop and debug before full training.
"""
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.data.dataset import ChannelDataset
from src.data.normalize import ChannelNormalizer
from src.losses.nmse import NMSELoss
from src.models.utils import build_model


def main(model_name: str):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tiny dataset
    normalizer = ChannelNormalizer.load(Path("data/normalize_stats.npz"))
    flatten = (model_name != "st_mamba")
    full_ds = ChannelDataset(Path("data/train"), normalizer=normalizer, flatten_spatial=flatten)
    tiny_ds = Subset(full_ds, list(range(8)))
    loader = DataLoader(tiny_ds, batch_size=8, shuffle=True)

    # Model
    model_cfg = load_config(f"configs/model/{model_name}.yaml")
    model = build_model(model_name, model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = NMSELoss()

    # Train 200 steps
    model.train()
    for step in range(200):
        for batch in loader:
            x = batch['history'].to(device)
            y = batch['future'].to(device)
            if flatten and x.dim() == 5:
                B, P = x.shape[0], x.shape[1]
                x = x.view(B, P, -1)
                y_target = y.view(B, y.shape[1], -1)
            else:
                y_target = y

            pred = model(x)
            loss = loss_fn(pred, y_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step % 20 == 0:
            print(f"Step {step:3d} | NMSE = {loss.item():.6f} | NMSE dB = {10*torch.log10(loss).item():.2f}")

    final_nmse_db = 10 * torch.log10(loss).item()
    if final_nmse_db < -30:
        print(f"[PASS] {model_name} overfits successfully. NMSE = {final_nmse_db:.2f} dB")
    else:
        print(f"[FAIL] {model_name} cannot overfit. NMSE = {final_nmse_db:.2f} dB. There is a BUG.")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "st_mamba")
```

---

## 10. EVALUATION

### 10.1 Metrics

**File:** `src/evaluation/metrics.py`

```python
"""
Evaluation metrics for channel prediction.

1. NMSE per frame: NMSE at each of Q future frames
2. Sum-rate: achievable sum-rate using predicted CSI with ZF precoder
3. BER: bit error rate with predicted CSI
"""
import torch
import numpy as np
from typing import Dict


def compute_nmse_per_frame(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    Args:
        pred, true: (B, Q, ...) reshapable to (B, Q, D)
    Returns:
        nmse_per_frame: (Q,) NMSE for each future frame
    """
    B, Q = pred.shape[:2]
    pred_flat = pred.reshape(B, Q, -1)
    true_flat = true.reshape(B, Q, -1)
    numerator = (pred_flat - true_flat).pow(2).sum(dim=-1)  # (B, Q)
    denominator = true_flat.pow(2).sum(dim=-1).clamp(min=1e-12)  # (B, Q)
    nmse = (numerator / denominator).mean(dim=0)  # (Q,)
    return nmse


def nmse_to_db(nmse: torch.Tensor) -> torch.Tensor:
    return 10 * torch.log10(nmse.clamp(min=1e-12))


def compute_sum_rate(
    He_pred: torch.Tensor,  # (B, Q, N_RF, N_M, 2)
    He_true: torch.Tensor,  # (B, Q, N_RF, N_M, 2)
    snr_db: float = 20.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute achievable sum-rate using predicted CSI vs ideal CSI.

    Pipeline:
      1. Compute ZF precoder A from He_pred
      2. Apply A to true channel He_true (real channel exists, we just precoded wrong)
      3. Sum-rate = log2(det(I + (1/N_M σ²) A He_true^H He_true A^H))
    """
    B, Q, N_RF, N_M, _ = He_pred.shape

    # Convert to complex tensors
    He_pred_c = torch.complex(He_pred[..., 0], He_pred[..., 1])  # (B, Q, N_RF, N_M)
    He_true_c = torch.complex(He_true[..., 0], He_true[..., 1])

    snr_linear = 10 ** (snr_db / 10)
    sigma2 = 1.0 / snr_linear

    sum_rates_pred = []
    sum_rates_ideal = []

    for q in range(Q):
        H_pred = He_pred_c[:, q]  # (B, N_RF, N_M)
        H_true = He_true_c[:, q]  # (B, N_RF, N_M)

        for b in range(B):
            Hp = H_pred[b]  # (N_RF, N_M)
            Ht = H_true[b]

            # ZF precoder from predicted CSI: A_p = Hp^H (Hp Hp^H)^(-1)
            HpH = Hp.conj().transpose(-2, -1)  # (N_M, N_RF)
            try:
                inv_term_p = torch.linalg.inv(Hp @ HpH + 1e-6 * torch.eye(N_RF, dtype=Hp.dtype, device=Hp.device))
                A_p = HpH @ inv_term_p  # (N_M, N_RF)
            except Exception:
                continue
            # Normalize to total power 1
            A_p = A_p / (torch.norm(A_p) + 1e-12)

            # Effective channel with predicted precoder: H_eff = Ht @ A_p
            H_eff_p = Ht @ A_p  # (N_RF, N_RF)
            sum_rate_p = torch.log2(torch.det(
                torch.eye(N_RF, dtype=H_eff_p.dtype, device=H_eff_p.device) +
                (1.0 / (N_M * sigma2)) * H_eff_p @ H_eff_p.conj().transpose(-2, -1)
            ).real.clamp(min=1e-12))
            sum_rates_pred.append(sum_rate_p)

            # Ideal CSI: A_t = Ht^H (Ht Ht^H)^(-1), normalized
            HtH = Ht.conj().transpose(-2, -1)
            try:
                inv_term_t = torch.linalg.inv(Ht @ HtH + 1e-6 * torch.eye(N_RF, dtype=Ht.dtype, device=Ht.device))
                A_t = HtH @ inv_term_t
            except Exception:
                continue
            A_t = A_t / (torch.norm(A_t) + 1e-12)
            H_eff_t = Ht @ A_t
            sum_rate_t = torch.log2(torch.det(
                torch.eye(N_RF, dtype=H_eff_t.dtype, device=H_eff_t.device) +
                (1.0 / (N_M * sigma2)) * H_eff_t @ H_eff_t.conj().transpose(-2, -1)
            ).real.clamp(min=1e-12))
            sum_rates_ideal.append(sum_rate_t)

    return {
        'sum_rate_predicted': torch.stack(sum_rates_pred).mean(),
        'sum_rate_ideal':     torch.stack(sum_rates_ideal).mean(),
    }
```

### 10.2 Evaluator

**File:** `src/evaluation/evaluator.py`

```python
"""
End-to-end evaluation for a single model on a test set.
"""
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from src.evaluation.metrics import compute_nmse_per_frame, nmse_to_db, compute_sum_rate


class Evaluator:
    def __init__(self, model, device, normalizer, flatten_input: bool = False):
        self.model = model.to(device).eval()
        self.device = device
        self.normalizer = normalizer
        self.flatten_input = flatten_input

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, snr_db_list: List[float]) -> Dict:
        """
        Evaluate on test set.

        Returns dict with:
            nmse_per_frame_db: (Q,) NMSE in dB per future frame
            nmse_vs_snr: dict {snr: nmse_db} averaged over all frames
            sum_rate: avg achievable sum-rate
        """
        all_preds = []
        all_trues = []

        for batch in tqdm(loader, desc="Evaluating"):
            x = batch['history'].to(self.device)
            y = batch['future'].to(self.device)

            if self.flatten_input and x.dim() == 5:
                B, P = x.shape[0], x.shape[1]
                x_in = x.view(B, P, -1)
            else:
                x_in = x

            pred = self.model(x_in)

            # Reshape pred back to spatial form if baseline
            if self.flatten_input:
                pred = pred.view(B, -1, *y.shape[2:])

            all_preds.append(pred.cpu())
            all_trues.append(y.cpu())

        preds = torch.cat(all_preds, dim=0)  # (N, Q, N_RF, N_M, 2)
        trues = torch.cat(all_trues, dim=0)

        # Denormalize for sum-rate computation
        preds_denorm = self.normalizer.inverse_transform(preds.numpy())
        trues_denorm = self.normalizer.inverse_transform(trues.numpy())
        preds_denorm = torch.from_numpy(preds_denorm).to(self.device)
        trues_denorm = torch.from_numpy(trues_denorm).to(self.device)

        # NMSE per frame
        nmse_per_frame = compute_nmse_per_frame(preds, trues)
        nmse_per_frame_db = nmse_to_db(nmse_per_frame).numpy()

        # Average NMSE
        nmse_avg_db = float(nmse_to_db(nmse_per_frame.mean()).item())

        # Sum-rate at each SNR
        sum_rates = {}
        for snr in snr_db_list:
            sr = compute_sum_rate(preds_denorm, trues_denorm, snr_db=snr)
            sum_rates[snr] = {
                'predicted': float(sr['sum_rate_predicted'].item()),
                'ideal':     float(sr['sum_rate_ideal'].item()),
            }

        return {
            'nmse_per_frame_db': nmse_per_frame_db.tolist(),
            'nmse_avg_db': nmse_avg_db,
            'sum_rates': sum_rates,
        }
```

### 10.3 Evaluation script

**File:** `scripts/03_evaluate.py`

```python
"""
Evaluate one trained model on test set.

Usage:
    python scripts/03_evaluate.py --model st_mamba --exp st_mamba_full_seed42
"""
import argparse
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataset import ChannelDataset
from src.data.normalize import ChannelNormalizer
from src.models.utils import build_model
from src.evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--exp", type=str, required=True, help="Experiment name")
    parser.add_argument("--snr_list", type=str, default="10,12,14,16,18,20")
    parser.add_argument("--test_scenario", type=str, default="v30", choices=["v30", "v50"])
    args = parser.parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snr_list = [float(x) for x in args.snr_list.split(',')]

    # Load model
    model_cfg = load_config(f"configs/model/{args.model}.yaml")
    model = build_model(args.model, model_cfg)
    ckpt_path = Path(f"results/checkpoints/{args.exp}/checkpoint_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    # Load test data
    normalizer = ChannelNormalizer.load(Path("data/normalize_stats.npz"))
    flatten = (args.model != "st_mamba")
    test_ds = ChannelDataset(
        Path(f"data/test_{args.test_scenario}"),
        normalizer=normalizer,
        flatten_spatial=flatten,
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    # Evaluate
    evaluator = Evaluator(model, device, normalizer, flatten_input=flatten)
    results = evaluator.evaluate(test_loader, snr_db_list=snr_list)

    # Save results
    out_dir = Path(f"results/tables/{args.exp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"results_{args.test_scenario}.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"NMSE avg: {results['nmse_avg_db']:.2f} dB")
    print(f"NMSE per frame: {[f'{x:.2f}' for x in results['nmse_per_frame_db']]}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
```

---

## 11. ABLATION STUDIES

Để chứng minh ĐÓNG GÓP của các thành phần, cần 4 ablation:

| Ablation | Mô tả | Mục đích |
|---|---|---|
| A1: `st_mamba_no_spatial` | Thay SpatialEncoder bằng Linear, giữ Mamba | Chứng minh 2D CNN spatial encoder có giá trị |
| A2: `st_mamba_no_mamba` | Thay Mamba bằng LSTM cùng params, giữ spatial CNN | Chứng minh Mamba > LSTM ở temporal |
| A3: `st_mamba_no_phys_loss` | Train chỉ với NMSE loss (β=γ=0) | Chứng minh physics-informed loss có giá trị |
| A4: `st_mamba_no_curriculum` | Train ngẫu nhiên (không curriculum) | Chứng minh curriculum learning giúp |

**File:** `scripts/05_ablation.py`

```python
"""
Run all ablation experiments.

Each ablation modifies exactly ONE component of ST-Mamba and re-trains.
After training, evaluate and compile ablation table.

Usage:
    python scripts/05_ablation.py --run a1   # run ablation A1
    python scripts/05_ablation.py --run all  # run all
    python scripts/05_ablation.py --report   # compile table from existing runs
"""
import argparse
import subprocess
from pathlib import Path
import json


ABLATIONS = {
    'a1': {
        'name': 'st_mamba_no_spatial',
        'config_overrides': {'use_linear_spatial': True},
        'description': 'Replace SpatialEncoder with Linear projection',
    },
    'a2': {
        'name': 'st_mamba_no_mamba',
        'config_overrides': {'temporal_backbone': 'lstm'},
        'description': 'Replace Mamba with LSTM (same params)',
    },
    'a3': {
        'name': 'st_mamba_no_phys_loss',
        'train_overrides': {'beta': 0.0, 'gamma': 0.0},
        'description': 'NMSE-only loss (no phase, no temporal)',
    },
    'a4': {
        'name': 'st_mamba_no_curriculum',
        'train_config': 'full',  # use 'full' instead of 'curriculum'
        'description': 'No curriculum learning',
    },
}


def run_ablation(key: str):
    cfg = ABLATIONS[key]
    # Apply overrides → train → evaluate
    # (Detailed implementation in actual code)
    print(f"Running ablation {key}: {cfg['description']}")
    # ... actual subprocess calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None,
                        choices=[*ABLATIONS.keys(), "all"])
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if args.run == "all":
        for k in ABLATIONS:
            run_ablation(k)
    elif args.run:
        run_ablation(args.run)

    if args.report:
        # Compile JSON results into a table
        table = {}
        baseline = json.load(open("results/tables/st_mamba_full_seed42/results_v30.json"))
        table['st_mamba_full'] = baseline['nmse_avg_db']
        for k, cfg in ABLATIONS.items():
            f = Path(f"results/tables/{cfg['name']}_seed42/results_v30.json")
            if f.exists():
                table[cfg['name']] = json.load(open(f))['nmse_avg_db']
        for k, v in table.items():
            print(f"  {k:35s} | NMSE: {v:6.2f} dB")


if __name__ == "__main__":
    main()
```

---

## 12. SO SÁNH TOÀN BỘ MODEL

**File:** `scripts/04_compare_all.py`

```python
"""
Compare all models (ST-Mamba + 4 baselines) on both test scenarios.
Generate final plots and tables for thesis.

Usage:
    python scripts/04_compare_all.py
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


MODELS = ['st_mamba', 'rnn', 'lstm', 'gru', 'transformer']
MODEL_DISPLAY = {
    'st_mamba': 'ST-Mamba (Proposed)',
    'rnn': 'RNN',
    'lstm': 'LSTM',
    'gru': 'GRU',
    'transformer': 'Transformer',
}
COLORS = {
    'st_mamba': 'red', 'rnn': 'blue', 'lstm': 'green',
    'gru': 'orange', 'transformer': 'purple',
}
SCENARIOS = ['v30', 'v50']


def load_results():
    results = {}
    for m in MODELS:
        results[m] = {}
        for sc in SCENARIOS:
            f = Path(f"results/tables/{m}_full_seed42/results_{sc}.json")
            if f.exists():
                with open(f) as fp:
                    results[m][sc] = json.load(fp)
    return results


def plot_nmse_per_frame(results, scenario, save_path):
    plt.figure(figsize=(8, 5))
    for m in MODELS:
        if scenario in results.get(m, {}):
            nmse = results[m][scenario]['nmse_per_frame_db']
            frames = list(range(1, len(nmse) + 1))
            plt.plot(frames, nmse, marker='o', label=MODEL_DISPLAY[m], color=COLORS[m],
                     linewidth=2 if m == 'st_mamba' else 1)
    plt.xlabel("Predicted Frame Index (× 0.625 ms)")
    plt.ylabel("NMSE (dB)")
    plt.title(f"NMSE vs Predicted Frame — v = {scenario[1:]} km/h, SNR = 20 dB")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sum_rate(results, scenario, save_path):
    plt.figure(figsize=(8, 5))
    snrs = sorted([float(s) for s in next(iter(results.values()))[scenario]['sum_rates'].keys()])

    # Ideal sum-rate (reference)
    if 'st_mamba' in results and scenario in results['st_mamba']:
        ideal = [results['st_mamba'][scenario]['sum_rates'][str(s)]['ideal'] for s in snrs]
        plt.plot(snrs, ideal, 'k--', label='Ideal CSI', linewidth=2)

    for m in MODELS:
        if scenario in results.get(m, {}):
            sr = [results[m][scenario]['sum_rates'][str(s)]['predicted'] for s in snrs]
            plt.plot(snrs, sr, marker='o', label=MODEL_DISPLAY[m], color=COLORS[m],
                     linewidth=2 if m == 'st_mamba' else 1)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Sum-rate (bps/Hz)")
    plt.title(f"Sum-rate vs SNR — v = {scenario[1:]} km/h")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    results = load_results()
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    for sc in SCENARIOS:
        plot_nmse_per_frame(results, sc, out_dir / f"nmse_per_frame_{sc}.png")
        plot_sum_rate(results, sc, out_dir / f"sum_rate_{sc}.png")

    # Summary table
    print("\n=== NMSE Average (dB) ===")
    print(f"{'Model':25s} | {'v=30 km/h':12s} | {'v=50 km/h':12s}")
    print("-" * 55)
    for m in MODELS:
        v30 = results.get(m, {}).get('v30', {}).get('nmse_avg_db', float('nan'))
        v50 = results.get(m, {}).get('v50', {}).get('nmse_avg_db', float('nan'))
        print(f"{MODEL_DISPLAY[m]:25s} | {v30:10.2f}  | {v50:10.2f}")


if __name__ == "__main__":
    main()
```

---

## 13. KẾ HOẠCH TUẦN

### Tuần 1: Setup + Data + Baselines

**Ngày 1-2:**
- **CHẠY ĐẦU TIÊN:** `python scripts/00_migrate_legacy.py` để di chuyển file cũ vào `legacy/`
- Verify thư mục root sạch (chỉ còn app.py, requirements.txt, legacy/, PLAN.md, .git/, venv/)
- Setup environment (Python 3.10, PyTorch 2.x, mamba-ssm)
- Implement `src/data/channel_generator.py` với CDL-A 23-cluster
- Implement `src/data/dataset.py` và `src/data/normalize.py`
- Chạy `scripts/01_generate_data.py` → tạo 10000 samples
- **Checkpoint:** Visualize 1 sample, check Doppler spectrum đúng

**Ngày 3-4:**
- Implement baselines (RNN, LSTM, GRU, Transformer)
- Verify số params trong [280K, 420K]
- Run sanity check (`scripts/06_sanity_check.py`) cho từng baseline
- **Checkpoint:** Tất cả baseline overfit được 8 samples → NMSE < -30 dB

**Ngày 5-7:**
- Train 4 baselines với `configs/train/full.yaml`
- Mỗi baseline: ~3-4h trên GTX 1650
- Evaluate trên test_v30 và test_v50
- **Checkpoint:** Có bảng NMSE baseline để so sánh

### Tuần 2: ST-Mamba

**Ngày 8-10:**
- Implement `src/models/spatial_encoder.py`, `mamba_block.py`, `spatial_decoder.py`
- Test Mamba install (mamba-ssm). Nếu không được → dùng fallback
- Implement `src/models/st_mamba.py`
- Run sanity check cho ST-Mamba
- **Checkpoint:** ST-Mamba overfit 8 samples → NMSE < -30 dB

**Ngày 11-12:**
- Implement loss functions (NMSE + phase + temporal)
- Train ST-Mamba với `configs/train/full.yaml`
- Monitor loss curves, validation NMSE
- **CRITICAL CHECKPOINT NGÀY 12:** So sánh ST-Mamba val NMSE với baselines
  - Nếu ST-Mamba thắng ≥1 dB → tiếp tục curriculum
  - Nếu ST-Mamba trong 0.5 dB của baselines → tiếp tục với curriculum
  - Nếu ST-Mamba thua >2 dB → DỪNG, debug

**Ngày 13-14:**
- Implement curriculum scheduler
- Train ST-Mamba với curriculum
- **Checkpoint:** Curriculum ST-Mamba beats non-curriculum?

### Tuần 3: Evaluation + Ablation + Thesis

**Ngày 15-16:**
- Evaluate ST-Mamba trên cả test_v30 và test_v50
- Generate plots: NMSE vs frame, sum-rate vs SNR, complexity
- Run `scripts/04_compare_all.py`

**Ngày 17-18:**
- Chạy 4 ablation studies (A1-A4)
- Mỗi ablation: ~2h train + eval
- Compile ablation table

**Ngày 19-21:**
- Viết báo cáo
- Chuẩn bị slide bảo vệ
- Gradio demo (giữ app.py từ project cũ, update model loading)

---

## 14. RISK MANAGEMENT

### 14.1 Risk: Mamba không install được trên Windows

**Mitigation:**
- Bước 1: Thử `pip install mamba-ssm causal-conv1d`
- Bước 2: Nếu fail (Windows CUDA build issues), dùng fallback trong `mamba_block.py`
- Bước 3: Nếu fallback chậm quá, dùng SSM module từ `pip install s4-pytorch` hoặc implementation từ https://github.com/alxndrTL/mamba.py

### 14.2 Risk: GTX 1650 OOM khi train

**Mitigation:**
- Đã set `use_amp: true` → giảm memory 50%
- Đã set `batch_size: 32` (đủ nhỏ)
- Nếu vẫn OOM: giảm batch_size xuống 16 hoặc 8 (sẽ chậm hơn nhưng work)
- Nếu vẫn OOM: bật gradient checkpointing trong Mamba blocks

### 14.3 Risk: ST-Mamba train không converge

**Diagnostic checklist:**
1. Run sanity check trước → có overfit được không?
2. Check loss curves: có NaN/Inf không?
3. Check gradient norms: có explode không (>100)?
4. Reduce learning rate xuống 1e-4
5. Disable mixed precision (set `use_amp: false`)

### 14.4 Risk: ST-Mamba thua baselines

**Cách xử lý:**
- **KHÔNG ĐỔI ĐỀ TÀI hay đổi tên đồ án.**
- **KHÔNG ĐỔI sang kiến trúc khác.**
- Debug:
  1. Re-check sanity (có overfit được không)
  2. Try different `n_mamba_layers` (2, 4, 6, 8)
  3. Try different `d_state` (8, 16, 32)
  4. Try learning rate (1e-4, 3e-4, 1e-3)
- Nếu vẫn thua: accept kết quả, framing trong báo cáo:
  - "ST-Mamba achieves competitive performance with significantly lower computational complexity"
  - Focus vào complexity advantage + ablation insights
  - Đây vẫn là đóng góp khoa học (negative result + analysis)

### 14.5 Risk: Thời gian không đủ

**Priority order (làm theo thứ tự, nếu thiếu thời gian cắt từ cuối):**
1. Data generation + 1 baseline (LSTM) — must
2. ST-Mamba training + evaluation — must
3. 3 baselines còn lại — should
4. Curriculum learning — should
5. 4 ablations — should  
6. Sum-rate evaluation — nice to have
7. Multiple seeds (only 1 seed = 42) — nice to have
8. Gradio app update — nice to have

---

## 15. DEPENDENCIES

`requirements.txt`:

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
pyyaml>=6.0
tqdm>=4.65.0
tensorboard>=2.12.0
gradio>=4.0.0

# Mamba (try official first, fallback if Windows install fails)
# mamba-ssm>=2.0.0
# causal-conv1d>=1.2.0
```

`environment.yml` (conda):

```yaml
name: channel-pred
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pytorch=2.1.*
  - torchvision
  - pytorch-cuda=11.8
  - numpy
  - scipy
  - matplotlib
  - seaborn
  - pandas
  - pyyaml
  - tqdm
  - tensorboard
  - pip
  - pip:
    - gradio
    - mamba-ssm
    - causal-conv1d
```

---

## 16. UTILITY FILES

### 16.1 Config loader

**File:** `src/utils/config.py`

```python
import yaml
from pathlib import Path
from typing import Union

def load_config(path: Union[str, Path]) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def merge_configs(*configs: dict) -> dict:
    """Merge dicts, later configs override earlier."""
    out = {}
    for c in configs:
        out.update(c)
    return out
```

### 16.2 Seed

**File:** `src/utils/seed.py`

```python
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 16.3 Logger

**File:** `src/utils/logger.py`

```python
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

class Logger:
    def __init__(self, log_dir: Path):
        self.writer = SummaryWriter(log_dir)

    def log_metrics(self, metrics: dict, step: int):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)
        # Also print to console
        print(f"Step {step}: " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    def close(self):
        self.writer.close()
```

### 16.4 Model factory

**File:** `src/models/utils.py`

```python
"""Model factory."""
from src.models.st_mamba import STMamba
from src.models.baselines.rnn import RNNBaseline
from src.models.baselines.lstm import LSTMBaseline
from src.models.baselines.gru import GRUBaseline
from src.models.baselines.transformer import TransformerBaseline


def build_model(model_name: str, config: dict):
    name = model_name.lower()
    if name == 'st_mamba':
        return STMamba(**{k: v for k, v in config.items() if k != 'model_name'})
    elif name == 'rnn':
        return RNNBaseline(**{k: v for k, v in config.items() if k != 'model_name'})
    elif name == 'lstm':
        return LSTMBaseline(**{k: v for k, v in config.items() if k != 'model_name'})
    elif name == 'gru':
        return GRUBaseline(**{k: v for k, v in config.items() if k != 'model_name'})
    elif name == 'transformer':
        return TransformerBaseline(**{k: v for k, v in config.items() if k != 'model_name'})
    else:
        raise ValueError(f"Unknown model: {model_name}")
```

---

## 17. HƯỚNG DẪN CHẠY (cho README.md)

```bash
# 0. MIGRATION — chạy ĐẦU TIÊN, chỉ 1 lần duy nhất khi setup project
#    Di chuyển file cũ vào legacy/ (KHÔNG xóa)
python scripts/00_migrate_legacy.py

# 1. Setup environment
conda env create -f environment.yml
conda activate channel-pred

# 2. (Optional) Install Mamba
pip install mamba-ssm causal-conv1d

# 3. Generate dataset (~15-30 min)
python scripts/01_generate_data.py

# 4. Sanity check each model (~2 min each)
python scripts/06_sanity_check.py st_mamba
python scripts/06_sanity_check.py rnn
python scripts/06_sanity_check.py lstm
python scripts/06_sanity_check.py gru
python scripts/06_sanity_check.py transformer

# 5. Train all models (~15-20h total)
python scripts/02_train.py --model rnn --train_config full
python scripts/02_train.py --model lstm --train_config full
python scripts/02_train.py --model gru --train_config full
python scripts/02_train.py --model transformer --train_config full
python scripts/02_train.py --model st_mamba --train_config full
python scripts/02_train.py --model st_mamba --train_config curriculum

# 6. Evaluate
for m in rnn lstm gru transformer st_mamba; do
    python scripts/03_evaluate.py --model $m --exp ${m}_full_seed42 --test_scenario v30
    python scripts/03_evaluate.py --model $m --exp ${m}_full_seed42 --test_scenario v50
done

# 7. Compare all + generate plots
python scripts/04_compare_all.py

# 8. Run ablations
python scripts/05_ablation.py --run all
python scripts/05_ablation.py --report
```

---

## 18. TARGETS — Kỳ vọng kết quả

| Metric | Target |
|---|---|
| ST-Mamba NMSE @ v=30km/h, SNR=20dB | **≤ -15 dB** (giống IEICE paper) |
| ST-Mamba NMSE @ v=50km/h, SNR=20dB | **≤ -12 dB** |
| ST-Mamba so với best baseline | **Thắng ≥ 2 dB** trên ít nhất 1 scenario |
| Params (fair comparison) | Tất cả trong **[280K, 420K]** |
| Sum-rate ST-Mamba | **≥ 95% của ideal CSI** ở SNR=20dB |
| Inference latency | ST-Mamba **≤ Transformer** (Mamba có linear complexity) |

**Nếu ST-Mamba thua best baseline 0-2 dB:** vẫn OK cho đồ án, nhấn mạnh:
- Linear complexity advantage
- Lower latency
- Novel architecture (first Mamba for channel prediction)

**Nếu ST-Mamba thua >2 dB:** debug theo Risk 14.4, KHÔNG đổi đề tài.

---

## 19. CHECKLIST CUỐI CÙNG TRƯỚC KHI BẢO VỆ

- [ ] **Migration done:** Tất cả file cũ đã trong `legacy/`, root project sạch
- [ ] Data generation script chạy được, tạo đủ 10000 samples
- [ ] Normalize stats được save
- [ ] Tất cả 5 model (st_mamba, rnn, lstm, gru, transformer) có params trong [280K, 420K]
- [ ] Tất cả 5 model pass sanity check (overfit 8 samples)
- [ ] Tất cả 5 model train xong và có checkpoint best
- [ ] Evaluation results saved cho cả test_v30 và test_v50
- [ ] Final plots: NMSE per frame, sum-rate vs SNR (cả 2 scenario)
- [ ] Bảng so sánh NMSE
- [ ] 4 ablation experiments hoàn thành
- [ ] Bảng ablation
- [ ] Báo cáo viết đầy đủ 5 chương
- [ ] Slide bảo vệ
- [ ] Gradio app update với model mới
- [ ] Tất cả checkpoint commit lên GitHub
- [ ] README.md hướng dẫn rõ ràng
- [ ] **Sau khi mọi thứ verify OK:** xóa folder `legacy/` (manual, KHÔNG để Claude Code tự làm)

---

## 20. CÁC NGUYÊN TẮC ĐIỀU PHỐI KHI BẢO VỆ

**Câu hỏi có thể có và câu trả lời gợi ý:**

| Câu hỏi | Trả lời |
|---|---|
| Tại sao chọn Mamba? | Mamba có linear complexity O(n) so với Transformer O(n²). Mamba mới được publish 2024, chưa có công trình nào ứng dụng vào channel prediction — đây là khoảng trống nghiên cứu. |
| Tại sao tách spatial-temporal? | Vì anten kề nhau có channel tương quan (Kronecker structure). 2D CNN khai thác được structure này, flatten thì không. |
| Tại sao Transformer thắng/thua? | Trong setup này, sequence ngắn (P=30) → self-attention O(n²) không phát huy ưu thế. RNN family có inductive bias mạnh hơn cho Markov dynamics. |
| Tại sao P=30, Q=5? | Theo IEICE 2026 paper — đây là setup chuẩn cho mmWave channel prediction. |
| Tại sao 64×8×4 (NBS×NRF×NM)? | Theo bài báo IEICE — massive MIMO chuẩn với hybrid beamforming, NRF=8 để giảm complexity. |
| Đóng góp khoa học là gì? | (1) First Mamba for channel prediction. (2) Spatial-temporal decoupling. (3) Physics-informed loss. |
| Tại sao fair comparison ở 350K params? | Vì kích thước model ảnh hưởng performance độc lập với kiến trúc — phải fix params để so kiến trúc thuần túy. |
| Nếu model thua baseline thì sao? | Đồ án vẫn có đóng góp: linear complexity, lower latency, novel architecture exploration, ablation insights. Negative result + thorough analysis vẫn là đóng góp khoa học. |

---

## KẾT THÚC FILE PLAN.md

Tổng độ dài file này: ~ 2000 dòng đặc tả kỹ thuật. Mọi quyết định kiến trúc, hyperparameter, file structure, và code skeleton đều có ở đây.

**LƯU Ý CUỐI CÙNG CHO CLAUDE CODE:**
- Đọc cả file trước khi bắt đầu code
- Implement theo thứ tự: data → baselines → ST-Mamba → training → evaluation → ablation
- Mỗi file code phải có sanity test inline
- Mỗi model phải verify params trong fair budget [280K, 420K]
- Mọi quyết định lớn (đổi kiến trúc, đổi hyperparameter) phải DỪNG và hỏi
- Không có 1 chi tiết nào trong file này được phép bỏ qua hoặc làm khác đi