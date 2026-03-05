import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

# ── Import các module model ───────────────────────────────────────────────────
try:
    from fcn  import FCN
    from loss import HomoscedasticPoseLoss
except ImportError as e:
    sys.exit(
        f"[ERROR] Không tìm thấy module: {e}\n"
        "Hãy đảm bảo fcn.py, cbam.py, resblock.py, loss.py "
        "nằm cùng thư mục với train.py"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def get_config():
    parser = argparse.ArgumentParser(description="Train 5DoF Capsule Pose Model")

    # Đường dẫn dữ liệu — chỉnh lại cho đúng với Drive của bạn
    parser.add_argument("--voltage_dir",  type=str,
                        default="/content/drive/MyDrive/ROI_voltage",
                        help="Thư mục chứa ROI_voltage_1.csv … ROI_voltage_N.csv")
    parser.add_argument("--label_dir",    type=str,
                        default="/content/drive/MyDrive/ROI_data",
                        help="Thư mục chứa ROI_data_1.csv … ROI_data_N.csv")
    parser.add_argument("--ckpt_dir",     type=str,
                        default="/content/drive/MyDrive/capsule_ckpt",
                        help="Thư mục lưu checkpoint — nên đặt trên Drive")

    # Hyperparameters
    parser.add_argument("--num_files",    type=int,   default=20)
    parser.add_argument("--batch_size",   type=int,   default=256,
                        help="512 phù hợp với T4 16 GB + AMP")
    parser.add_argument("--num_epochs",   type=int,   default=200)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio",    type=float, default=0.1)
    parser.add_argument("--save_every",   type=int,   default=5,
                        help="Lưu checkpoint định kỳ mỗi N epoch")
    parser.add_argument("--patience",     type=int,   default=30,
                        help="Early stopping: dừng nếu val không giảm sau N epoch")
    parser.add_argument("--num_workers",  type=int,   default=2)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--csv_header",   action="store_true",
                        help="Thêm flag này nếu file CSV có dòng header")

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class CapsuleDataset(Dataset):
    """
    Đọc num_files cặp file CSV:
        ROI_voltage_i.csv  →  (N, 64)  Hall sensor voltages
        ROI_data_i.csv     →  (N,  5)  labels: x, y, z, roll, pitch

    Normalize bằng StandardScaler, reshape voltage → (N, 1, 8, 8).
    """

    def __init__(self, voltage_dir, label_dir, num_files,
                 has_header=False, volt_scaler=None, label_scaler=None):

        header = 0 if has_header else None
        voltages, labels = [], []

        for i in range(1, num_files + 1):
            v_path = os.path.join(voltage_dir, f"ROI_voltage_{i}.csv")
            l_path = os.path.join(label_dir,   f"ROI_data_{i}.csv")

            if not os.path.exists(v_path):
                raise FileNotFoundError(f"Không tìm thấy: {v_path}")
            if not os.path.exists(l_path):
                raise FileNotFoundError(f"Không tìm thấy: {l_path}")

            v_df = pd.read_csv(v_path, header=header)
            l_df = pd.read_csv(l_path, header=header)

            if len(v_df) != len(l_df):
                raise ValueError(
                    f"File {i}: voltage rows={len(v_df)}, label rows={len(l_df)}")
            if v_df.shape[1] != 64:
                raise ValueError(
                    f"File {i}: cần 64 voltage cols, nhận được {v_df.shape[1]}")
            if l_df.shape[1] != 5:
                raise ValueError(
                    f"File {i}: cần 5 label cols, nhận được {l_df.shape[1]}")

            voltages.append(v_df.values.astype(np.float32))
            labels.append(l_df.values.astype(np.float32))

        voltages = np.concatenate(voltages, axis=0)   # (N_total, 64)
        labels   = np.concatenate(labels,   axis=0)   # (N_total,  5)
        print(f"[Dataset] Tổng {len(voltages):,} samples từ {num_files} file pairs")

        # Fit scaler nếu chưa có (training), dùng lại nếu đã có (inference)
        if volt_scaler is None:
            self.volt_scaler  = StandardScaler().fit(voltages)
            self.label_scaler = StandardScaler().fit(labels)
        else:
            self.volt_scaler  = volt_scaler
            self.label_scaler = label_scaler

        voltages = self.volt_scaler.transform(voltages)
        labels   = self.label_scaler.transform(labels)

        # Reshape (N, 64) → (N, 1, 8, 8) để đưa vào Conv2d
        self.X = torch.tensor(voltages, dtype=torch.float32).reshape(-1, 1, 8, 8)
        self.Y = torch.tensor(labels,   dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path, epoch, model, criterion,
                    optimizer, scheduler, val_loss, best_val):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "criterion": criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "val_loss":  val_loss,
        "best_val":  best_val,
    }, path)


def load_checkpoint(path, model, criterion, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    # Xử lý prefix "_orig_mod." do torch.compile thêm vào state_dict
    def strip(state):
        return {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(strip(ckpt["model"]))
    criterion.load_state_dict(ckpt["criterion"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_val"]


def append_log(log_file, entry):
    """Ghi log từng epoch vào JSON (append, không ghi đè)."""
    log = []
    if os.path.exists(log_file):
        with open(log_file) as f:
            try:
                log = json.load(f)
            except json.JSONDecodeError:
                log = []
    log.append(entry)
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = get_config()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # ── In thông tin cấu hình ────────────────────────────────────────────────
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "N/A"
    print("=" * 62)
    print("  Capsule Robot 5DoF — Pose Estimation Training")
    print("=" * 62)
    print(f"  Device      : {device}  ({gpu_name})")
    print(f"  Epochs      : {cfg.num_epochs}  |  Batch : {cfg.batch_size}  |  LR : {cfg.lr}")
    print(f"  Voltage dir : {cfg.voltage_dir}")
    print(f"  Label dir   : {cfg.label_dir}")
    print(f"  Ckpt dir    : {cfg.ckpt_dir}")
    print("=" * 62 + "\n")

    # ── Dataset & DataLoader ─────────────────────────────────────────────────
    full_ds = CapsuleDataset(
        cfg.voltage_dir, cfg.label_dir, cfg.num_files,
        has_header=cfg.csv_header)

    n_total = len(full_ds)
    n_val   = max(1, int(cfg.val_ratio * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed))

    dl_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=True, drop_last=True, **dl_kwargs)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size,
        shuffle=False, **dl_kwargs)

    print(f"[Dataset] Train={n_train:,}  |  Val={n_val:,}\n")

    # Lưu scaler vào Drive để dùng lại khi inference
    scaler_path = os.path.join(cfg.ckpt_dir, "scalers.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump({"volt":  full_ds.volt_scaler,
                     "label": full_ds.label_scaler}, f)
    print(f"[Scaler] Saved → {scaler_path}\n")

    # ── Model, Loss, Optimizer, Scheduler ────────────────────────────────────
    model     = FCN(out_dim=5).to(device)
    criterion = HomoscedasticPoseLoss().to(device)
    all_params = list(model.parameters()) + list(criterion.parameters())

    optimizer = torch.optim.AdamW(
        all_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=1e-6)

    # ── Đường dẫn checkpoint & log ───────────────────────────────────────────
    ckpt_latest = os.path.join(cfg.ckpt_dir, "latest.pt")
    ckpt_best   = os.path.join(cfg.ckpt_dir, "best.pt")
    log_file    = os.path.join(cfg.ckpt_dir, "train_log.json")

    start_epoch = 1
    best_val    = float("inf")

    # ── Auto-resume nếu có checkpoint ────────────────────────────────────────
    if os.path.exists(ckpt_latest):
        print(f"[Resume] Phát hiện checkpoint → {ckpt_latest}")
        start_epoch, best_val = load_checkpoint(
            ckpt_latest, model, criterion, optimizer, scheduler, device)
        start_epoch += 1
        print(f"[Resume] Tiếp tục từ epoch {start_epoch}  |  best_val={best_val:.6f}\n")
    else:
        print("[Train] Không có checkpoint — bắt đầu từ đầu\n")

    # ── torch.compile (PyTorch >= 2.0 → ~15-20% faster trên T4) ─────────────
    try:
        model = torch.compile(model)
        print("[Compile] torch.compile enabled\n")
    except Exception:
        print("[Compile] torch.compile không khả dụng (PyTorch < 2.0), bỏ qua\n")

    use_amp    = (device == "cuda")
    amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Training loop ─────────────────────────────────────────────────────────
    no_improve = 0
    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  {'LR':>10}")
    print("-" * 46)

    for epoch in range(start_epoch, cfg.num_epochs + 1):

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_b, Y_b in train_loader:
            X_b = X_b.to(device, non_blocking=True)
            Y_b = Y_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(enabled=use_amp):
                pred = model(X_b)
                loss = criterion(pred, Y_b)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            train_loss += loss.item() * len(X_b)
        train_loss /= n_train
        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, Y_b in val_loader:
                X_b = X_b.to(device, non_blocking=True)
                Y_b = Y_b.to(device, non_blocking=True)
                with torch.amp.autocast(enabled=use_amp):
                    pred = model(X_b)
                    loss = criterion(pred, Y_b)
                val_loss += loss.item() * len(X_b)
        val_loss /= n_val

        lr_now = optimizer.param_groups[0]["lr"]
        marker = " *" if val_loss < best_val else ""
        print(f"{epoch:>6}/{cfg.num_epochs}  "
              f"{train_loss:>11.5f}  {val_loss:>10.5f}  "
              f"{lr_now:>10.2e}{marker}", flush=True)

        # Ghi log JSON — append mỗi epoch, không ghi đè
        append_log(log_file, {
            "epoch": epoch,
            "train": float(train_loss),
            "val":   float(val_loss),
            "lr":    float(lr_now),
        })

        # Lưu latest SAU MỖI EPOCH → resume an toàn khi Colab ngắt
        save_checkpoint(ckpt_latest, epoch, model, criterion,
                        optimizer, scheduler, val_loss, best_val)

        # Lưu best model
        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            save_checkpoint(ckpt_best, epoch, model, criterion,
                            optimizer, scheduler, val_loss, best_val)
            print(f"         → Best model saved  (val={best_val:.5f})")
        else:
            no_improve += 1

        # Lưu checkpoint định kỳ (backup thêm)
        if epoch % cfg.save_every == 0:
            periodic_path = os.path.join(cfg.ckpt_dir, f"epoch_{epoch:04d}.pt")
            save_checkpoint(periodic_path, epoch, model, criterion,
                            optimizer, scheduler, val_loss, best_val)

        # Early stopping
        if no_improve >= cfg.patience:
            print(f"\n[EarlyStop] Dừng tại epoch {epoch} "
                  f"— không cải thiện sau {cfg.patience} epochs")
            break

    print(f"\n{'=' * 62}")
    print(f"  Training hoàn tất!  Best val loss = {best_val:.6f}")
    print(f"  Checkpoints → {cfg.ckpt_dir}")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()