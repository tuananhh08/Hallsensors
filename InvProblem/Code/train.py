import os, sys, json, pickle, argparse, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fcn  import FCN
from loss import HomoscedasticPoseLoss


# =============================================================================
# CONFIG
# =============================================================================

def get_config():
    p = argparse.ArgumentParser()
    p.add_argument("--voltage_dir",  type=str, default="/content/drive/MyDrive/Hallsensors_data/ROI_voltage")
    p.add_argument("--label_dir",    type=str, default="/content/drive/MyDrive/Hallsensors_data/ROI_data")
    p.add_argument("--ckpt_dir",     type=str, default="/content/drive/MyDrive/capsule_ckpt")
    p.add_argument("--num_files",    type=int,   default=20)
    p.add_argument("--batch_size",   type=int,   default=1024)
    p.add_argument("--num_epochs",   type=int,   default=200)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio",    type=float, default=0.1)
    p.add_argument("--save_every",   type=int,   default=5)
    p.add_argument("--patience",     type=int,   default=30)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


# =============================================================================
# DATASET
# =============================================================================

def _auto_read_csv(path):

    df = pd.read_csv(path, header=None, low_memory=False)
    try:
        df.iloc[0].astype(float)
        has_header = False
    except (ValueError, TypeError):
        has_header = True

    if has_header:
        df = pd.read_csv(path, header=0, low_memory=False)

    df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
    return df


class CapsuleDataset(Dataset):
    def __init__(self, voltage_dir, label_dir, num_files,
                 volt_scaler=None, label_scaler=None):
        voltages, labels = [], []

        for i in range(1, num_files + 1):
            v_path = os.path.join(voltage_dir, f"ROI_voltage_{i}.csv")
            l_path = os.path.join(label_dir,   f"ROI_data_{i}.csv")

            if not os.path.exists(v_path):
                raise FileNotFoundError(f"Khong tim thay: {v_path}")
            if not os.path.exists(l_path):
                raise FileNotFoundError(f"Khong tim thay: {l_path}")

            v_df = _auto_read_csv(v_path)
            l_df = _auto_read_csv(l_path)

            # Can bang so hang neu lech <= 1 (header thua)
            if abs(len(v_df) - len(l_df)) > 1:
                raise ValueError(
                    f"File {i}: lech nhieu qua — voltage={len(v_df)}, label={len(l_df)}")
            min_rows = min(len(v_df), len(l_df))
            v_df = v_df.iloc[:min_rows]
            l_df = l_df.iloc[:min_rows]

            if v_df.shape[1] != 64:
                raise ValueError(f"File {i}: can 64 voltage cols, co {v_df.shape[1]}")
            if l_df.shape[1] != 5:
                raise ValueError(f"File {i}: can 5 label cols, co {l_df.shape[1]}")

            voltages.append(v_df.values.astype(np.float32))
            labels.append(l_df.values.astype(np.float32))
            print(f"  File {i:>2}: {min_rows:>7,} samples")

        voltages = np.concatenate(voltages, axis=0)
        labels   = np.concatenate(labels,   axis=0)
        print(f"  Tong: {len(voltages):,} samples\n")

        if volt_scaler is None:
            self.volt_scaler  = StandardScaler().fit(voltages)
            self.label_scaler = StandardScaler().fit(labels)
        else:
            self.volt_scaler  = volt_scaler
            self.label_scaler = label_scaler

        voltages = self.volt_scaler.transform(voltages)
        labels   = self.label_scaler.transform(labels)

        self.X = torch.tensor(voltages, dtype=torch.float32).reshape(-1, 1, 8, 8)
        self.Y = torch.tensor(labels,   dtype=torch.float32)

    def __len__(self):          return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def save_checkpoint(path, epoch, model, criterion, optimizer, scheduler,
                    val_loss, best_val):
    torch.save({
        "epoch": epoch, "model": model.state_dict(),
        "criterion": criterion.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "val_loss": val_loss, "best_val": best_val,
    }, path)


def load_checkpoint(path, model, criterion, optimizer, scheduler, device):
    ckpt  = torch.load(path, map_location=device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    criterion.load_state_dict(ckpt["criterion"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_val"]


def append_log(log_file, entry):
    log = []
    if os.path.exists(log_file):
        with open(log_file) as f:
            try:   log = json.load(f)
            except json.JSONDecodeError: log = []
    log.append(entry)
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg    = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  5DoF Capsule Robot - Pose Estimation Training")
    print("=" * 60)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"  Device : {device} ({gpu_name})")
    print(f"  Epochs : {cfg.num_epochs}  Batch: {cfg.batch_size}  LR: {cfg.lr}")
    print(f"  Ckpt   : {cfg.ckpt_dir}")
    print("=" * 60 + "\n")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    ckpt_latest = os.path.join(cfg.ckpt_dir, "latest.pt")
    ckpt_best   = os.path.join(cfg.ckpt_dir, "best.pt")
    log_file    = os.path.join(cfg.ckpt_dir, "train_log.json")
    scaler_file = os.path.join(cfg.ckpt_dir, "scalers.pkl")

    # Dataset
    print("Loading dataset ...")
    full_ds = CapsuleDataset(cfg.voltage_dir, cfg.label_dir, cfg.num_files)
    with open(scaler_file, "wb") as f:
        pickle.dump({"volt": full_ds.volt_scaler, "label": full_ds.label_scaler}, f)
    print(f"Scalers saved -> {scaler_file}")

    n_total = len(full_ds)
    n_val   = max(1, int(cfg.val_ratio * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed))
    print(f"Split -> train: {n_train:,}  val: {n_val:,}\n")

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin, drop_last=True,
        persistent_workers=(cfg.num_workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin,
        persistent_workers=(cfg.num_workers > 0))

    # Model
    model      = FCN(out_dim=5).to(device)
    criterion  = HomoscedasticPoseLoss().to(device)
    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer  = torch.optim.AdamW(all_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=1e-6)

    try:
        model = torch.compile(model)
        print("torch.compile enabled")
    except Exception:
        print("torch.compile not available - skipping")

    # Resume
    start_epoch, best_val = 1, float("inf")
    if os.path.exists(ckpt_latest):
        print(f"Resuming from {ckpt_latest} ...")
        start_epoch, best_val = load_checkpoint(
            ckpt_latest, model, criterion, optimizer, scheduler, device)
        start_epoch += 1
        print(f"  -> Epoch {start_epoch}  best_val={best_val:.6f}\n")
    else:
        print("Training from scratch\n")

    use_amp    = (device.type == "cuda")
    amp_scaler = torch.amp.GradScaler(enabled=use_amp)
    no_improve = 0

    print(f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'LR':>10}  {'Time':>7}")
    print("-" * 52)

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for X_b, Y_b in train_loader:
            X_b = X_b.to(device, non_blocking=True)
            Y_b = Y_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(enabled=use_amp):
                loss = criterion(model(X_b), Y_b)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            train_loss += loss.item() * len(X_b)
        train_loss /= n_train
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, Y_b in val_loader:
                X_b = X_b.to(device, non_blocking=True)
                Y_b = Y_b.to(device, non_blocking=True)
                with torch.cuda.autocast(enabled=use_amp):
                    val_loss += criterion(model(X_b), Y_b).item() * len(X_b)
        val_loss /= n_val

        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"{epoch:>6}  {train_loss:>10.5f}  {val_loss:>10.5f}  "
              f"{lr_now:>10.2e}  {elapsed:>6.1f}s", flush=True)

        append_log(log_file, {"epoch": epoch, "train": train_loss,
                               "val": val_loss, "lr": lr_now})

        # Save latest (moi epoch - an toan khi Colab ngat)
        save_checkpoint(ckpt_latest, epoch, model, criterion,
                        optimizer, scheduler, val_loss, best_val)

        # Save best
        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            save_checkpoint(ckpt_best, epoch, model, criterion,
                            optimizer, scheduler, val_loss, best_val)
            print(f"         >> Best saved (val={best_val:.6f})", flush=True)
        else:
            no_improve += 1

        # Save dinh ky
        if epoch % cfg.save_every == 0:
            save_checkpoint(os.path.join(cfg.ckpt_dir, f"epoch_{epoch:04d}.pt"),
                            epoch, model, criterion, optimizer, scheduler,
                            val_loss, best_val)

        # Early stopping
        if no_improve >= cfg.patience:
            print(f"\nEarly stopping (no improvement for {cfg.patience} epochs)")
            break

    print(f"\nDone! Best val loss = {best_val:.6f}")
    print(f"Checkpoints -> {cfg.ckpt_dir}")


if __name__ == "__main__":
    main()
