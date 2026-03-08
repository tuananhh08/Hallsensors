import os, sys, json, pickle, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fcn  import FCN
from loss import HuberPoseLoss


# CONFIG

def get_config():
    p = argparse.ArgumentParser()
    p.add_argument("--voltage_dir",   type=str, default="/content/drive/MyDrive/Hallsensors_data/ROI_voltage")
    p.add_argument("--label_dir",     type=str, default="/content/drive/MyDrive/Hallsensors_data/ROI_data")
    p.add_argument("--ckpt_dir",      type=str, default="/content/drive/MyDrive/capsule_ckpt")
    p.add_argument("--num_files",     type=int,   default=20)
    p.add_argument("--val_files",     type=int,   default=4,
                   help="So file cuoi dung lam val (default=4 tuc 4 file random)")
    p.add_argument("--batch_size",    type=int,   default=512)
    p.add_argument("--num_epochs",    type=int,   default=200)
    p.add_argument("--lr",            type=float, default=5e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-3)
    p.add_argument("--ang_weight",    type=float, default=0.006,
                   help="He so Huber(cos_pitch, cos_yaw) so voi Huber(xyz)")
    p.add_argument("--delta_xyz",     type=float, default=0.005,
                   help="Nguong Huber cho xyz (m), default=0.005 (0.5cm)")
    p.add_argument("--delta_ang",     type=float, default=0.10,
                   help="Nguong Huber cho cos angle, default=0.10 (~6 do)")
    p.add_argument("--warmup_epochs", type=int,   default=5)
    p.add_argument("--save_every",    type=int,   default=5)
    p.add_argument("--patience",      type=int,   default=20)
    p.add_argument("--num_workers",   type=int,   default=2)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


# DATASET

def _auto_read_csv(path):
    """Doc CSV, tu dong detect header, ep float, drop hang NaN."""
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

    def __init__(self, voltage_path, label_path, volt_scaler, label_scaler):
        v_df = _auto_read_csv(voltage_path)
        l_df = _auto_read_csv(label_path)

        if abs(len(v_df) - len(l_df)) > 1:
            raise ValueError(
                f"Lech hang: voltage={len(v_df)}, label={len(l_df)} | {voltage_path}")
        min_rows = min(len(v_df), len(l_df))
        v_df = v_df.iloc[:min_rows]
        l_df = l_df.iloc[:min_rows]

        if v_df.shape[1] != 64:
            raise ValueError(f"Can 64 voltage cols, co {v_df.shape[1]} | {voltage_path}")
        if l_df.shape[1] != 5:
            raise ValueError(f"Can 5 label cols, co {l_df.shape[1]} | {label_path}")

        voltages = volt_scaler.transform(v_df.values.astype(np.float32))
        labels   = label_scaler.transform(l_df.values.astype(np.float32))

        self.X = torch.tensor(voltages, dtype=torch.float32).reshape(-1, 1, 8, 8)
        self.Y = torch.tensor(labels,   dtype=torch.float32)
        self.n = min_rows

    def __len__(self):          return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


def build_datasets(voltage_dir, label_dir, num_files, val_files,
                   scaler_file, cache_dir, seed=42):

    n_train_files = num_files - val_files
    os.makedirs(cache_dir, exist_ok=True)

    # Random shuffle file indices theo seed 
    all_ids = list(range(1, num_files + 1))
    rng = np.random.default_rng(seed)
    rng.shuffle(all_ids)
    train_ids = sorted(all_ids[:n_train_files])
    val_ids   = sorted(all_ids[n_train_files:])

    # Luu / load split info
    split_path = os.path.join(cache_dir, "split_info.json")
    if not os.path.exists(split_path):
        with open(split_path, "w") as f:
            json.dump({"train_files": train_ids,
                       "val_files":   val_ids,
                       "seed":        seed}, f, indent=2)
        print(f"  Split info saved -> {split_path}")
    else:
        with open(split_path) as f:
            saved     = json.load(f)
        train_ids = saved["train_files"]
        val_ids   = saved["val_files"]

    print(f"  Train files: {train_ids}")
    print(f"  Val   files: {val_ids}")

    # ── Fit scaler CHI tren tap train 
    if os.path.exists(scaler_file):
        print(f"  Loading scalers from {scaler_file}")
        with open(scaler_file, "rb") as f:
            scalers = pickle.load(f)
        volt_scaler  = scalers["volt"]
        label_scaler = scalers["label"]
    else:
        print("  Fitting scalers on train files ...")
        all_v, all_l = [], []
        for i in train_ids:
            v_df = _auto_read_csv(os.path.join(voltage_dir, f"ROI_voltage_{i}.csv"))
            l_df = _auto_read_csv(os.path.join(label_dir,   f"ROI_data_{i}.csv"))
            min_rows = min(len(v_df), len(l_df))
            all_v.append(v_df.iloc[:min_rows].values.astype(np.float32))
            all_l.append(l_df.iloc[:min_rows].values.astype(np.float32))
        # Voltage: MinMaxScaler tranh NaN do std qua nho
        volt_scaler  = MinMaxScaler(feature_range=(0, 1)).fit(np.concatenate(all_v))
        # Label: StandardScaler vi cac cot co scale khac nhau
        label_scaler = StandardScaler().fit(np.concatenate(all_l))
        with open(scaler_file, "wb") as f:
            pickle.dump({"volt": volt_scaler, "label": label_scaler}, f)
        print(f"  Scalers saved -> {scaler_file}")

    # ── Load / cache tung file 
    def load_file(i):
        cache_path = os.path.join(cache_dir, f"file_{i:02d}.pt")
        if os.path.exists(cache_path):
            c  = torch.load(cache_path, weights_only=True)
            ds = CapsuleDataset.__new__(CapsuleDataset)
            ds.X = c["X"]
            ds.Y = c["Y"]
            ds.n = len(ds.X)
            return ds
        v_path = os.path.join(voltage_dir, f"ROI_voltage_{i}.csv")
        l_path = os.path.join(label_dir,   f"ROI_data_{i}.csv")
        ds = CapsuleDataset(v_path, l_path, volt_scaler, label_scaler)
        torch.save({"X": ds.X, "Y": ds.Y}, cache_path)
        return ds

    print(f"\n  Loading train files ...")
    train_datasets = []
    for i in train_ids:
        ds = load_file(i)
        print(f"    File {i:>2}: {ds.n:>7,} samples")
        train_datasets.append(ds)

    print(f"\n  Loading val files ...")
    val_datasets = []
    for i in val_ids:
        ds = load_file(i)
        print(f"    File {i:>2}: {ds.n:>7,} samples")
        val_datasets.append(ds)

    train_ds = ConcatDataset(train_datasets)
    val_ds   = ConcatDataset(val_datasets)
    print(f"\n  Total -> train: {len(train_ds):,}  val: {len(val_ds):,}\n")
    return train_ds, val_ds, volt_scaler, label_scaler

# CHECKPOINT HELPERS

def save_checkpoint(path, epoch, model, optimizer, scheduler, val_loss, best_val):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "val_loss":  val_loss,
        "best_val":  best_val,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt        = torch.load(path, map_location=device, weights_only=False)
    raw_state   = ckpt["model"]
    is_compiled = hasattr(model, "_orig_mod")
    if is_compiled:
        state = (raw_state if any(k.startswith("_orig_mod.") for k in raw_state)
                 else {"_orig_mod." + k: v for k, v in raw_state.items()})
    else:
        state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
    model.load_state_dict(state)
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


# MAIN

def main():
    cfg    = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 65)
    print("  5DoF Capsule Robot - Pose Estimation Training")
    print("=" * 65)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"  Device      : {device} ({gpu_name})")
    print(f"  Epochs      : {cfg.num_epochs}  |  Batch : {cfg.batch_size}  |  LR: {cfg.lr}")
    print(f"  Loss        : HuberPoseLoss  ang_weight={cfg.ang_weight}"
          f"  delta_xyz={cfg.delta_xyz}  delta_ang={cfg.delta_ang}")
    print(f"  Weight decay: {cfg.weight_decay}  |  Warmup: {cfg.warmup_epochs} epochs")
    print(f"  Split       : {cfg.num_files - cfg.val_files} train / "
          f"{cfg.val_files} val files  (random seed={cfg.seed})")
    print(f"  Ckpt dir    : {cfg.ckpt_dir}")
    print("=" * 65 + "\n")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    ckpt_latest = os.path.join(cfg.ckpt_dir, "latest.pt")
    ckpt_best   = os.path.join(cfg.ckpt_dir, "best.pt")
    log_file    = os.path.join(cfg.ckpt_dir, "train_log.json")
    scaler_file = os.path.join(cfg.ckpt_dir, "scalers.pkl")
    cache_dir   = os.path.join(cfg.ckpt_dir, "file_cache")

    # ── Dataset 
    print("Loading dataset ...")
    train_ds, val_ds, volt_scaler, label_scaler = build_datasets(
        cfg.voltage_dir, cfg.label_dir,
        cfg.num_files, cfg.val_files,
        scaler_file, cache_dir, seed=cfg.seed)

    n_train = len(train_ds)
    n_val   = len(val_ds)

    # ── DataLoader ────────────────────────────────────────────────────────────
    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin, drop_last=True,
        persistent_workers=(cfg.num_workers > 0))
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin,
        persistent_workers=(cfg.num_workers > 0))

    # ── Model + Loss ──────────────────────────────────────────────────────────
    model     = FCN(out_dim=5).to(device)
    criterion = HuberPoseLoss(
        ang_weight=cfg.ang_weight,
        delta_xyz=cfg.delta_xyz,
        delta_ang=cfg.delta_ang)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Warmup -> CosineAnnealingLR
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=cfg.warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs - cfg.warmup_epochs, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.warmup_epochs])

    # ── torch.compile TRUOC resume ────────────────────────────────────────────
    try:
        model = torch.compile(model)
        print("torch.compile enabled")
    except Exception:
        print("torch.compile not available - skipping")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch, best_val = 1, float("inf")
    if os.path.exists(ckpt_latest):
        print(f"Resuming from {ckpt_latest} ...")
        start_epoch, best_val = load_checkpoint(
            ckpt_latest, model, optimizer, scheduler, device)
        start_epoch += 1
        print(f"  -> Epoch {start_epoch}  best_val={best_val:.6f}\n")
    else:
        print("Training from scratch\n")

    # ── AMP ───────────────────────────────────────────────────────────────────
    use_amp    = (device.type == "cuda")
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Training loop ─────────────────────────────────────────────────────────
    no_improve = 0
    hdr = (f"{'Epoch':>6}  {'Train':>9}  {'Val':>9}  "
           f"{'Huber_xyz':>10}  {'Huber_ang':>10}  {'LR':>8}  {'Time':>7}")
    print(hdr)
    print("-" * len(hdr))

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for X_b, Y_b in train_loader:
            X_b = X_b.to(device, non_blocking=True)
            Y_b = Y_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, _, _ = criterion(model(X_b), Y_b)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            train_loss += loss.item() * len(X_b)
        train_loss /= n_train
        scheduler.step()

        # Validate
        model.eval()
        val_loss = val_xyz = val_ang = 0.0
        with torch.no_grad():
            for X_b, Y_b in val_loader:
                X_b = X_b.to(device, non_blocking=True)
                Y_b = Y_b.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss, loss_xyz, loss_ang = criterion(model(X_b), Y_b)
                n = len(X_b)
                val_loss += loss.item()     * n
                val_xyz  += loss_xyz.item() * n
                val_ang  += loss_ang.item() * n
        val_loss /= n_val
        val_xyz  /= n_val
        val_ang  /= n_val

        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"{epoch:>6}  {train_loss:>9.5f}  {val_loss:>9.5f}  "
              f"{val_xyz:>10.5f}  {val_ang:>10.5f}  "
              f"{lr_now:>8.2e}  {elapsed:>6.1f}s", flush=True)

        append_log(log_file, {
            "epoch": epoch, "train": train_loss, "val": val_loss,
            "val_xyz": val_xyz, "val_ang": val_ang, "lr": lr_now})

        # Save latest moi epoch (an toan khi Colab ngat)
        save_checkpoint(ckpt_latest, epoch, model, optimizer, scheduler,
                        val_loss, best_val)

        # Save best
        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            save_checkpoint(ckpt_best, epoch, model, optimizer, scheduler,
                            val_loss, best_val)
            print(f"         >> Best saved  val={best_val:.6f} "
                  f"(xyz={val_xyz:.5f}  ang={val_ang:.5f})", flush=True)
        else:
            no_improve += 1

        # Save dinh ky
        if epoch % cfg.save_every == 0:
            save_checkpoint(
                os.path.join(cfg.ckpt_dir, f"epoch_{epoch:04d}.pt"),
                epoch, model, optimizer, scheduler, val_loss, best_val)

        # Early stopping
        if no_improve >= cfg.patience:
            print(f"\nEarly stopping (no improvement for {cfg.patience} epochs)")
            break

    print(f"\nDone! Best val loss = {best_val:.6f}")
    print(f"Checkpoints -> {cfg.ckpt_dir}")


if __name__ == "__main__":
    main()