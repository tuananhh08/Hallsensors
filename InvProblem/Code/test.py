import argparse, sys, os, pickle, platform
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ─── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--voltage",  default="Helical_calib_data.csv")
parser.add_argument("--coords",   default="Hall_points_coordinates.csv")
parser.add_argument("--ckpt_dir", default="./ckpt")
parser.add_argument("--code_dir", default=".")
parser.add_argument("--out",      default="helical_result.png")
args = parser.parse_args()

# ─── Import model ─────────────────────────────────────────────────────────────
sys.path.insert(0, args.code_dir)
from fcn import FCN  # noqa: E402

# ─── Load data ────────────────────────────────────────────────────────────────
def _read(path):
    """Doc CSV tu dong detect header, giu nguyen toan bo du lieu."""
    df = pd.read_csv(path, header=None)
    try:
        df.iloc[0].astype(float)
        has_header = False
    except (ValueError, TypeError):
        has_header = True
    if has_header:
        df = pd.read_csv(path, header=0)
    return df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

print("Loading data...")
volt_df  = _read(args.voltage)
coord_df = pd.read_csv(args.coords)

voltages = volt_df.values.astype(np.float32)    # (N, 64)
gt_xyz   = coord_df[["x", "y", "z"]].values     # (N, 3)

print(f"  Voltage : {voltages.shape}")
print(f"  GT xyz  : {gt_xyz.shape}")

# ─── Load scalers ─────────────────────────────────────────────────────────────
scaler_path = os.path.join(args.ckpt_dir, "scalers.pkl")
print(f"Loading scalers from {scaler_path} ...")
with open(scaler_path, "rb") as f:
    scalers = pickle.load(f)
volt_scaler  = scalers["volt"]
label_scaler = scalers["label"]

# ─── Preprocess voltage ───────────────────────────────────────────────────────
volt_scaled = volt_scaler.transform(voltages)
volt_tensor = torch.tensor(volt_scaled, dtype=torch.float32).view(-1, 1, 8, 8)

# ─── Build model ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = FCN(out_dim=5).to(device)

# torch.compile chi tren Linux (Colab), bo qua tren Windows
if platform.system() != "Windows":
    try:
        model = torch.compile(model)
        print("torch.compile enabled")
    except Exception:
        print("torch.compile not available - skipping")
else:
    print("torch.compile disabled (Windows)")

# ─── Load checkpoint ──────────────────────────────────────────────────────────
ckpt_path = os.path.join(args.ckpt_dir, "best.pt")
print(f"Loading checkpoint from {ckpt_path} ...")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

raw_state   = ckpt["model"]
is_compiled = hasattr(model, "_orig_mod")
if is_compiled:
    state = (raw_state if any(k.startswith("_orig_mod.") for k in raw_state)
             else {"_orig_mod." + k: v for k, v in raw_state.items()})
else:
    state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}

model.load_state_dict(state)
model.eval()
print(f"  Checkpoint epoch : {ckpt.get('epoch', '?')}")
print(f"  Best val loss    : {ckpt.get('best_val', 0):.6f}")

# ─── Inference ────────────────────────────────────────────────────────────────
print("Running inference...")
with torch.no_grad():
    pred_scaled = model(volt_tensor.to(device)).cpu().numpy()   # (N, 5)

pred_full = label_scaler.inverse_transform(pred_scaled)         # (N, 5)
pred_xyz  = pred_full[:, :3]                                    # (N, 3)

print("\n  Sample pred (first 3 points):")
for i in range(min(3, len(pred_xyz))):
    print(f"    [{i}] pred=({pred_xyz[i,0]:.4f}, {pred_xyz[i,1]:.4f}, {pred_xyz[i,2]:.4f})"
          f"  gt=({gt_xyz[i,0]:.4f}, {gt_xyz[i,1]:.4f}, {gt_xyz[i,2]:.4f})")

# ─── Metrics ──────────────────────────────────────────────────────────────────
errors   = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
mae_xyz  = np.abs(pred_xyz - gt_xyz).mean(axis=0)
rmse     = np.sqrt(np.mean(errors ** 2))
mean_err = errors.mean()
max_err  = errors.max()

print("\n─── Kết quả ────────────────────────────────────────")
print(f"  Mean Euclidean error : {mean_err * 1000:.2f} mm")
print(f"  RMSE                 : {rmse     * 1000:.2f} mm")
print(f"  Max error            : {max_err  * 1000:.2f} mm")
print(f"  MAE x                : {mae_xyz[0] * 1000:.2f} mm")
print(f"  MAE y                : {mae_xyz[1] * 1000:.2f} mm")
print(f"  MAE z                : {mae_xyz[2] * 1000:.2f} mm")
print("────────────────────────────────────────────────────\n")

# ─── Visualize ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection="3d")

ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2],
        color="blue", linewidth=2, label="Ground Truth")
ax.scatter(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2],
           color="blue", s=30, zorder=5)

ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], linestyle = '--',
        color="red", linewidth=2, label="Our approach")
ax.scatter(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2],
           color="red", s=30, zorder=5)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title(
    f"Helical Trajectory — Model Inference\n"
    f"Mean err: {mean_err*1000:.2f} mm  |  "
    f"RMSE: {rmse*1000:.2f} mm  |  "
    f"Max: {max_err*1000:.2f} mm",
    fontsize=11
)
ax.legend(fontsize=10)
ax.grid(True)

plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"Saved: {args.out}")
plt.show()