import argparse
import sys
import os
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ─── Args ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--voltage",  default="Helical_voltage.csv")
parser.add_argument("--coords",   default="Helical_points_coordinates.csv")
parser.add_argument("--ckpt_dir", default="/content/drive/MyDrive/capsule_ckpt")
parser.add_argument("--code_dir", default="/content/Hallsensors/InvProblem/Code")
parser.add_argument("--out",      default="helical_result.png")
args = parser.parse_args()

# ─── Import model ────────────────────────────────────────────────────────────

sys.path.insert(0, args.code_dir)
from fcn import FCN  # noqa: E402

# ─── Load data ───────────────────────────────────────────────────────────────

print("Loading data...")
volt_df  = pd.read_csv(args.voltage, header=None)
coord_df = pd.read_csv(args.coords)

voltages = volt_df.values.astype(np.float32)    # (25, 64)
gt_xyz   = coord_df[["x", "y", "z"]].values     # (25, 3)

print(f"  Voltage shape : {voltages.shape}")
print(f"  GT coords     : {gt_xyz.shape}")

# ─── Load scalers ────────────────────────────────────────────────────────────

scaler_path = os.path.join(args.ckpt_dir, "scalers.pkl")
print(f"Loading scalers from {scaler_path} ...")
with open(scaler_path, "rb") as f:
    scalers = pickle.load(f)

volt_scaler  = scalers["volt"]    # MinMaxScaler  (fit trên 64 voltage cols)
label_scaler = scalers["label"]   # StandardScaler (fit trên 5 label cols)

# ─── Preprocess voltage ──────────────────────────────────────────────────────

volt_scaled = volt_scaler.transform(voltages)              # (25, 64)
volt_tensor = torch.tensor(volt_scaled, dtype=torch.float32).view(-1, 1, 8, 8)

# ─── Build model ─────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# out_dim=5 vì model train với [x, y, z, cos_alpha, cos_beta]
model = FCN(out_dim=5).to(device)

# torch.compile TRƯỚC load — giống hệt train.py
try:
    model = torch.compile(model)
    print("torch.compile enabled")
except Exception:
    print("torch.compile not available - skipping")

# ─── Load checkpoint ─────────────────────────────────────────────────────────

ckpt_path = os.path.join(args.ckpt_dir, "best.pt")
print(f"Loading checkpoint from {ckpt_path} ...")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

# Xử lý _orig_mod. prefix — copy từ load_checkpoint() trong train.py
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

# ─── Inference ───────────────────────────────────────────────────────────────

print("Running inference...")
with torch.no_grad():
    pred_scaled = model(volt_tensor.to(device)).cpu().numpy()  # (25, 5) — scaled space

# Inverse transform toàn bộ 5 cột về đơn vị thực
pred_full = label_scaler.inverse_transform(pred_scaled)        # (25, 5)
pred_xyz  = pred_full[:, :3]                                   # (25, 3) — chỉ lấy xyz

print(f"\n  Sample pred (first 3 points):")
for i in range(min(3, len(pred_xyz))):
    print(f"    [{i}] pred=({pred_xyz[i,0]:.4f}, {pred_xyz[i,1]:.4f}, {pred_xyz[i,2]:.4f})"
          f"  gt=({gt_xyz[i,0]:.4f}, {gt_xyz[i,1]:.4f}, {gt_xyz[i,2]:.4f})")

# ─── Metrics ─────────────────────────────────────────────────────────────────

errors   = np.linalg.norm(pred_xyz - gt_xyz, axis=1)  # Euclidean error per point (m)
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

# ─── Visualize ───────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection="3d")

# Ground truth
ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2],
        color="green", linewidth=2, label="Ground Truth")
ax.scatter(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2],
           color="green", s=30, zorder=5)

# Predicted
ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2],
        color="blue", linewidth=2, label="Our approach (FCN)")
ax.scatter(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2],
           color="blue", s=30, zorder=5)

# # Đường nối error từng cặp điểm
# for i in range(len(gt_xyz)):
#     ax.plot([gt_xyz[i, 0], pred_xyz[i, 0]],
#             [gt_xyz[i, 1], pred_xyz[i, 1]],
#             [gt_xyz[i, 2], pred_xyz[i, 2]],
#             color="gray", linewidth=0.7, alpha=0.5)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title(
    f"Helical Trajectory — FCN Inference\n"
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