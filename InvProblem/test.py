import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fcn import FCNBaseline   

#config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"
dz = 0.025  

#load model
model = FCNBaseline(out_dim=5).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


#load position
hall_df = pd.read_csv(
    "Hall_sensor_positions.csv",
    header=None,
    names=["x", "y", "z"]
)
hall_pts = hall_df.values

hall_pts[:, 2] += dz

#ROI
sensor_center = hall_pts.mean(axis=0)

roi_width = 0.12    # 12 cm
roi_depth = 0.12    # 12 cm
roi_height = 0.07   # 7 cm

x_min = sensor_center[0] - roi_width / 2
x_max = sensor_center[0] + roi_width / 2

y_min = sensor_center[1] - roi_depth / 2
y_max = sensor_center[1] + roi_depth / 2

z_min = sensor_center[2]
z_max = sensor_center[2] + roi_height

#load B
B_df = pd.read_csv("B_output.csv", header=None)
B = B_df.values.reshape(-1, 1, 8, 8)
B_tensor = torch.tensor(B, dtype=torch.float32).to(DEVICE)

#infer
with torch.no_grad():
    pred_pose = model(B_tensor)   # (25, 5)
# data = np.load("B_normalized.npz")
# B = data["B"]   # (N, 64)

# print("B std overall:", B.std())
# print("B std per sample:", np.std(B, axis=1)[:10])
pred_pose = pred_pose.cpu().numpy()
pred_xyz = pred_pose[:, :3]
pred_alpha_beta = pred_pose[:, 3:5]   # giữ để đầy đủ pose 5D

# print("Pred pose shape:", pred_pose.shape)
# print("Pred pose (first 5):\n", pred_pose[:5])
# print("Std of pred xyz:", pred_pose[:, :3].std(axis=0))


#load ground truth
helical_df = pd.read_csv("Helical_points_coordinates.csv")

gt_xyz = helical_df.iloc[:, :3].values
gt_alpha = np.ones((gt_xyz.shape[0], 1))
gt_beta  = np.ones((gt_xyz.shape[0], 1))

gt_pose = np.hstack([gt_xyz, gt_alpha, gt_beta])  # (25, 5)


def draw_roi(ax):
    xs = [x_min, x_max]
    ys = [y_min, y_max]
    zs = [z_min, z_max]

    for x in xs:
        for y in ys:
            ax.plot([x, x], [y, y], zs, color="gray", linewidth=1)
    for x in xs:
        for z in zs:
            ax.plot([x, x], ys, [z, z], color="gray", linewidth=1)
    for y in ys:
        for z in zs:
            ax.plot(xs, [y, y], [z, z], color="gray", linewidth=1)


# VISUALIZATION
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# ROI
draw_roi(ax)

# Sensors
ax.scatter(
    hall_pts[:, 0],
    hall_pts[:, 1],
    hall_pts[:, 2],
    c="black",
    s=15,
    label="Hall Sensors"
)

# Ground truth helix
ax.plot(
    gt_pose[:, 0],
    gt_pose[:, 1],
    gt_pose[:, 2],
    "o-",
    linewidth=2,
    label="Ground Truth"
)

# Predicted helix
ax.plot(
    pred_pose[:, 0],
    pred_pose[:, 1],
    pred_pose[:, 2],
    "x--",
    linewidth=2,
    label="Predicted"
)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.legend()
ax.grid(True)

plt.show()
