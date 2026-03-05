import numpy as np
import pandas as pd
from pathlib import Path

# CONSTANTS
MU_0_4PI = 1e-7               # μ0 / (4π)
VCC = 3.3
V_Q = VCC / 2
SENSITIVITY = 15e-3 / 1e-3   # 15 mV/mT → V/T
m0 = 1.0

# LOAD SENSOR POSITIONS
sensor_pos = pd.read_csv("Sensors_pos.csv").values
Ns = sensor_pos.shape[0]

# COMPUTE
def compute_m_vectors(cos_pitch, cos_yaw):

    sin_pitch = np.sqrt(1 - cos_pitch**2)
    sin_yaw = np.sqrt(1 - cos_yaw**2)

    mx = m0 * cos_pitch * cos_yaw
    my = m0 * cos_pitch * sin_yaw
    mz = m0 * sin_pitch

    return np.stack([mx, my, mz], axis=1)


def compute_B_all(roi_xyz, sensor_pos, m_vecs):

    r_vec = sensor_pos[None, :, :] - roi_xyz[:, None, :]
    r_norm = np.linalg.norm(r_vec, axis=2, keepdims=True) + 1e-12

    m_dot_r = np.sum(m_vecs[:, None, :] * r_vec, axis=2, keepdims=True)

    term1 = 3 * m_dot_r * r_vec / (r_norm ** 5)
    term2 = m_vecs[:, None, :] / (r_norm ** 3)

    B_vec = MU_0_4PI * (term1 - term2)
    B_mag = np.linalg.norm(B_vec, axis=2)

    return B_mag


def B_to_voltage(B):
    return V_Q + SENSITIVITY * B

# PROCESS ALL ROI FILES
roi_folder = Path("ROI_data")
out_folder = Path("/content/drive/MyDrive/Hallsensors_data/ROI_voltage")
out_folder.mkdir(parents=True, exist_ok=True)

for i in range(1, 21):
    roi_file = roi_folder / f"ROI_data_{i}.csv"
    out_file = out_folder / f"ROI_voltage_{i}.csv"

    df = pd.read_csv(roi_file)

    roi_xyz = df.iloc[:, :3].to_numpy()
    cos_pitch = df["cos_pitch"].to_numpy()
    cos_yaw = df["cos_yaw"].to_numpy()

    # compute moment vectors
    m_vecs = compute_m_vectors(cos_pitch, cos_yaw)

    # compute B
    B_all = compute_B_all(roi_xyz, sensor_pos, m_vecs)

    # convert to Volt
    V_all = B_to_voltage(B_all)

    pd.DataFrame(V_all).to_csv(out_file, header=False, index=False)

    print(f"Saved {out_folder / f'ROI_voltage_{i}.csv'}")
print("DONE")