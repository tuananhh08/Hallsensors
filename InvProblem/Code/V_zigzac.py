import numpy as np
import pandas as pd
from pathlib import Path

# CONSTANTS
MU_0_4PI    = 1e-7              # mu0 / (4pi)
VCC         = 3.3               # V
V_Q         = VCC / 2           # 1.65 V (offset)
SENSITIVITY = 15e-3 / 1e-3      # 15 mV/mT -> V/T
m0          = 1.0

# PATHS 
BASE_DIR = Path(__file__).parent
sensor_pos = pd.read_csv(BASE_DIR / "Sensors_pos.csv").values
Ns = sensor_pos.shape[0]
zigzac_file = Path(r"D:\Downloads\Hallsensors\InvProblem\PosData\Zig_zac_points_coordinates.csv")

# output path
out_file = BASE_DIR / "Zigzac_voltage.csv"


# FUNCTIONS
def compute_m_vectors(cos_alpha, cos_beta):

    sin_alpha = np.sqrt(np.clip(1 - cos_alpha**2, 0, 1))
    sin_beta  = np.sqrt(np.clip(1 - cos_beta**2,  0, 1))

    mx = m0 * cos_alpha * cos_beta
    my = m0 * cos_alpha * sin_beta
    mz = m0 * sin_beta

    return np.stack([mx, my, mz], axis=1)  # (N, 3)


def compute_B_all(roi_xyz, sensor_pos, m_vecs):
    # r_vec
    r_vec  = sensor_pos[None, :, :] - roi_xyz[:, None, :]
    r_norm = np.linalg.norm(r_vec, axis=2, keepdims=True)

    r_norm = np.clip(r_norm, 1e-4, None)  # toi thieu 0.1mm

    m_dot_r = np.sum(m_vecs[:, None, :] * r_vec, axis=2, keepdims=True)

    term1 = 3 * m_dot_r * r_vec / (r_norm ** 5)
    term2 = m_vecs[:, None, :] / (r_norm ** 3)

    B_vec = MU_0_4PI * (term1 - term2)
    B_mag = np.linalg.norm(B_vec, axis=2)  # (N, Ns)

    return B_mag


def B_to_voltage(B):
    return V_Q + SENSITIVITY * B


# LOAD ZIGZAC DATA
df = pd.read_csv(zigzac_file)

zigzac_xyz = df.iloc[:, :3].to_numpy()
cos_alpha  = df["cos_alpha"].to_numpy()
cos_beta    = df["cos_beta"].to_numpy()

print(f"Loaded {len(zigzac_xyz)} zigzac points")

# COMPUTE MAGNETIC MOMENT
m_vecs = compute_m_vectors(cos_alpha, cos_beta)

# COMPUTE B
B_all = compute_B_all(zigzac_xyz, sensor_pos, m_vecs)

# COMPUTE VOLTAGE
V_all = B_to_voltage(B_all)

# CLIP SENSOR RANGE
V_all = np.clip(V_all, 0, VCC)

# SAVE
pd.DataFrame(V_all).to_csv(out_file, header=False, index=False)

print(f"Saved voltage data -> {out_file}")
print(f"Voltage range: {V_all.min():.4f} V  to  {V_all.max():.4f} V")