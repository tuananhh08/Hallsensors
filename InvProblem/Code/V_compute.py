import numpy as np
import pandas as pd
from pathlib import Path

# CONSTANTS
MU_0_4PI = 1e-7   # mu0 / (4pi)
VCC      = 3.3    # V
m0       = 1.0

# PATHS
BASE_DIR   = Path(__file__).parent
sensor_pos = pd.read_csv(BASE_DIR / "Hall_sensor_positions.csv").values
Ns         = sensor_pos.shape[0]
print(f"Loaded {Ns} sensors from {BASE_DIR / 'Hall_sensor_positions.csv'}")

# Load calibration 
calib_df = pd.read_csv(BASE_DIR / "Calibration_grid_result.csv")
calib_df = calib_df.sort_values("sensor_index").reset_index(drop=True)

assert len(calib_df) == Ns, \
    f"Số sensor trong calibration ({len(calib_df)}) != sensor_pos ({Ns})"

# Shape (Ns,) 
V_Q_arr  = calib_df["offset_a_V"].to_numpy()       # V_Q riêng từng sensor
GAIN_arr = calib_df["gain_g_V_per_T"].to_numpy()   # sensitivity riêng từng sensor

print(f"Calibration loaded: {len(calib_df)} sensors")
print(f"  V_Q   range: [{V_Q_arr.min():.4f}, {V_Q_arr.max():.4f}] V")
print(f"  Gain  range: [{GAIN_arr.min():.4f}, {GAIN_arr.max():.4f}] V/T")

roi_folder = BASE_DIR / "ROI_data"
out_folder = BASE_DIR / "ROI_voltage"
out_folder.mkdir(parents=True, exist_ok=True)


# FUNCTIONS
def compute_m_vectors(cos_alpha, cos_beta):
    """
    Tinh vector moment tu truong tu cos_alpha va cos_beta.
    sin >= 0 vi alpha, beta trong [0, 180 deg].
    """
    sin_alpha = np.sqrt(np.clip(1 - cos_alpha**2, 0, 1))
    sin_beta  = np.sqrt(np.clip(1 - cos_beta**2,  0, 1))

    mx = m0 * cos_alpha * cos_beta
    my = m0 * cos_alpha * sin_beta
    mz = m0 * sin_alpha

    return np.stack([mx, my, mz], axis=1)   # (N, 3)


def compute_B_all(roi_xyz, sensor_pos, m_vecs):
    """
    Tinh do lon cam ung tu B tai tung sensor tu cong thuc dipole.
    Returns: B_mag (N, Ns)
    """
    # r_vec: (N, Ns, 3)
    r_vec  = sensor_pos[None, :, :] - roi_xyz[:, None, :]
    r_norm = np.linalg.norm(r_vec, axis=2, keepdims=True)
    r_norm = np.clip(r_norm, 1e-4, None)   # tranh singularity

    m_dot_r = np.sum(m_vecs[:, None, :] * r_vec, axis=2, keepdims=True)

    term1 = 3 * m_dot_r * r_vec / (r_norm ** 5)
    term2 = m_vecs[:, None, :] / (r_norm ** 3)

    B_vec = MU_0_4PI * (term1 - term2)
    B_mag = np.linalg.norm(B_vec, axis=2)   # (N, Ns)

    return B_mag


def B_to_voltage(B_mag, V_Q_arr, GAIN_arr):
    """
    Chuyen B sang voltage dung tham so calibration rieng tung sensor.
    V = V_Q[s] + GAIN[s] * B[s]

    Args:
        B_mag   : (N, Ns)
        V_Q_arr : (Ns,)  — offset rieng tung sensor
        GAIN_arr: (Ns,)  — gain rieng tung sensor
    Returns:
        V_all   : (N, Ns)
    """
    return V_Q_arr[None, :] + GAIN_arr[None, :] * B_mag


# MAIN
num_files     = 32
total_clipped = 0
total_rows    = 0

for i in range(1, num_files + 1):
    roi_file = roi_folder / f"ROI_data_{i}.csv"
    out_file = out_folder / f"ROI_voltage_{i}.csv"

    if not roi_file.exists():
        print(f"[WARNING] Khong tim thay {roi_file} — bo qua")
        continue

    df = pd.read_csv(roi_file)

    roi_xyz   = df.iloc[:, :3].to_numpy()
    cos_alpha = df["cos_alpha"].to_numpy()
    cos_beta  = df["cos_beta"].to_numpy()

    # Kiem tra khoang cach min
    r_min = np.linalg.norm(
        roi_xyz[:, None, :] - sensor_pos[None, :, :], axis=2).min()
    if r_min < 0.005:
        print(f"  [WARNING] File {i}: khoang cach min = {r_min*100:.2f} cm < 0.5cm")

    # Tinh moment vector
    m_vecs = compute_m_vectors(cos_alpha, cos_beta)

    # Tinh B
    B_all = compute_B_all(roi_xyz, sensor_pos, m_vecs)

    # Chuyen sang Voltage dung calibration rieng tung sensor
    V_all = B_to_voltage(B_all, V_Q_arr, GAIN_arr)

    # Clip ve [0, VCC]
    n_clipped     = np.sum((V_all < 0) | (V_all > VCC))
    V_all         = np.clip(V_all, 0, VCC)
    total_clipped += n_clipped
    total_rows    += V_all.size

    pd.DataFrame(V_all).to_csv(out_file, header=False, index=False)

    print(f"  File {i:>2}: {len(df):>7,} samples  "
          f"V=[{V_all.min():.4f}, {V_all.max():.4f}]  "
          f"clipped={n_clipped:,} ({100*n_clipped/V_all.size:.2f}%)")

print(
    f"\nDONE — Total clipped: {total_clipped:,} / {total_rows:,} "
    f"({100*total_clipped/total_rows:.2f}%)" if total_rows > 0
    else "\nDONE — Khong co file nao duoc xu ly."
)
print(f"Output -> {out_folder}")