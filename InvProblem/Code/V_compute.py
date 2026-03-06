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
BASE_DIR   = Path(__file__).parent
sensor_pos = pd.read_csv(BASE_DIR / "Sensors_pos.csv").values
Ns         = sensor_pos.shape[0]
print(f"Loaded {Ns} sensors from {BASE_DIR / 'Sensors_pos.csv'}")

roi_folder = BASE_DIR   
out_folder = BASE_DIR / "ROI_voltage"  
out_folder.mkdir(parents=True, exist_ok=True)


# FUNCTIONS
def compute_m_vectors(cos_pitch, cos_yaw):
    """
    Tinh vector moment tu truong tu cos_pitch va cos_yaw.
    sin >= 0 vi pitch, yaw trong [0, 180 deg].
    """
    sin_pitch = np.sqrt(np.clip(1 - cos_pitch**2, 0, 1))
    sin_yaw   = np.sqrt(np.clip(1 - cos_yaw**2,   0, 1))

    mx = m0 * cos_pitch * cos_yaw
    my = m0 * cos_pitch * sin_yaw
    mz = m0 * sin_pitch

    return np.stack([mx, my, mz], axis=1)  # (N, 3)


def compute_B_all(roi_xyz, sensor_pos, m_vecs):
    # r_vec: (N, Ns, 3) — vector tu robot toi sensor
    r_vec  = sensor_pos[None, :, :] - roi_xyz[:, None, :]
    r_norm = np.linalg.norm(r_vec, axis=2, keepdims=True)

    # Clip r_norm de tranh singularity (truong hop robot qua gan sensor)
    r_norm = np.clip(r_norm, 1e-4, None)  # toi thieu 0.1mm

    m_dot_r = np.sum(m_vecs[:, None, :] * r_vec, axis=2, keepdims=True)

    term1 = 3 * m_dot_r * r_vec / (r_norm ** 5)
    term2 = m_vecs[:, None, :] / (r_norm ** 3)

    B_vec = MU_0_4PI * (term1 - term2)
    B_mag = np.linalg.norm(B_vec, axis=2)  # (N, Ns)

    return B_mag


def B_to_voltage(B):
    return V_Q + SENSITIVITY * B


# MAIN
num_files    = 20
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
    cos_pitch = df["cos_pitch"].to_numpy()
    cos_yaw   = df["cos_yaw"].to_numpy()

    # Kiem tra khoang cach min de canh bao
    r_min = np.linalg.norm(
        roi_xyz[:, None, :] - sensor_pos[None, :, :], axis=2).min()
    if r_min < 0.005:
        print(f"  [WARNING] File {i}: khoang cach min toi sensor = {r_min*100:.2f} cm < 0.5cm")

    # Tinh moment vector
    m_vecs = compute_m_vectors(cos_pitch, cos_yaw)

    # Tinh B
    B_all = compute_B_all(roi_xyz, sensor_pos, m_vecs)

    # Chuyen sang Voltage
    V_all = B_to_voltage(B_all)

    # Clip ve [0, VCC] — phan anh sensor vat ly bi bao hoa
    n_clipped     = np.sum((V_all < 0) | (V_all > VCC))
    V_all         = np.clip(V_all, 0, VCC)
    total_clipped += n_clipped
    total_rows    += V_all.size

    pd.DataFrame(V_all).to_csv(out_file, header=False, index=False)

    print(f"  File {i:>2}: {len(df):>7,} samples  "
          f"V=[{V_all.min():.4f}, {V_all.max():.4f}]  "
          f"clipped={n_clipped:,} ({100*n_clipped/V_all.size:.2f}%)")

print(f"\nDONE — Total clipped: {total_clipped:,} / {total_rows:,} "
      f"({100*total_clipped/total_rows:.2f}%)" if total_rows > 0 else
      "\nDONE — Khong co file nao duoc xu ly. Kiem tra lai roi_folder.")
print(f"Output -> {out_folder}")