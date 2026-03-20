# import numpy as np
# import pandas as pd
# from pathlib import Path

# # CONSTANTS
# MU_0_4PI = 1e-7   # mu0 / (4pi)
# VCC      = 3.3    # V
# m0       = 1.0

# # PATHS
# BASE_DIR   = Path(__file__).parent
# sensor_pos = pd.read_csv(BASE_DIR / "Hall_sensor_positions.csv").values
# Ns         = sensor_pos.shape[0]
# print(f"Loaded {Ns} sensors from {BASE_DIR / 'Hall_sensor_positions.csv'}")

# # Load calibration 
# calib_df = pd.read_csv(BASE_DIR / "Calibration_grid_result.csv")
# calib_df = calib_df.sort_values("sensor_index").reset_index(drop=True)

# assert len(calib_df) == Ns, \
#     f"So sensor trong calibration ({len(calib_df)}) != sensor_pos ({Ns})"

# V_Q_arr  = calib_df["offset_a_V"].to_numpy()       # (Ns,)
# GAIN_arr = calib_df["gain_g_V_per_T"].to_numpy()   # (Ns,)

# print(f"Calibration loaded: {len(calib_df)} sensors")
# print(f"  V_Q   range: [{V_Q_arr.min():.4f}, {V_Q_arr.max():.4f}] V")
# print(f"  Gain  range: [{GAIN_arr.min():.4f}, {GAIN_arr.max():.4f}] V/T")

# input_file = BASE_DIR / "Grid_points_coordinates.csv"
# out_file   = BASE_DIR / "Grid_voltage.csv"


# # FUNCTIONS
# def compute_m_vectors(cos_alpha, cos_beta):
#     """
#     Tinh vector moment tu truong tu cos_alpha va cos_beta.
#     sin >= 0 vi alpha, beta trong [0, 180 deg].
#     m = [1, 0, 0] khi cos_alpha=1, cos_beta=1.
#     """
#     sin_alpha = np.sqrt(np.clip(1 - cos_alpha**2, 0, 1))
#     sin_beta  = np.sqrt(np.clip(1 - cos_beta**2,  0, 1))

#     mx = m0 * cos_alpha * cos_beta
#     my = m0 * cos_alpha * sin_beta
#     mz = m0 * sin_alpha

#     return np.stack([mx, my, mz], axis=1)   # (N, 3)


# def compute_B_all(roi_xyz, sensor_pos, m_vecs):

#     r_vec  = sensor_pos[None, :, :] - roi_xyz[:, None, :]   # (N, Ns, 3)
#     r_norm = np.linalg.norm(r_vec, axis=2, keepdims=True)
#     r_norm = np.clip(r_norm, 1e-4, None)                     # tranh singularity

#     m_dot_r = np.sum(m_vecs[:, None, :] * r_vec, axis=2, keepdims=True)

#     term1 = 3 * m_dot_r * r_vec / (r_norm ** 5)
#     term2 = m_vecs[:, None, :] / (r_norm ** 3)

#     B_vec = MU_0_4PI * (term1 - term2)   # (N, Ns, 3)
#     Bz    = B_vec[:, :, 2]               # (N, Ns) — chi lay thanh phan z

#     return Bz


# def B_to_voltage(Bz, V_Q_arr, GAIN_arr):
#     """
#     V = V_Q[s] + GAIN[s] * Bz[s]
#     Bz va GAIN deu co dau — phan anh dung vat ly cam bien Hall.
#     """
#     return V_Q_arr[None, :] + GAIN_arr[None, :] * Bz


# # MAIN
# print(f"\nProcessing {input_file.name} ...")
# df = pd.read_csv(input_file)

# roi_xyz   = df.iloc[:, :3].to_numpy()
# cos_alpha = df["cos_alpha"].to_numpy()
# cos_beta  = df["cos_beta"].to_numpy()

# # Kiem tra khoang cach min
# r_min = np.linalg.norm(
#     roi_xyz[:, None, :] - sensor_pos[None, :, :], axis=2).min()
# if r_min < 0.005:
#     print(f"[WARNING] Khoang cach min toi sensor = {r_min*100:.2f} cm < 0.5cm")

# # Tinh moment vector
# m_vecs = compute_m_vectors(cos_alpha, cos_beta)

# # Tinh Bz
# Bz = compute_B_all(roi_xyz, sensor_pos, m_vecs)

# # Chuyen sang Voltage
# V_all = B_to_voltage(Bz, V_Q_arr, GAIN_arr)

# # Clip ve [0, VCC]
# n_clipped = np.sum((V_all < 0) | (V_all > VCC))
# V_all     = np.clip(V_all, 0, VCC)

# pd.DataFrame(V_all).to_csv(out_file, header=False, index=False)

# print(f"  Samples  : {len(df):,}")
# print(f"  V range  : [{V_all.min():.4f}, {V_all.max():.4f}] V")
# print(f"  Clipped  : {n_clipped:,} / {V_all.size:,} ({100*n_clipped/V_all.size:.2f}%)")
# print(f"\nDONE — Output -> {out_file}")


"""
compute_grid_voltage.py
Tinh voltage tu file Grid_points_coordinates.csv voi:
  - Vcc = 3.3 V
  - V_Q = Vcc / 2 = 1.65 V (offset chung)
  - Sensitivity = 7.5 mV/mT = 7.5e-3 / 1e-3 = 7.5 V/T (chung cho tat ca sensor)
  - B chi lay thanh phan Bz (theo truc z cua sensor)
  - m = [1, 0, 0] khi cos_alpha = cos_beta = 1

Output: Grid_voltage.csv (khong header, moi dong la 64 gia tri voltage)

"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Constants 
MU_0_4PI    = 1e-7              
VCC         = 3.3                
V_Q         = VCC / 2            
SENSITIVITY = 7.5e-3 / 1e-3     
m0          = 1.0

# ─── Args 
parser = argparse.ArgumentParser()
parser.add_argument("--coords",  default="Grid_points_coordinates.csv",
                    help="File toa do robot (x,y,z,cos_alpha,cos_beta)")
parser.add_argument("--sensors", default="Hall_sensor_positions.csv",
                    help="File vi tri 64 cam bien Hall")
parser.add_argument("--out",     default="Grid_voltage_no_calib.csv",
                    help="File output voltage")
args = parser.parse_args()

BASE_DIR = Path(__file__).parent

# ─── Load data 
print(f"Loading sensor positions from {args.sensors} ...")
sensor_pos = pd.read_csv(BASE_DIR / args.sensors).values   # (64, 3)
Ns = sensor_pos.shape[0]
print(f"  Loaded {Ns} sensors")

print(f"Loading coordinates from {args.coords} ...")
coord_df  = pd.read_csv(BASE_DIR / args.coords)
roi_xyz   = coord_df[["x", "y", "z"]].values              # (N, 3)
cos_alpha = coord_df["cos_alpha"].values                   # (N,)
cos_beta  = coord_df["cos_beta"].values                    # (N,)
N = len(roi_xyz)
print(f"  Loaded {N} poses")


# Functions 
def compute_m_vectors(cos_alpha, cos_beta):
    """
    Tinh vector moment tu truong m tu cos_alpha va cos_beta.
    sin >= 0 vi alpha, beta thuoc [0, 180 deg].
    Khi cos_alpha=1, cos_beta=1 -> m = [1, 0, 0].
    """
    sin_alpha = np.sqrt(np.clip(1 - cos_alpha**2, 0, 1))
    sin_beta  = np.sqrt(np.clip(1 - cos_beta**2,  0, 1))

    mx = m0 * cos_alpha * cos_beta
    my = m0 * cos_alpha * sin_beta
    mz = m0 * sin_alpha

    return np.stack([mx, my, mz], axis=1)   # (N, 3)


def compute_Bz(roi_xyz, sensor_pos, m_vecs):
    """
    Tinh thanh phan Bz (theo truc z) tai tung sensor tu cong thuc dipole.
    Chi lay Bz vi cam bien Hall chi nhaycam voi thanh phan B doc truc cam bien.

    """
    r_vec  = sensor_pos[None, :, :] - roi_xyz[:, None, :]  # (N, Ns, 3)
    r_norm = np.linalg.norm(r_vec, axis=2, keepdims=True)
    r_norm = np.clip(r_norm, 1e-4, None)                    # tranh singularity < 0.1mm

    m_dot_r = np.sum(m_vecs[:, None, :] * r_vec, axis=2, keepdims=True)

    term1 = 3 * m_dot_r * r_vec / (r_norm ** 5)
    term2 = m_vecs[:, None, :] / (r_norm ** 3)

    B_vec = MU_0_4PI * (term1 - term2)   
    Bz    = B_vec[:, :, 2]               

    return Bz


def Bz_to_voltage(Bz):
    """
    V = V_Q + SENSITIVITY * Bz
    Sensitivity chung cho tat ca sensor: 7.5 V/T
    """
    return V_Q + SENSITIVITY * Bz


# ─── Main 
print("\nComputing magnetic moment vectors ...")
m_vecs = compute_m_vectors(cos_alpha, cos_beta)

# Kiem tra m vector voi mau dau tien
print(f"  m[0] = [{m_vecs[0,0]:.4f}, {m_vecs[0,1]:.4f}, {m_vecs[0,2]:.4f}]"
      f"  (ky vong [1,0,0] khi cos_alpha=cos_beta=1)")

print("Computing Bz at all sensors ...")
Bz = compute_Bz(roi_xyz, sensor_pos, m_vecs)   # (N, 64)
print(f"  Bz range: [{Bz.min():.6f}, {Bz.max():.6f}] T")

print("Converting Bz to voltage ...")
V_all = Bz_to_voltage(Bz)                       # (N, 64)

# Kiem tra khoang cach min
r_min = np.linalg.norm(
    roi_xyz[:, None, :] - sensor_pos[None, :, :], axis=2).min()
if r_min < 0.005:
    print(f"  [WARNING] Khoang cach min toi sensor = {r_min*100:.2f} cm < 0.5cm")

# Clip ve [0, VCC] — sensor vat ly bi bao hoa
n_clipped = np.sum((V_all < 0) | (V_all > VCC))
V_all     = np.clip(V_all, 0, VCC)

print(f"\n─── Ket qua ─────────────────────────────────")
print(f"  Samples  : {N:,}")
print(f"  V range  : [{V_all.min():.4f}, {V_all.max():.4f}] V")
print(f"  V_Q      : {V_Q} V  |  Sensitivity: {SENSITIVITY} V/T")
print(f"  Clipped  : {n_clipped:,} / {V_all.size:,} "
      f"({100*n_clipped/V_all.size:.2f}%)")
print(f"─────────────────────────────────────────────")

# Save — khong header, khong index
out_path = BASE_DIR / args.out
pd.DataFrame(V_all).to_csv(out_path, header=False, index=False)
print(f"\nSaved -> {out_path}")