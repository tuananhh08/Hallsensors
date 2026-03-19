import numpy as np
import pandas as pd
import os

#load input files
sensor_df = pd.read_csv("Hall_sensor_positions.csv") #vi tri 8x8 sensors

# helical_pts = helical_df.values
sensor_pos = sensor_df.values
# helical_xyz = helical_pts[:,:3]

sensor_center = sensor_pos.mean(axis=0)

#define ROI 
roi_width = 0.14     #14cm
roi_depth = 0.14     #14cm
roi_height = 0.07    #7cm

x_min = sensor_center[0] - roi_width / 2
x_max = sensor_center[0] + roi_width / 2

y_min = sensor_center[1] - roi_depth / 2
y_max = sensor_center[1] + roi_depth / 2

z_min = sensor_center[2] + 0.01
z_max = sensor_center[2] + roi_height

#ROI
num_xy = 28     #26
num_z = 12      
#num_angle = 26 

x_vals = np.linspace(x_min, x_max, num_xy)
y_vals = np.linspace(y_min, y_max, num_xy)
z_vals = np.linspace(z_min, z_max, num_z)
# pitch_vals = np.linspace(0,180, num_angle)
# yaw_vals = np.linspace(0,180,num_angle)

num_files = 32
columns = ["x", "y", "z", "cos_alpha", "cos_beta"]

total_samples = (
    len(x_vals) * len(y_vals) * len(z_vals) 
    # len(pitch_vals) * len(yaw_vals)
)

rows_per_file = total_samples // num_files

# -------- Folder path --------
output_folder = r"D:\Downloads\Hallsensors\InvProblem\Code\ROI_data"
os.makedirs(output_folder, exist_ok=True)

buffer = []
file_idx = 1
counter = 0

for x in x_vals:
    for y in y_vals:
        for z in z_vals:

            cos_alpha = 1
            cos_beta = 1

            buffer.append([x, y, z, cos_alpha, cos_beta])
            counter += 1

            if counter % rows_per_file == 0:
                df = pd.DataFrame(buffer, columns=columns)

                file_path = os.path.join(output_folder, f"ROI_data_{file_idx}.csv")
                df.to_csv(file_path, index=False)

                print(f"Saved {file_path}")

                buffer = []
                file_idx += 1

# write csv files
if buffer:
    df = pd.DataFrame(buffer, columns=columns)
    file_path = os.path.join(output_folder, f"ROI_data_{file_idx}.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {file_path}")

print("Data generation completed.")


