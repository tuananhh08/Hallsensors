import numpy as np
import pandas as pd

#load input files
# helical_df = pd.read_csv("Helical_points_coordinates.csv") #vi tri cac diem he Helical
sensor_df = pd.read_csv("Sensors_pos.csv") #vi tri 8x8 sensors

# helical_pts = helical_df.values
sensor_pos = sensor_df.values
# helical_xyz = helical_pts[:,:3]

sensor_center = sensor_pos.mean(axis=0)

#define ROI 
roi_width = 0.15     
roi_depth = 0.15     
roi_height = 0.07    

x_min = sensor_center[0] - roi_width / 2
x_max = sensor_center[0] + roi_width / 2

y_min = sensor_center[1] - roi_depth / 2
y_max = sensor_center[1] + roi_depth / 2

z_min = sensor_center[2]
z_max = sensor_center[2] + roi_height

#ROI
num_xy = 30
num_z = 20
num_angle = 18

x_vals = np.linspace(x_min, x_max, num_xy)
y_vals = np.linspace(y_min, y_max, num_xy)
z_vals = np.linspace(z_min, z_max, num_z)
pitch_vals = np.linspace(0,180, num_angle)
yaw_vals = np.linspace(0,180,num_angle)

num_files = 20
columns = ["x", "y", "z", "cos_pitch", "cos_yaw"]

total_samples = (
    len(x_vals) * len(y_vals) * len(z_vals) *
    len(pitch_vals) * len(yaw_vals)
)

rows_per_file = total_samples // num_files

buffer = []
file_idx = 1
counter = 0

for x in x_vals:
    for y in y_vals:
        for z in z_vals:
            for pitch in pitch_vals:
                for yaw in yaw_vals:
                    cos_pitch = np.cos(np.deg2rad(pitch))
                    cos_yaw = np.cos(np.deg2rad(yaw))

                    buffer.append([x, y, z, cos_pitch, cos_yaw])
                    counter += 1

                    if counter % rows_per_file == 0:
                        df = pd.DataFrame(buffer, columns=columns)
                        df.to_csv(f"ROI_data_{file_idx}.csv", index=False)
                        print(f"Saved ROI_data_{file_idx}.csv")

                        buffer = []
                        file_idx += 1

# write csv files
if buffer:
    df = pd.DataFrame(buffer, columns=columns)
    df.to_csv(f"ROI_data_{file_idx}.csv", index=False)
    print(f"Saved ROI_data_{file_idx}.csv")

print("Data generation completed.")


