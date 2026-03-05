import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#load input files
helical_df = pd.read_csv("Helical_points_coordinates.csv")
sensor_df = pd.read_csv("Sensors_pos.csv")
ROI_points_df = pd.read_csv("ROI_data.csv")

pts = ROI_points_df.values
roi_pts = pts[:,:3]             #toa do (x,y,z) roi_points
helical_pts = helical_df.values
sensor_pos = sensor_df.values   #toa do sensor
helical_xyz = helical_pts[:,:3] #toa do (x,y,z) helical_points

sensor_center = sensor_pos.mean(axis=0)

#define ROI 
roi_width = 0.12     
roi_depth = 0.12     
roi_height = 0.07    

x_min = sensor_center[0] - roi_width / 2
x_max = sensor_center[0] + roi_width / 2

y_min = sensor_center[1] - roi_depth / 2
y_max = sensor_center[1] + roi_depth / 2

z_min = sensor_center[2]
z_max = sensor_center[2] + roi_height

def inside_roi(p):
    x, y, z = p
    return (
        x_min <= x <= x_max and
        y_min <= y <= y_max and
        z_min <= z <= z_max
    )

# Check points
hall_inside = np.array([inside_roi(p) for p in sensor_pos])
helical_inside = np.array([inside_roi(p) for p in helical_xyz])

print("Hall sensors inside ROI:", hall_inside.sum(), "/", len(hall_inside))
print("Helical points inside ROI:", helical_inside.sum(), "/", len(helical_inside))


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    roi_pts[:, 0], roi_pts[:, 1], roi_pts[:, 2],
    c="gray", s=2, alpha=0.15, label="ROI points"
)

ax.scatter(
    sensor_pos[:, 0], sensor_pos[:, 1], sensor_pos[:, 2],
    c="blue", s=30, label="Hall sensors"
)

ax.scatter(
    helical_xyz[:, 0], helical_xyz[:, 1], helical_xyz[:, 2],
    c="red", s=20, label="Helical points"
)


ax.scatter(
    sensor_center[0], sensor_center[1], sensor_center[2],
    c="black", s=50, marker="x", label="Sensor center"
)

roi_x = [x_min, x_max]
roi_y = [y_min, y_max]
roi_z = [z_min, z_max]

for x in roi_x:
    for y in roi_y:
        ax.plot([x, x], [y, y], roi_z, color="green")

for x in roi_x:
    for z in roi_z:
        ax.plot([x, x], roi_y, [z, z], color="green")

for y in roi_y:
    for z in roi_z:
        ax.plot(roi_x, [y, y], [z, z], color="green")

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("ROI")

ax.legend(loc = "upper right")
ax.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.show()


