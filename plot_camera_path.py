import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path('/Users/berk-cancelebi/Documents/Dolgov-Software_Engineering/ArUco_Projekt/final_combined_output_camera_path.csv')
if not csv_path.exists():
    print('CSV not found:', csv_path)
    raise SystemExit(1)

xs, ys, zs, yaws = [], [], [], []
with open(csv_path, newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    has_yaw = 'yaw' in [h.lower() for h in header]
    for row in reader:
        if has_yaw:
            x,y,z,yaw = row
            xs.append(float(x)); ys.append(float(y)); zs.append(float(z)); yaws.append(float(yaw))
        else:
            x,y,z = row
            xs.append(float(x)); ys.append(float(y)); zs.append(float(z)); yaws.append(0.0)

xs = np.array(xs); ys = np.array(ys); yaws = np.array(yaws)

plt.figure(figsize=(10,8))
plt.scatter(xs, ys, s=6, c=np.linspace(0,1,len(xs)), cmap='viridis')
plt.plot(xs, ys, alpha=0.5)
# draw arrows every N points
N = max(1, len(xs)//40)
for i in range(0, len(xs), N):
    dx = np.cos(yaws[i]) * 5
    dy = np.sin(yaws[i]) * 5
    plt.arrow(xs[i], ys[i], dx, dy, head_width=2, head_length=3, color='red')

plt.gca().set_aspect('equal', 'box')
plt.title('Camera path (top-down)')
plt.xlabel('X')
plt.ylabel('Y')
out_png = csv_path.parent / 'camera_path_topdown.png'
plt.savefig(out_png, dpi=200)
print('Saved plot to', out_png)

# Try to overlay known marker positions from process_video if available
try:
    from process_video import marker_data
    # marker_data: dict[id] -> np.array([x,y,z])
    for mid, pos in marker_data.items():
        mx, my = pos[0], pos[1]
        plt.plot(mx, my, marker='s', color='blue', markersize=8)
        plt.text(mx + 2, my + 2, f'ID {mid}', color='white', fontsize=9, bbox=dict(facecolor='blue', alpha=0.6))
    # overwrite with markers plotted and save another file
    out_png2 = csv_path.parent / 'camera_path_topdown_with_markers.png'
    plt.savefig(out_png2, dpi=200)
    print('Saved plot with markers to', out_png2)
except Exception as e:
    # not critical
    print('Could not overlay marker_data:', e)
