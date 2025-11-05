ArUco Video Processing

Quick start

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run processing (GUI):

```bash
python process_video.py
```

3. Run headless (write outputs only):

```bash
python process_video.py --headless
```

Outputs
- `final_combined_output.mp4` - combined map + frame video
- `final_combined_output_camera_path.csv` - CSV with columns x,y,z,yaw
- `camera_path_topdown.png` - PNG top-down visualization with yaw arrows

Developer notes
- Camera pose math: OpenCV's solvePnP returns rvec,tvec such that X_cam = R * X_world + t.
  The code converts this to camera-in-world by R_world = R_cam.T and t_world = -R_cam.T @ t_cam.
- Rotation smoothing uses quaternion SLERP. Translation smoothing is exponential (alpha=0.95).

Testing
- Run small helper tests with pytest:

```bash
pytest -q
```
