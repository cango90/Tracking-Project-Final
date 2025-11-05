import cv2
import numpy as np
import math
from typing import Dict, Tuple, List, Any, Optional

# ==============================================================================
# 1. KONSTANTEN & WELT-DEFINITIONEN
# ==============================================================================

VIDEO_FILE = '/Users/berk-cancelebi/Documents/Dolgov-Software_Engineering/ArUco_Projekt/VID_20251008_083345.mp4'
OUTPUT_FILE_COMBI = '/Users/berk-cancelebi/Documents/Dolgov-Software_Engineering/ArUco_Projekt/final_combined_output.mp4'
MARKER_SIDE_LENGTH = 12.0               
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Kamerakalibrierung (Globale Konstanten für die Übergabe)
K_CAMERA = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 480.0], [0.0, 0.0, 1.0]])
DIST_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

ID_A, ID_B, ID_C = 2, 0, 1 

POS_A = np.array([45.0, -30.0, 0.0])    
POS_B = np.array([0.0, 0.0, 0.0])       
POS_C = np.array([90.0, 20.0, 0.0])     

marker_data: Dict[int, np.ndarray] = {
    ID_A: POS_A,
    ID_B: POS_B,
    ID_C: POS_C
}

half_side = MARKER_SIDE_LENGTH / 2.0 

BASE_CORNERS = np.array([
    [-half_side, half_side, 0.0], [ half_side, half_side, 0.0], 
    [ half_side,-half_side, 0.0], [-half_side,-half_side, 0.0] 
], dtype=np.float32)

FRUSTUM_BASE_POINTS = np.float32([
    [0, 0, 0], [15, 5, 0], [-15, 5, 0], [-15, 30, 0], [15, 30, 0]     
])

CAMERA_PATH = []
# Smooth translation only (more stable than smoothing full 4x4 matrix)
SMOOTHING_ALPHA = 0.95
LAST_T_WORLD: Optional[np.ndarray] = None
LAST_QUAT: Optional[np.ndarray] = None
H_smooth_pose = None


# ==============================================================================
# 2. HILFSFUNKTIONEN (Signaturen auf globalen Zugriff angepasst)
# ==============================================================================

def get_obj_and_img_points(corners: np.ndarray, ids: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Sammelt alle 3D-Weltpunkte und ihre 2D-Bildpunkte für solvePnP.
       Greift direkt auf globale marker_data und BASE_CORNERS zu.
    """
    obj_points = []
    img_points = []
    
    if ids is None:
        return np.array([]), np.array([])

    for i in range(len(ids)):
        marker_id = ids[i][0] if isinstance(ids[i], np.ndarray) else ids[i] 
        
        # Zugriff auf globale Variablen
        if marker_id in marker_data:
            marker_pos = marker_data[marker_id]
            world_corners = BASE_CORNERS + marker_pos
            image_corners = corners[i][0]
            
            obj_points.extend(world_corners)
            img_points.extend(image_corners)

    return np.array(obj_points, dtype=np.float32), np.array(img_points, dtype=np.float32)

def calculate_camera_pose(obj_points: np.ndarray, img_points: np.ndarray, K_mat: np.ndarray, dist_coeffs: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, float]:
    """Berechnet die Posen (rvec, tvec) und den Reprojektionsfehler."""
    
    if len(obj_points) < 4:
        return False, np.array([]), np.array([]), 0.0

    retval, rvec, tvec = cv2.solvePnP(obj_points, img_points, K_mat, dist_coeffs) 
    
    if retval:
        img_points_reprojected, _ = cv2.projectPoints(obj_points, rvec, tvec, K_mat, dist_coeffs)
        error_pixels = np.sqrt(np.sum((img_points - img_points_reprojected.reshape(-1, 2))**2, axis=1))
        rms_error = np.mean(error_pixels)
        return True, rvec, tvec, rms_error
    else:
        return False, np.array([]), np.array([]), 0.0

def pose_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Konvertiert rvec und tvec in eine 4x4 homogene Transformationsmatrix.

    IMPORTANT: OpenCV's solvePnP returns rvec,tvec such that
        X_cam = R * X_world + tvec
    therefore the matrix built here is the transform from WORLD -> CAMERA
    (i.e. T_cam_world = [R t; 0 1]). Callers that need CAMERA in WORLD
    coordinates should invert this matrix (see `invert_pose_matrix`).
    """
    R, _ = cv2.Rodrigues(rvec)
    H = np.eye(4, dtype=np.float64)
    H[:3, :3] = R
    H[:3, 3] = tvec.flatten()
    return H

def invert_pose_matrix(H: np.ndarray) -> np.ndarray:
    """Invertiert eine 4x4 homogene Pose effizient.

    Wenn H = [R t; 0 1] (T_cam_world), dann die Inverse ist
        H_inv = [R.T  -R.T @ t; 0 1]
    welche Transformiert Punkte von CAMERA -> WORLD.
    """
    R = H[:3, :3]
    t = H[:3, 3].reshape(3, 1)
    H_inv = np.eye(4, dtype=np.float64)
    H_inv[:3, :3] = R.T
    H_inv[:3, 3] = (-R.T @ t).flatten()
    return H_inv

def get_axis_points(marker_side_length: float) -> np.ndarray:
    """Definiert die 3D-Punkte für die Achsen, verschoben an die untere linke Ecke."""
    axis_length = marker_side_length / 2.0
    offset = marker_side_length / 2.0 
    
    axis_points_3d = np.float32([
        [-offset, -offset, 0],          
        [-offset + axis_length, -offset, 0],  
        [-offset, -offset + axis_length, 0],  
        [-offset, -offset, -axis_length]      
    ])
    return axis_points_3d


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion (w, x, y, z)."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    else:
        if m00 > m11 and m00 > m22:
            s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = q
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = np.array([
        [ww + xx - yy - zz, 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     ww - xx + yy - zz, 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     ww - xx - yy + zz]
    ], dtype=np.float64)
    return R


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between quaternions q0->q1 by t in [0,1]."""
    # Normalize
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # If the quaternions are nearly identical, use linear interpolation
        result = q0 + t*(q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * q0) + (s1 * q1)

# ==============================================================================
# 3. HAUPTVISUALISIERUNGS-SCHLEIFE
# ==============================================================================

def process_video():
    global H_smooth_pose
    import argparse
    parser = argparse.ArgumentParser(description='Process video and track ArUco markers')
    parser.add_argument('--headless', action='store_true', help='Run without displaying windows and save outputs only')
    args, _ = parser.parse_known_args()
    headless = args.headless
    
    VIDEO_FILE_LOCAL = '/Users/berk-cancelebi/Documents/Dolgov-Software_Engineering/ArUco_Projekt/VID_20251008_083345.mp4'
    OUTPUT_FILE_COMBI = '/Users/berk-cancelebi/Documents/Dolgov-Software_Engineering/ArUco_Projekt/final_combined_output.mp4'

    cap = cv2.VideoCapture(VIDEO_FILE_LOCAL) 
    parameters = cv2.aruco.DetectorParameters() 

    if not cap.isOpened():
        print(f"FEHLER: Konnte Videodatei '{VIDEO_FILE_LOCAL}' nicht öffnen.")
        return
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = 30.0 

    MAP_HEIGHT = frame_height 
    output_width = frame_width * 2
    output_height = frame_height 
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_combi = cv2.VideoWriter(OUTPUT_FILE_COMBI, fourcc, output_fps, (output_width, output_height))
    
    print(f"Starte Tracking und Fusion. Ergebnis gespeichert in {OUTPUT_FILE_COMBI}")
    
    MAP_SIZE = frame_width
    MAP_SCALE = 5                  
    MAP_CENTER_X = frame_width // 2
    MAP_CENTER_Y = frame_height // 2
    
    axis_points_3d = get_axis_points(MARKER_SIDE_LENGTH)
    
    is_map_initialized = False
    TVEC_MAP_FINAL = np.array([[-45.0], [10.0], [90.0]], dtype=np.float32) 
    RVEC_MAP = np.array([[-0.5], [0.0], [0.0]], dtype=np.float32) 
    K_map_perspective = np.array([[700, 0, MAP_CENTER_X], [0, 700, MAP_CENTER_Y], [0, 0, 1]], dtype=np.float32)


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_visual = frame.copy()
        map_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8) 
        map_image.fill(30) 
        
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, DICTIONARY, parameters=parameters)

        if ids is not None and len(ids) >= 2: 
            
            # KORRIGIERT: Nur corners und ids übergeben
            obj_points, img_points = get_obj_and_img_points(corners, ids)
            
            if len(obj_points) >= 8:
                
                # KORRIGIERT: Kamerakonstanten übergeben
                retval, rvec_raw, tvec_raw, rms_error = calculate_camera_pose(obj_points, img_points, K_CAMERA, DIST_COEFFS)
                
                if retval: 
                    
                    # --- 1. DYNAMISCHE KARTENZENTRIERUNG ---
                    if not is_map_initialized:
                        camera_pos_x = tvec_raw[0][0]
                        camera_pos_y = tvec_raw[1][0]
                        TVEC_MAP_FINAL = np.array([[-camera_pos_x + 10.0], [-camera_pos_y - 10.0], [150.0]], dtype=np.float32)
                        is_map_initialized = True
                    
                    # 2. Compute camera pose in WORLD coordinates from rvec/tvec
                    # OpenCV: X_cam = R_cam * X_world + t_cam
                    R_cam, _ = cv2.Rodrigues(rvec_raw)
                    # Convert to world frame: R_world = R_cam.T ; t_world = -R_cam.T @ t_cam
                    R_world = R_cam.T
                    t_world = (-R_world @ tvec_raw).reshape(3)

                    # Smooth translation only for a stable path
                    global LAST_T_WORLD
                    if LAST_T_WORLD is None:
                        LAST_T_WORLD = t_world.copy()
                    else:
                        LAST_T_WORLD = (SMOOTHING_ALPHA * LAST_T_WORLD) + ((1.0 - SMOOTHING_ALPHA) * t_world)

                    # Use smoothed translation and SLERP-smoothed rotation
                    tvec_smooth_world = LAST_T_WORLD

                    # Rotation smoothing via SLERP
                    global LAST_QUAT
                    q_curr = rotmat_to_quat(R_world)
                    if LAST_QUAT is None:
                        LAST_QUAT = q_curr
                    else:
                        # Interpolate towards curr: use small step (1 - alpha)
                        t = (1.0 - SMOOTHING_ALPHA)
                        LAST_QUAT = slerp(LAST_QUAT, q_curr, t)
                        LAST_QUAT = LAST_QUAT / np.linalg.norm(LAST_QUAT)

                    R_smooth_world = quat_to_rotmat(LAST_QUAT)

                    x, y, z = float(tvec_smooth_world[0]), float(tvec_smooth_world[1]), float(tvec_smooth_world[2])
                    
                    # 3. Visualisierung im ORIGINAL-Frame
                    
                    rvecs_single, tvecs_single, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIDE_LENGTH, K_CAMERA, DIST_COEFFS)

                    for i in range(len(ids)):
                        marker_id = ids[i][0] 
                        rvec_i, tvec_i = rvecs_single[i], tvecs_single[i]
                        
                        img_points_axis, _ = cv2.projectPoints(get_axis_points(MARKER_SIDE_LENGTH), rvec_i, tvec_i, K_CAMERA, DIST_COEFFS)
                        img_points_axis = np.int32(img_points_axis).reshape(-1, 2)
                        
                        p_origin, p_x, p_y, p_z = tuple(img_points_axis[0]), tuple(img_points_axis[1]), tuple(img_points_axis[2]), tuple(img_points_axis[3])
                        
                        cv2.line(frame_visual, p_origin, p_x, (0, 0, 255), 3)  
                        cv2.line(frame_visual, p_origin, p_y, (0, 255, 0), 3)  
                        cv2.line(frame_visual, p_origin, p_z, (255, 0, 0), 3)  
                    
                        corner_point = corners[i][0][3].astype(np.int32) 
                        cv2.circle(frame_visual, tuple(corner_point), 5, (255, 0, 0), -1) 
                        
                        text_id_vid = f"ID {marker_id}"
                        text_pos_vid = (corners[i][0][0].astype(np.int32) + np.array([-10, -10])).flatten()
                        cv2.putText(frame_visual, text_id_vid, tuple(text_pos_vid), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) 
                    
                    
                    # Textanzeige mit Genauigkeitswert
                    text_pos = f"Pos: X={x:.2f} Y={y:.2f} Z={z:.2f} cm (Ref ID 0)"
                    text_error = f"Fehler: {rms_error:.3f} px" 
                    
                    cv2.putText(frame_visual, text_pos, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame_visual, text_error, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) 
                    
                    # Pfadverfolgung (Der glatte Graph) -- use camera origin in WORLD coordinates
                    # Store camera pose + yaw (compute yaw from smoothed rotation)
                    # Yaw extraction (around Z) from rotation matrix
                    yaw = math.atan2(R_smooth_world[1,0], R_smooth_world[0,0])
                    CAMERA_PATH.append(np.array([x, y, z, yaw], dtype=np.float32))
                    path_points_3d = np.array([p[:3] for p in CAMERA_PATH]).reshape(-1, 3)
                            
                    # 4. Visualisierung in der 3D-Kartenansicht (links)
                    
                    for marker_id, pos_center in marker_data.items():
                        
                        marker_corners_world = BASE_CORNERS + pos_center
                        
                        marker_points_map, _ = cv2.projectPoints(marker_corners_world, RVEC_MAP, TVEC_MAP_FINAL, K_map_perspective, None)
                        marker_points_map = np.int32(marker_points_map).reshape(-1, 2)
                        
                        cv2.polylines(map_image, [marker_points_map], isClosed=True, color=(255, 0, 0), thickness=2) 
                        center_map = np.mean(marker_points_map, axis=0).astype(np.int32)
                        cv2.circle(map_image, tuple(center_map), 4, (0, 0, 255), -1) 
                        
                        text_id = f"ID {marker_id}"
                        cv2.putText(map_image, text_id, (center_map[0] + 8, center_map[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        
                    # b) Kamera-Frustum projizieren
                    # Compute frustum points in WORLD coordinates by transforming from CAMERA frame
                    frustum_world = []
                    for p in FRUSTUM_BASE_POINTS:
                        # p is defined in camera coordinates (frustum). Transform to world:
                        p_cam = p.reshape(3,)
                        p_world = (R_smooth_world @ p_cam) + tvec_smooth_world
                        frustum_world.append(p_world)

                    frustum_world = np.array(frustum_world)
                    
                    frustum_points_map, _ = cv2.projectPoints(frustum_world, RVEC_MAP, TVEC_MAP_FINAL, K_map_perspective, None)
                    frustum_2d_map = np.int32(frustum_points_map).reshape(-1, 2)
                    
                    if len(frustum_2d_map) == 5:
                        p_orig = frustum_2d_map[0]
                        
                        polygon_points = np.array([frustum_2d_map[1:]], np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(map_image, [polygon_points], (150, 150, 150)) 
                        cv2.polylines(map_image, polygon_points, isClosed=True, color=(255, 255, 255), thickness=1)
                        cv2.line(map_image, p_orig, frustum_2d_map[3], (150, 150, 150), 1)
                        cv2.line(map_image, p_orig, frustum_2d_map[4], (150, 150, 150), 1)
                        cv2.circle(map_image, p_orig, 4, (0, 255, 255), -1) 

                    # c) Pfad zeichnen (Projektion des geglätteten Pfades)
                    if len(CAMERA_PATH) > 1:
                        path_points_map, _ = cv2.projectPoints(path_points_3d, RVEC_MAP, TVEC_MAP_FINAL, K_map_perspective, None)
                        path_points_map = np.int32(path_points_map).reshape(-1, 2)
                        
                        for i in range(1, len(path_points_map)):
                            cv2.line(map_image, tuple(path_points_map[i-1]), tuple(path_points_map[i]), (0, 255, 0), 2) 

                    # d) Textanzeige
                    cv2.putText(map_image, f"Fehler: {rms_error:.3f} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        # 5. Endgültige Fusion und Speicherung
        
        map_image_resized = cv2.resize(map_image, (frame_width, frame_height))
            
        combined_frame = np.hstack((map_image_resized, frame_visual))
        
        output_total_width = frame_width * 2
        output_total_height = frame_height 
        
        out_combi.write(combined_frame)

        if not headless:
            cv2.imshow('Kombinierte Ansicht (Map + Video)', combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out_combi.release() 
    cv2.destroyAllWindows()
    print(f"Video-Verarbeitung beendet. Ergebnis gespeichert in {OUTPUT_FILE_COMBI}")

    # Save camera path to CSV for offline inspection
    try:
        import csv
        csv_path = OUTPUT_FILE_COMBI.replace('.mp4', '_camera_path.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Check if yaw present
            if CAMERA_PATH and len(CAMERA_PATH[0]) == 4:
                writer.writerow(['x', 'y', 'z', 'yaw'])
                for p in CAMERA_PATH:
                    writer.writerow([float(p[0]), float(p[1]), float(p[2]), float(p[3])])
            else:
                writer.writerow(['x', 'y', 'z'])
                for p in CAMERA_PATH:
                    writer.writerow([float(p[0]), float(p[1]), float(p[2])])
        print(f"Camera path saved to {csv_path} ({len(CAMERA_PATH)} points)")
    except Exception as e:
        print(f"Failed to save camera path CSV: {e}")

if __name__ == "__main__":
    process_video()