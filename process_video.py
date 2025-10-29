import cv2
import numpy as np
import math

# ==============================================================================
# 1. KONSTANTEN & WELT-DEFINITIONEN
# ==============================================================================

# A. Marker-Spezifikationen
VIDEO_FILE = '/Users/berk-cancelebi/Documents/Dolgov-Software_Engineering/ArUco_Projekt/VID_20251008_083345.mp4'
OUTPUT_FILE_COMBI = '/Users/berk-cancelebi/Documents/Dolgov-Software_Engineering/ArUco_Projekt/final_combined_output.mp4'
MARKER_SIDE_LENGTH = 12.0               
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# B. Kamerakalibrierungsparameter (PLATZHALTER!)
cameraMatrix = np.array([
    [1000.0, 0.0, 640.0],
    [0.0, 1000.0, 480.0],
    [0.0, 0.0, 1.0]
])
distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])


# C. Die Feste Welt-Definition (ID 0 ist (0, 0, 0))
ID_A = 2  # Marker 2
ID_B = 0  # NEUER URSPRUNG
ID_C = 1  # Marker 1

POS_A = np.array([45.0, -30.0, 0.0])    
POS_B = np.array([0.0, 0.0, 0.0])       
POS_C = np.array([90.0, 20.0, 0.0])     

marker_data = {
    ID_A: POS_A,
    ID_B: POS_B,
    ID_C: POS_C
}

half_side = MARKER_SIDE_LENGTH / 2.0
BASE_CORNERS = np.array([
    [-half_side, half_side, 0.0], 
    [ half_side, half_side, 0.0], 
    [ half_side,-half_side, 0.0], 
    [-half_side,-half_side, 0.0] 
], dtype=np.float32)

# GLOBALE VARIABLEN
CAMERA_PATH = []
SMOOTHING_ALPHA = 0.99  
H_smooth_pose = None  

FRUSTUM_BASE_POINTS = np.float32([
    [0, 0, 0],      
    [15, 5, 0],     
    [-15, 5, 0],    
    [-15, 30, 0],   
    [15, 30, 0]     
])


# ==============================================================================
# 2. HILFSFUNKTIONEN
# ==============================================================================

def get_obj_and_img_points(corners, ids, marker_data, base_corners):
    """Sammelt alle 3D-Weltpunkte und ihre 2D-Bildpunkte für solvePnP."""
    obj_points = []
    img_points = []
    
    if ids is None:
        return np.array([]), np.array([])

    for i in range(len(ids)):
        marker_id = ids[i][0] if isinstance(ids[i], np.ndarray) else ids[i] 
        
        if marker_id in marker_data:
            marker_pos = marker_data[marker_id]
            world_corners = base_corners + marker_pos
            image_corners = corners[i][0]
            
            obj_points.extend(world_corners)
            img_points.extend(image_corners)

    return np.array(obj_points, dtype=np.float32), np.array(img_points, dtype=np.float32)

def pose_to_matrix(rvec, tvec):
    """Konvertiert rvec und tvec in eine 4x4 Homogene Transformationsmatrix (World_T_Camera)."""
    R, _ = cv2.Rodrigues(rvec)
    H = np.eye(4, dtype=np.float64)
    H[:3, :3] = R
    H[:3, 3] = tvec.flatten()
    return H

def get_axis_points(marker_side_length):
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

# ==============================================================================
# 3. VIDEO-VERARBEITUNG mit FUSION
# ==============================================================================

def process_video():
    global H_smooth_pose 
    cap = cv2.VideoCapture(VIDEO_FILE)
    parameters = cv2.aruco.DetectorParameters() 

    if not cap.isOpened():
        print(f"FEHLER: Konnte Videodatei '{VIDEO_FILE}' nicht öffnen.")
        return
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    MAP_HEIGHT = frame_height // 2 
    output_width = frame_width
    output_height = frame_height + MAP_HEIGHT 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_combi = cv2.VideoWriter(OUTPUT_FILE_COMBI, fourcc, fps, (output_width, output_height))
    
    print(f"Starte Tracking und Fusion. Ergebnis gespeichert in {OUTPUT_FILE_COMBI}")
    
    MAP_SIZE = frame_width
    MAP_SCALE = 5                  
    MAP_CENTER_X = MAP_SIZE // 2
    MAP_CENTER_Y = MAP_HEIGHT // 2
    
    axis_points_3d = get_axis_points(MARKER_SIDE_LENGTH)
    
    # KORREKTUR: Optimierte Beobachterperspektive (Zoom erhöht, Sichtfeld maximiert)
    K_map_perspective = np.array([[700, 0, MAP_CENTER_X], [0, 700, MAP_CENTER_Y], [0, 0, 1]], dtype=np.float32)
    rvec_map = np.array([[-0.5], [0.0], [0.0]], dtype=np.float32) 
    tvec_map = np.array([[-45.0], [10.0], [90.0]], dtype=np.float32) # 90 cm entfernt (Z), zentriert um X=-45


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_visual = frame.copy()
        map_image = np.zeros((MAP_SIZE // 2, MAP_SIZE, 3), dtype=np.uint8)
        map_image.fill(30) 
        
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, DICTIONARY, parameters=parameters)

        if ids is not None and len(ids) >= 2: 
            
            obj_points, img_points = get_obj_and_img_points(corners, ids, marker_data, BASE_CORNERS)
            
            if len(obj_points) >= 8:
                
                # Berechnung der ROHE Pose (World_T_Camera)
                retval, rvec_raw, tvec_raw = cv2.solvePnP(
                    obj_points, img_points, cameraMatrix, distCoeffs
                )
                
                if retval: 
                    
                    # 1. BERECHNUNG DES REPROJEKTIONSFEHLERS (VALISIERUNG)
                    img_points_reprojected, _ = cv2.projectPoints(obj_points, rvec_raw, tvec_raw, cameraMatrix, distCoeffs)
                    img_points_reprojected = img_points_reprojected.reshape(-1, 2)
                    error_pixels = np.sqrt(np.sum((img_points - img_points_reprojected)**2, axis=1))
                    rms_error = np.mean(error_pixels)
                    
                    
                    # 2. Glättung der Pose für den Pfad
                    H_raw = pose_to_matrix(rvec_raw, tvec_raw)
                    if H_smooth_pose is None:
                        H_smooth_pose = H_raw
                    else:
                        H_smooth_pose = (SMOOTHING_ALPHA * H_smooth_pose) + ((1.0 - SMOOTHING_ALPHA) * H_raw)
                    
                    tvec_smooth = H_smooth_pose[:3, 3]
                    rvec_smooth = cv2.Rodrigues(H_smooth_pose[:3, :3])[0]
                    R_smooth = H_smooth_pose[:3, :3]
                    
                    x = tvec_smooth[0]
                    y = tvec_smooth[1]
                    z = tvec_smooth[2]
                    
                    # 3. Visualisierung im ORIGINAL-Frame
                    
                    # Achsen auf ALLEN Markern zeichnen (an der Ecke)
                    rvecs_single, tvecs_single, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, MARKER_SIDE_LENGTH, cameraMatrix, distCoeffs
                    )

                    for i in range(len(ids)):
                        rvec_i = rvecs_single[i]
                        tvec_i = tvecs_single[i]
                        
                        img_points_axis, _ = cv2.projectPoints(axis_points_3d, rvec_i, tvec_i, cameraMatrix, distCoeffs)
                        img_points_axis = np.int32(img_points_axis).reshape(-1, 2)
                        
                        p_origin = tuple(img_points_axis[0])
                        p_x = tuple(img_points_axis[1])
                        p_y = tuple(img_points_axis[2])
                        p_z = tuple(img_points_axis[3])
                        
                        cv2.line(frame_visual, p_origin, p_x, (0, 0, 255), 3)  # Rot X
                        cv2.line(frame_visual, p_origin, p_y, (0, 255, 0), 3)  # Grün Y
                        cv2.line(frame_visual, p_origin, p_z, (255, 0, 0), 3)  # Blau Z
                    
                        # KORREKTUR: Marker-Ecke als BLAUER Punkt (Fixpunkt)
                        corner_point = corners[i][0][3].astype(np.int32) 
                        cv2.circle(frame_visual, tuple(corner_point), 5, (255, 0, 0), -1) 
                    
                    
                    # Textanzeige mit Genauigkeitswert
                    text_pos = f"Pos: X={x:.2f} Y={y:.2f} Z={z:.2f} cm (Ref ID 0)"
                    text_error = f"Fehler: {rms_error:.3f} px" 
                    
                    cv2.putText(frame_visual, text_pos, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame_visual, text_error, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) 
                    
                    # Pfadverfolgung (Der glatte Graph im ORIGINAL-FRAME)
                    CAMERA_PATH.append(np.array([x, y, 0.0], dtype=np.float32))
                    
                    if len(CAMERA_PATH) > 1:
                        path_points_3d = np.array(CAMERA_PATH).reshape(-1, 3)
                        path_points_2d, _ = cv2.projectPoints(
                            path_points_3d, rvec_smooth, tvec_smooth, cameraMatrix, distCoeffs
                        )
                        path_points_2d = np.int32(path_points_2d).reshape(-1, 2)
                            
                    # 4. Visualisierung in der 2D-Draufsicht (Karte)
                    
                    # a) Marker als 3D-Rechtecke projizieren
                    for marker_id, pos_center in marker_data.items():
                        
                        marker_corners_world = BASE_CORNERS + pos_center
                        
                        # Projiziere die 3D-Ecken in die 2D-Kartenperspektive
                        marker_points_map, _ = cv2.projectPoints(
                            marker_corners_world, rvec_map, tvec_map, K_map_perspective, None
                        )
                        marker_points_map = np.int32(marker_points_map).reshape(-1, 2)
                        
                        # Zeichne die Markerrechtecke (Festpunkte)
                        cv2.polylines(map_image, [marker_points_map], isClosed=True, color=(255, 0, 0), thickness=2) 

                        # Zeichne den Mittelpunkt (rot)
                        center_map = np.mean(marker_points_map, axis=0).astype(np.int32)
                        cv2.circle(map_image, tuple(center_map), 4, (0, 0, 255), -1) 
                        
                        
                    # b) Kamera-Frustum projizieren
                    frustum_world = []
                    for p in FRUSTUM_BASE_POINTS:
                        p_hom = np.array([p[0], p[1], p[2], 1.0])
                        p_world_hom = H_smooth_pose @ p_hom 
                        frustum_world.append(p_world_hom[:3])
                        
                    frustum_world = np.array(frustum_world)
                    
                    # Projiziere das Frustum in die 2D-Kartenperspektive
                    frustum_points_map, _ = cv2.projectPoints(frustum_world, rvec_map, tvec_map, K_map_perspective, None)
                    frustum_2d_map = np.int32(frustum_points_map).reshape(-1, 2)
                    
                    if len(frustum_2d_map) == 5:
                        p_orig = frustum_2d_map[0]
                        p_tl = frustum_2d_map[3] 
                        p_tr = frustum_2d_map[4] 
                        
                        # Fülle die Fläche des Sichtkegels (Polygon)
                        polygon_points = np.array([frustum_2d_map[1:]], np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(map_image, [polygon_points], (150, 150, 150)) 
                        
                        # Zeichne die Linien der Kameraform (Rechteck)
                        cv2.polylines(map_image, polygon_points, isClosed=True, color=(255, 255, 255), thickness=1)
                        
                        # Verbinde den Ursprung zur Sicht (Simulation des Sichtkegels)
                        cv2.line(map_image, p_orig, p_tl, (150, 150, 150), 1)
                        cv2.line(map_image, p_orig, p_tr, (150, 150, 150), 1)
                        cv2.circle(map_image, p_orig, 4, (0, 255, 255), -1) 

                    # c) Pfad zeichnen (Projektion des geglätteten Pfades)
                    if len(CAMERA_PATH) > 1:
                        path_points_map, _ = cv2.projectPoints(path_points_3d, rvec_map, tvec_map, K_map_perspective, None)
                        path_points_map = np.int32(path_points_map).reshape(-1, 2)
                        
                        for i in range(1, len(path_points_map)):
                            cv2.line(map_image, tuple(path_points_map[i-1]), tuple(path_points_map[i]), (0, 255, 0), 2) 

                    # Textanzeige
                    text_map = f"Fehler: {rms_error:.3f} px" 
                    cv2.putText(map_image, text_map, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        # 5. Endgültige Fusion und Speicherung
        
        if frame_visual.shape[1] != map_image.shape[1]:
            map_image = cv2.resize(map_image, (frame_width, MAP_HEIGHT))
            
        combined_frame = np.vstack((map_image, frame_visual))
        
        out_combi.write(combined_frame) 
        
        cv2.imshow('Kombinierte Ansicht (Map + Video)', combined_frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_combi.release() 
    cv2.destroyAllWindows()
    print(f"Video-Verarbeitung beendet. Ergebnis gespeichert in {OUTPUT_FILE_COMBI}")

if __name__ == "__main__":
    process_video()