import numpy as np
import math
from process_video import invert_pose_matrix, rotmat_to_quat, quat_to_rotmat, slerp


def test_invert_pose_roundtrip():
    H = np.eye(4)
    H[:3, :3] = np.eye(3)
    H[:3, 3] = np.array([10.0, 5.0, 3.0])
    H_inv = invert_pose_matrix(H)
    p = np.array([1.0, 2.0, 3.0, 1.0])
    p_cam = H @ p
    p_world_rt = H_inv @ p_cam
    assert np.allclose(p_world_rt, p)


def test_slerp_midpoint_yaw():
    angle0 = 0.0
    angle1 = math.pi / 2
    R0 = np.array([[math.cos(angle0), -math.sin(angle0), 0], [math.sin(angle0), math.cos(angle0), 0], [0, 0, 1]])
    R1 = np.array([[math.cos(angle1), -math.sin(angle1), 0], [math.sin(angle1), math.cos(angle1), 0], [0, 0, 1]])
    q0 = rotmat_to_quat(R0)
    q1 = rotmat_to_quat(R1)
    qmid = slerp(q0, q1, 0.5)
    Rmid = quat_to_rotmat(qmid)
    yaw_mid = math.atan2(Rmid[1, 0], Rmid[0, 0])
    assert abs(yaw_mid - (math.pi / 4)) < 0.02
