import numpy as np
import math
from process_video import pose_to_matrix, invert_pose_matrix, rotmat_to_quat, quat_to_rotmat, slerp

print('Running quick pose helper tests...')

# Test 1: inversion round-trip
R = np.eye(3)
t = np.array([[10.0],[5.0],[3.0]])
# make fake rvec from R (Rodrigues expects a rotation vector but we will just use R)
# We'll create H as world->camera using t
H = np.eye(4)
H[:3,:3] = R
H[:3,3] = t.flatten()
H_inv = invert_pose_matrix(H)
# transform a point
p_world = np.array([1.0,2.0,3.0,1.0])
p_cam = H @ p_world
p_world_rt = H_inv @ p_cam
if np.allclose(p_world_rt, p_world):
    print('Invert test: PASS')
else:
    print('Invert test: FAIL', p_world_rt, p_world)

# Test 2: SLERP between identity and 90deg yaw
angle0 = 0.0
angle1 = math.pi/2
# rotation matrices
R0 = np.array([[math.cos(angle0), -math.sin(angle0),0],[math.sin(angle0), math.cos(angle0),0],[0,0,1]])
R1 = np.array([[math.cos(angle1), -math.sin(angle1),0],[math.sin(angle1), math.cos(angle1),0],[0,0,1]])
q0 = rotmat_to_quat(R0)
q1 = rotmat_to_quat(R1)
q_mid = slerp(q0, q1, 0.5)
R_mid = quat_to_rotmat(q_mid)
# extract yaw
yaw_mid = math.atan2(R_mid[1,0], R_mid[0,0])
print('Expected ~45deg (0.7854), got', yaw_mid)
if abs(yaw_mid - (math.pi/4)) < 0.02:
    print('SLERP test: PASS')
else:
    print('SLERP test: FAIL')
