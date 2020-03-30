import numpy as np
from scipy.spatial import Delaunay
import cv2
import math
import sys
from thirdparty.Ransac.ransac import *

def augment(xyzs):
	axyz = np.ones((len(xyzs), 4))
	axyz[:, :3] = xyzs
	return axyz

def estimate(xyzs):
	axyz = augment(xyzs[:3])
	return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
	return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

def  get_norm_svd(camera_motion_ts):
    U, s, V = np.linalg.svd(camera_motion_ts.T, full_matrices=True)
    estimated_norm = U[:,2]
    if estimated_norm[1]<0:
        estimated_norm = -estimated_norm
    estimated_norm = np.matrix(estimated_norm)
    return estimated_norm

def  get_pitch_svd(camera_motion_ts):
    U, s, V = np.linalg.svd(camera_motion_ts.T, full_matrices=True)
    estimated_norm = U[:,2]
    if estimated_norm[1]<0:
        estimated_norm = -estimated_norm
    estimated_norm = np.matrix(estimated_norm)
    norm_norm_2 = estimated_norm*estimated_norm.T
    norm_norm=np.sqrt(norm_norm_2)
    estimated_pitch = math.asin(estimated_norm[0,1]/norm_norm_2)
    return estimated_pitch

def augment_line(xys):
	axy = np.ones((len(xys), 3))
	axy[:, :2] = xys
	return axy

def estimate_line(xys):
	axy = augment_line(xys[:2])
	return np.linalg.svd(axy)[-1][-1, :]

def is_inlier_line(coeffs, xy, threshold):
	return np.abs(coeffs.dot(augment_line([xy]).T)) < threshold


def get_pitch(camera_motion_ts):
    camera_motion = np.sum(camera_motion_ts,0)
    camera_motion = np.matrix(camera_motion)
    motion_norm_2 = camera_motion*camera_motion.T
    motion_norm=np.sqrt(motion_norm_2)
    estimated_pitch = math.asin(-camera_motion[0,1]/motion_norm_2)
    return estimated_pitch

def get_pitch_line_ransac(road_points,max_iterations,threshold):
    points_number = road_points.shape[0]
    goal_inliers = points_number*0.8
    m, b = run_ransac(road_points, estimate_line, lambda x, y: is_inlier_line(x, y, threshold), 2, goal_inliers, max_iterations)
    return m,b

def get_pitch_ransac(road_points,max_iterations,threshold):
    points_number = road_points.shape[0]
    goal_inliers = points_number*0.8
    m, b = run_ransac(road_points, estimate, lambda x, y: is_inlier(x, y, threshold), 3, goal_inliers, max_iterations)
    return m,b
def get_inliers(parameter,data,threshold):
    parameter = np.matrix(parameter)
    data = np.matrix(data)
    error = data*(parameter[0,0:-1].T)+parameter[0,-1]
    error = np.abs(np.array(error))
    error = error.T
    error=error[0]
    return error<threshold
camera_focus=718.856
camera_cx=607.1928
camera_cy=185.2157

#initialization
#camera_motion_path = sys.argv[1]
#camera_pose_path = sys.argv[2]

#camera_motions = np.loadtxt(camera_motion_path)
#camera_motion_ts = camera_motions[:,3::4]

#camera_poses = np.loadtxt(camera_pose_path)
#camera_pose_ts = camera_poses[:,3::4]
#estimated_pitchs = []
#frame_count = 1
#while frame_count<4541:
#    print '**********',frame_count,'*****************'
#    # load image and feature points
#    estimated_pitch = get_pitch(camera_motion_ts[0:frame_count+1,0:3])
#    estimated_pitchs.append(estimated_pitch)
#    print estimated_pitch*180/3.1415926
#    frame_count = frame_count+1
#estimated_pitchs = np.array(estimated_pitchs)
#np.savetxt('result/estimated_norms',estimated_pitchs)



