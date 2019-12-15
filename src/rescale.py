from scipy.spatial import Delaunay
from estimate_road_norm import *
import numpy as np
import math
from collections import deque 
# recover ego-motion scale by road geometry information
# @author xiangwei(wangxiangwei.cpp@gmail.com) and zhanghui()
# 

# @function: calcualte the road pitch angle from motion matrix
# @input: the tansformation matrix in SE(3) 
# @output:translation angle calculate by t in R^3 and 
# rotarion angle calculate from rotation matrix R in SO(3)





class ScaleEstimator:
    def __init__(self,absolute_reference,window_size=6):
        self.absolute_reference = absolute_reference
        self.camera_pitch       = 0
        self.scale  = 1
        self.inliers = None
        self.scale_queue = deque()
        self.window_size = window_size
    def initial_estimation(self,motion_matrix):

        return pitch_angle,rotation_angle

    def scale_calculation(self,feature3d,feature2d):
        tri = Delaunay(feature2d)
        triangle_ids = tri.simplices
        b_matrix = np.matrix(np.ones((3,1),np.float))
        data = []
        #calculating the geometry model of each triangle
        for triangle_id in triangle_ids:
            point_selected = feature3d[triangle_id]
            a_array = np.array(point_selected)
            a_matrix = np.matrix(a_array)
            a_matrix_inv = a_matrix.I
            norm = a_matrix_inv*b_matrix
            norm_norm_2 = norm.T*norm#the square norm of norm
            height = 1/math.sqrt(norm_norm_2)
            if norm[1,0]<0:
                norm = -norm
                height = -height
            pitch = math.asin(-norm[1,0]/math.sqrt(norm_norm_2[0,0]))
            pitch_deg = pitch*180/3.1415926
            pitch_height = [norm[1,0]/math.sqrt(norm_norm_2[0,0]),pitch_deg,height]
            data.append(pitch_height)
        data = np.array(data) # all data is saved here

        # initial select by prior information
        data_sub = data[data[:,1]>self.camera_pitch-95]#>80 deg
        data_sub = data[data[:,1]<self.camera_pitch-85]#>80 deg
        data_sub = data_sub[data_sub[:,2]>0]#under
        data_id = []
        # collect suitable points and triangle
        for i in range(0,triangle_ids.shape[0]):
            triangle_id = triangle_ids[i]
            pitch_deg = data[i,1]
            height = data[i,2]
            triangle_points =  np.array(feature2d[triangle_id],np.int32)
            if(pitch_deg>self.camera_pitch-95 and pitch_deg<self.camera_pitch-85):
                if(height>0):
                        data_id.append(triangle_id[0])
                        data_id.append(triangle_id[1])
                        data_id.append(triangle_id[2])
                        pts = triangle_points.reshape((-1,1,2))
        point_selected = feature3d[data_id]
        self.initial_points = feature2d[data_id]

        # initial ransac
        if(point_selected.shape[0]>=12):
            # ransac
            a_array = np.array(point_selected)
            m,b = get_pitch_ransac(a_array,30,0.005)
            inlier_id = get_inliers(m,feature3d[:,:],0.01)
            inliers = feature3d[inlier_id,:]
            inliers_2d = feature2d[inlier_id,:]
            self.inliers = inliers_2d
            outliers_2d = feature2d[inlier_id==False,:]

            road_model_ransac = np.matrix(m)
            norm = road_model_ransac[0,0:-1]
            h_bar = -road_model_ransac[0,-1]
            if norm[0,1]<0:
                norm = -norm
                h_bar = -h_bar
            norm_norm_2 = norm*norm.T
            norm_norm = math.sqrt(norm_norm_2)/h_bar
            norm = norm/h_bar
            ransac_camera_height = 1/norm_norm
            pitch = math.asin(norm[0,1]/norm_norm)
            scale = self.absolute_reference/ransac_camera_height
            self.scale = scale 
        self.scale_queue.append(self.scale)
        if len(self.scale_queue)>self.window_size:
            self.scale_queue.popleft()
        return np.mean(self.scale_queue)


def main():
    # get initial motion pitch by motion matrix
    # triangle region norm and height
    # selected
    # calculate the road norm
    # filtering
    # scale recovery
    camera_height = 1.7
    scale_estimator = ScaleEstimator(camera_height)
    scale = scale_estimator.scale_calculation()

if __name__ == '__main__':
    main()
