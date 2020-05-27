# recover ego-motion scale by road geometry information
# @author xiangwei(wangxiangwei.cpp@gmail.com) and zhanghui()
# 

# @function: calcualte the road pitch angle from motion matrix
# @input: the tansformation matrix in SE(3) 
# @output:translation angle calculate by t in R^3 and 
# rotarion angle calculate from rotation matrix R in SO(3)



from scipy.spatial import Delaunay
from estimate_road_norm import *
import numpy as np
import math
from collections import deque 
import matplotlib.pyplot as plt
import scale_calculator as sc
import graph 
import param

class ScaleEstimator:
    def __init__(self,absolute_reference,window_size=6):
        self.absolute_reference = absolute_reference
        self.camera_pitch       = 0
        self.scale  = 1
        self.inliers = None
        self.scale_queue = deque()
        self.window_size = window_size
        self.vanish =  185
        self.sc = sc.ScaleEstimator(absolute_reference,window_size)
        self.gc = graph.GraphChecker([[3,1],[2,2],[2,2],[0,4]])
        self.gs = graph.GraphGrow()
        self.img_w =  param.img_w
        self.img_h =  param.img_h
    def initial_estimation(self,motion_matrix):

        return 0

    '''
    check the three vertice in triangle whether satisfy
    (d_1-d_2)*(v_1-v_2)<=0 if not they are outlier
    True means outlier
    '''
    def check_triangle(self,v,d):
        flag=[False,False,False]
        a = (v[0]-v[1])*(d[0]-d[1])
        b = (v[0]-v[2])*(d[0]-d[2])
        c = (v[1]-v[2])*(d[1]-d[2])
        if a>0:
            flag[0]=True
            flag[1]=True
        if b>0:
            flag[0]=True
            flag[1]=True
        if c>0:
            flag[1]=True
            flag[2]=True
        return flag

    def find_outliers(self,feature3d,feature2d,triangle_ids):
        # suppose every is inlier
        outliers = np.ones((feature3d.shape[0]))
        for triangle_id in triangle_ids:
            data=[]
            depths   = feature3d[triangle_id,2]
            pixel_vs = feature2d[triangle_id,1]
            flag    = self.check_triangle(pixel_vs,depths)
            outlier = triangle_id[flag] 
            outliers[outlier]-=np.ones(outliers[outlier].shape[0])
        
        return outliers
    
    
    def flat_selection(self,feature3d,triangle_ids):    
        self.b_matrix = np.ones((3,1),np.float)
           #calculating the geometry model of each triangle
        triangles = np.array([np.matrix(feature3d[triangle_id]) for triangle_id in triangle_ids])
        triangles_i = np.array([np.matrix(feature3d[triangle_id]).I for triangle_id in triangle_ids])
        normals     = (triangles_i@self.b_matrix).reshape(-1,3)
        normals_len = np.sqrt(np.sum(normals*normals,1)).reshape(-1,1)
        normals     = normals/normals_len
        pitch_deg   = np.arcsin(-normals[:,1])*180/np.pi #[-90,90]
        
        valid_pitch_id = pitch_deg<-80
        valid_pitch_id_tight = pitch_deg<-85
        
        print('triangle left ',np.sum(valid_pitch_id),'from',valid_pitch_id.shape[0])
        heights     = (1/normals_len).reshape(-1)
        unvalid_pitch_id = pitch_deg>=-80
        height_level = 0.9*(np.median(heights[valid_pitch_id]))
        self.height_level = height_level
        print('height level',height_level)
        valid_height_id = heights>height_level
        print(len(valid_height_id))
        valid_id = valid_pitch_id_tight & valid_height_id
        print('triangle left final',np.sum(valid_id),'from',valid_id.shape[0])
        #valid_id = valid_pitch_id
        #valid_id = self.gs.process(triangle_ids,heights,pitch_deg)
        #valid_points_id = np.unique(triangle_ids[valid_id].reshape(-1))
        valid_points_id = triangle_ids[valid_id].reshape(-1)
        return list(valid_points_id),heights[valid_pitch_id]
    
    '''
    @func:select features on road
    @input: feature3d nx3 features relative 3d coordinate
    @input: feature2d nx2 features pixel coordinate
    1. remove feature above vanishing point
    2. remove feature not fit (v1-v2)(d1-d2)<0
    3. remove unflat feature and higher feature
    '''

    def feature_selection(self,feature3d,feature2d):
        # step 1: remove feature above vanishing point
        lower_feature_ids = feature2d[:,1]>self.vanish
        feature2d = feature2d[lower_feature_ids,:]
        feature3d = feature3d[lower_feature_ids,:]
        
        #distance_level = np.median(feature3d[:,2])
        #near_feature_ids = feature3d[:,2]<2*distance_level
        #feature2d = feature2d[near_feature_ids,:]
        #feature3d = feature3d[near_feature_ids,:]
        # step 2：remove feature by fit (v1-v2)(d1-d2)<0 
        tri = Delaunay(feature2d)
        triangle_ids = tri.simplices
        #valid_id     = self.sc.find_reliability_by_graph(feature3d,feature2d,triangle_ids)
        valid_id  = self.gc.find_inliers(feature3d,feature2d,triangle_ids)
        #outliers = self.find_outliers(feature3d,feature2d,triangle_ids)
        #valid_id = (outliers>=0)
        #valid_id = []
        print('feature rejected ',np.sum(valid_id==False))
        print('feature left     ',np.sum(valid_id))
        if(np.sum(valid_id)>10):
            feature2d = feature2d[valid_id,:]
            feature3d = feature3d[valid_id,:]
            tri = Delaunay(feature2d)
            triangle_ids = tri.simplices
        # step 3 remove unflat and high
        data_id,h = self.flat_selection(feature3d,triangle_ids)
        point_selected = feature3d[data_id]
        '''
        ax1 = plt.subplot(111)
        ax1.plot(feature3d[:,2],-1/feature3d[:,1],'.g')
        ax1.plot(point_selected[:,2],-1/point_selected[:,1],'.y')
        ax1.set_ylim(-2,0)
        plt.show()
        '''
        return point_selected,h
        #return feature3d

    def scale_calculation_ransac(self,point_selected):
        if(point_selected.shape[0]>=12):
            # ransac
            a_array = np.array(point_selected)
            m,b = get_pitch_ransac(a_array,100,0.005)
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
#0.3 is the max accellerate
            if scale - self.scale >0.3:
                self.scale += 0.3 
            elif scale - self.scale<-0.3:
                self.scale -= 0.3
            else:
                self.scale = scale
        self.scale_queue.append(self.scale)
        if len(self.scale_queue)>self.window_size:
            self.scale_queue.popleft()
        return np.median(self.scale_queue),1
    def scale_calculation_static_tri(self,heights):

        if len(heights)>12:
            scale_norm,_,_ = self.sc.road_model_calculation_static_tri(heights)
            scale = scale_norm*self.absolute_reference
            self.scale =scale
            return scale,0
        else:
            return self.scale,0



    def scale_calculation(self,feature3d,feature2d,img=None):
        point_selected,h = self.feature_selection(feature3d,feature2d)
        return self.scale_calculation_ransac(point_selected)
        #return self.scale_calculation_static_tri(h)
        #return self.sc.scale_calculation_static(point_selected)
        # initial ransac
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
