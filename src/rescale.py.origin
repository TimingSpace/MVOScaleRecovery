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



class ScaleEstimator:
    def __init__(self,absolute_reference,window_size=6):
        self.absolute_reference = absolute_reference
        self.camera_pitch       = 0
        self.scale  = 1
        self.inliers = None
        self.scale_queue = deque()
        self.window_size = window_size
        self.vanish =  185
    def initial_estimation(self,motion_matrix):

        return 0
    def visualize_distance(self,feature3d,feature2d,img):
        near = np.min(feature3d[:,2])
        far   = np.max(feature3d[:,2])
        print(near,far)
        for i in range(feature3d.shape[0]):
            pos_y_norm = (feature3d[i,2]-near)/(far-near)
            cv2.circle(img,(int(feature2d[i,0]),int(feature2d[i,1])),3,(255*pos_y_norm,0,255-255*pos_y_norm),-1)
        cv2.imshow('img',img)
        cv2.waitKey()

    def visualize_feature(self,feature3d,feature2d,img):
        heighest = np.min(feature3d[:,1])
        lowest   = np.max(feature3d[:,1])
        print(heighest,lowest)
        for i in range(feature3d.shape[0]):
            pos_y_norm = (feature3d[i,1]-heighest)/(lowest-heighest)
            if feature3d[i][1]>0:
                cv2.circle(img,(int(feature2d[i,0]),int(feature2d[i,1])),3,(int(255*pos_y_norm),0,int(255-255*pos_y_norm)),-1)
            else:
                cv2.circle(img,(int(feature2d[i,0]),int(feature2d[i,1])),3,(int(255*pos_y_norm),0,int(255-255*pos_y_norm)),-1)

    def visualize(self,feature3d,feature2d,img):
        #lower_label = feature3d[:,1]>0
        #feature3d = feature3d[lower_label,:]
        #feature2d = feature2d[lower_label,:]
        tri = Delaunay(feature2d)
        triangle_ids = tri.simplices
        b_matrix = np.matrix(np.ones((3,1),np.float))
        #calculating the geometry model of each triangle
        datas =[]
        for triangle_id in triangle_ids:
            data=[]
            point_selected = feature3d[triangle_id]
            a_array = np.array(point_selected)
            a_matrix = np.matrix(a_array)
            a_matrix_inv = a_matrix.I
            norm = a_matrix_inv*b_matrix
            norm_norm_2 = norm.T*norm#the orm of norm
            height = 1/math.sqrt(norm_norm_2)
            norm   = norm/math.sqrt(norm_norm_2)
            if norm[1,0]<0:
                norm = -norm
                height = -height
            data = [norm[0,0],norm[1,0],norm[2,0],height]
            #print(data)
            datas.append(data)
        datas = np.array(datas)
        datas_80 = datas[datas[:,1]>0.9847,:]

        print(np.median(datas_80[:,3]))
        precomput_h = 0.9*np.median(datas_80[:,3])
        datas_low = datas_80[datas_80[:,3]>precomput_h,:]
        print(datas.shape[0],datas_80.shape[0],datas_low.shape[0])

        #plt.hist(feature3d[feature3d[:,1]>0,1])       
        #np.savetxt('test_norm_height.txt',datas)
        #plt.show()
        for i in range(0,triangle_ids.shape[0]):
            triangle_id = triangle_ids[i]
            triangle_points =  np.array(feature2d[triangle_id],np.int32)
            pts = triangle_points.reshape((-1,1,2))
            #color = list(np.abs(datas[i,0:3]*255).astype(np.int32))
            pitch = math.asin(datas[i,1])
            pitch_deg = pitch*180/3.1415926
            color = datas[i,0:3]
            if pitch_deg>80 and datas[i,3]>precomput_h:
                color=[0,255,0]
                height = datas[i,3]
                color = (abs(int(color[0]*255)),abs(int(color[1]*255)),abs(int(color[2]*255)))
                cv2.polylines(img,[pts],True,color)
                #cv2.fillPoly(img,[pts],color)
            elif datas[i,3]<precomput_h:
                color=[0,0,255]
                height = datas[i,3]
                color = (abs(int(color[0]*255)),abs(int(color[1]*255)),abs(int(color[2]*255)))
                cv2.polylines(img,[pts],True,color)
                #cv2.fillPoly(img,[pts],color)


            else:
                color=[255,0,0]
                height = datas[i,3]
                color = (abs(int(color[0]*255)),abs(int(color[1]*255)),abs(int(color[2]*255)))
                cv2.polylines(img,[pts],True,color)
                #cv2.fillPoly(img,[pts],color)


    '''
    check the three vertice in triangle whether satisfy
    (d_1-d_2)*(v_1-v_2)<=0 if not they are outlier
    True means inlier
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


    def scale_calculation(self,feature3d,feature2d,img=None):
        
        lower_feature_ids = feature2d[:,1]>self.vanish
        feature2d = feature2d[lower_feature_ids,:]
        feature3d = feature3d[lower_feature_ids,:]
        tri = Delaunay(feature2d)
        triangle_ids = tri.simplices
        outliers = self.find_outliers(feature3d,feature2d,triangle_ids)

        print('feature rejected ',np.sum(outliers<0))
        print('feature left     ',np.sum(outliers>=0))
        if(np.sum(outliers==1)>10):
            feature2d = feature2d[outliers>=0,:]
            feature3d = feature3d[outliers>=0,:]
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
        data_sub = data_sub[data[:,1]<self.camera_pitch-80]#>80 deg
        data_sub = data_sub[data_sub[:,2]>0]#under
        precomput_h = 0.9*np.median(data_sub[:,2])
        data_sub_low = data_sub[data_sub[:,2]>precomput_h,:]
        data_id = []
        # collect suitable points and triangle
        for i in range(0,triangle_ids.shape[0]):
            triangle_id = triangle_ids[i]
            pitch_deg = data[i,1]
            height = data[i,2]
            triangle_points =  np.array(feature2d[triangle_id],np.int32)
            if(pitch_deg>self.camera_pitch-95 and pitch_deg<self.camera_pitch-85):
                if(height>precomput_h):
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
            print('inilier',inliers.shape[0])
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
