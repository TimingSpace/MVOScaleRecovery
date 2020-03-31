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

class Reconstruct:
    def __init__(self):
        self.threshold = 1

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
        
        datas_x  = datas[(datas[:,0]>0.9847)|(datas[:,0]<-0.9847),:]
        datas_z  = datas[(datas[:,2]>0.9847)|(datas[:,2]<-0.9847),:]

        data_label = np.zeros(datas.shape[0])
        data_label += 1*((datas[:,0]>0.9847)|(datas[:,0]<-0.9847)).astype(int)
        data_label += 2*((datas[:,1]>0.9847)|(datas[:,1]<-0.9847)).astype(int)
        data_label += 3*((datas[:,2]>0.9847)|(datas[:,2]<-0.9847)).astype(int)
        for i in range(0,triangle_ids.shape[0]):
            triangle_id = triangle_ids[i]
            triangle_points =  np.array(feature2d[triangle_id],np.int32)
            pts = triangle_points.reshape((-1,1,2))
            #color = list(np.abs(datas[i,0:3]*255).astype(np.int32))
            pitch = math.asin(datas[i,1])
            pitch_deg = pitch*180/3.1415926
            color = datas[i,0:3]
            if data_label[i]==1:
                color=[255,0,0]
                color = (abs(int(color[0]*255)),abs(int(color[1]*255)),abs(int(color[2]*255)))
                cv2.polylines(img,[pts],True,color)
                #cv2.fillPoly(img,[pts],color)
            elif data_label[i]==2:
                color=[0,255,0]
                color = (abs(int(color[0]*255)),abs(int(color[1]*255)),abs(int(color[2]*255)))
                cv2.polylines(img,[pts],True,color)
                #cv2.fillPoly(img,[pts],color)
            elif data_label[i]==3:
                color=[0,0,255]
                color = (abs(int(color[0]*255)),abs(int(color[1]*255)),abs(int(color[2]*255)))
                cv2.polylines(img,[pts],True,color)
            else:
                color=[0,0,0]
                height = datas[i,3]
                color = (abs(int(color[0]*255)),abs(int(color[1]*255)),abs(int(color[2]*255)))
                cv2.polylines(img,[pts],True,color)
                #cv2.fillPoly(img,[pts],color)

    
def main():
    # get initial motion pitch by motion matrix
    # triangle region norm and height
    # selected
    # calculate the road norm
    # filtering
    # scale recovery
    print('test')

if __name__ == '__main__':
    main()
