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
import open3d as o3d

class Reconstruct:
    def __init__(self,cam):
        self.threshold = 1
        self.cam = cam
        pixel_u=np.array(list(range(0,cam.width))*cam.height).reshape(cam.height,cam.width) 
        pixel_v=np.array(list(range(0,cam.height))*cam.width).reshape(cam.width,cam.height) 
        pixel_v=pixel_v.transpose()
        pixel = np.stack((pixel_u,pixel_v))
        pixel = pixel.transpose((1,2,0)).reshape(-1,2)
        self.pixel_ori = pixel.copy()

        pixel =pixel.astype(float)
        pixel[:,0] = (pixel[:,0]-self.cam.cx)/self.cam.fx
        d = (pixel[:,0]-self.cam.cx)/self.cam.fx

        pixel[:,1] = (pixel[:,1]-self.cam.cy)/self.cam.fy
        self.pixel = pixel.reshape(cam.height,cam.width,2)
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
    def triangle_model(self,feature3d,triangle_ids):
        datas =[]
        b_matrix = np.matrix(np.ones((3,1),np.float))
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
        return datas
    def depth_generate(self,tri,datas,img):
        depth_img = np.zeros((self.cam.height,self.cam.width,3))
        pixel_tris = tri.find_simplex(self.pixel_ori)
        self.pixel_tris = pixel_tris.reshape(self.cam.height,self.cam.width)
        point_clouds =[]
        colors       =[]
        for v in range(0,self.cam.height):
            for u in range(0,self.cam.width):
                tri_id = self.pixel_tris[v,u]
                if tri_id != -1:
                    pixel = self.pixel[v,u]
                    norm_tri = datas[tri_id,0:3]
                    h_tri    = datas[tri_id,3]
                    depth = h_tri/(norm_tri[0]*pixel[0]+norm_tri[1]*pixel[1]+norm_tri[2])
                    if(depth)<0:
                        print('there should be an error',depth,pixel,datas[tri_id])
                    depth_img[v,u] =depth
                    point = [pixel[0]*depth,pixel[1]*depth,depth]
                    point_clouds.append(point)
                    colors.append(img[v,u,::-1]/255.0)
        depth_img = depth_img/np.max(depth_img)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(point_clouds))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        o3d.io.write_point_cloud("pointcloud.ply", pcd)
        cv2.imshow('img_depth',depth_img/np.max(depth_img))
        print(np.min(depth_img),np.max(depth_img),np.mean(depth_img))

    def visualize_triangle(self,feature2d,triangle_ids,datas,img):
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

    def visualize_feature(self,feature3d,feature2d,img):
        near = np.min(feature3d[:,2])
        far   = np.max(feature3d[:,2])
        #print(near,far)
        for i in range(feature3d.shape[0]):
            pos_y_norm = (feature3d[i,2]-near)/(far-near)
            color=(255*pos_y_norm,0,255-255*pos_y_norm)
            cv2.circle(img,(int(feature2d[i,0]),int(feature2d[i,1])),3,color,-1)



    def visualize(self,feature3d,feature2d,img):
        print('feature total   ',feature3d.shape[0])
        img_c = img.copy()
        lower_label = feature3d[:,1]>0
        feature3d = feature3d[lower_label,:]
        feature2d = feature2d[lower_label,:]
        print('feature lower   ',feature3d.shape[0])
        tri = Delaunay(feature2d)
        triangle_ids = tri.simplices
        datas = self.triangle_model(feature3d,triangle_ids)
        self.visualize_triangle(feature2d,triangle_ids,datas,img)
        self.visualize_feature(feature3d,feature2d,img)       

        cv2.imshow('img',img)
        outliers = self.find_outliers(feature3d,feature2d,triangle_ids)
        print('feature rejected ',np.sum(outliers<0))
        print('feature left     ',np.sum(outliers>=0))
        #calculating the geometry model of each triangle
        feature2d = feature2d[outliers>=0,:]
        feature3d = feature3d[outliers>=0,:]
        tri = Delaunay(feature2d)
        triangle_ids = tri.simplices
        datas = self.triangle_model(feature3d,triangle_ids)
        #self.depth_generate(tri,datas,img_c)
        self.visualize_triangle(feature2d,triangle_ids,datas,img_c)
        self.visualize_feature(feature3d,feature2d,img_c)       
        cv2.imshow('img_no_outlier',img_c)
        key = cv2.waitKey()
        if key&255 == ord('q'):
            return False
        return True

        
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
