import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import cv2
import math
import sys

frame_count = 1

image_name_list_file = sys.argv[1]
feature_pos_path = sys.argv[2]
image_name_list = open(image_name_list_file)
image_name = image_name_list.readline()
while frame_count<4541:
    image_name = image_name_list.readline()
    image_name=image_name[:-1]
    img = cv2.imread(image_name)
    feature_pos_name = feature_pos_path+"/"+str(frame_count)+".txt"
    points3d = np.loadtxt(feature_pos_path+str(frame_count)+".txt")
    camera_focus=718.856
    camera_cx=607.1928
    camera_cy=182.2157
    points = points3d[:,0:2]
    tri = Delaunay(points)
    triangle_ids = tri.simplices
    b_matrix = np.matrix(np.ones((3,1),np.float))
    data = []
    figure = plt.figure()
    for triangle_id in triangle_ids:
        point_selected = points3d[triangle_id]
        a_array = np.array(point_selected)
        a_array[:,0] = a_array[:,2]*(a_array[:,0]-camera_cx)/camera_focus
        a_array[:,1] = a_array[:,2]*(a_array[:,1]-camera_cy)/camera_focus
        a_matrix = np.matrix(a_array)
        #print a_matrix
        a_matrix_inv = a_matrix.I
        norm = a_matrix_inv*b_matrix
        norm_norm_2 = norm.T*norm
        pitch = math.asin(norm[1,0]/math.sqrt(norm_norm_2[0,0]))
        height = np.mean(a_array[:,1])
        pitch_deg = abs(pitch)*180/3.1415926
        #print norm[1,0]/math.sqrt(norm_norm_2[0,0]),pitch_deg,height
        pitch_height = [norm[1,0]/math.sqrt(norm_norm_2[0,0]),pitch_deg,height]
        data.append(pitch_height)
        if(pitch_deg>80 and pitch_deg<100):
            if(height>0):
                polygon = plt.Polygon(points[triangle_id],fill='g',color='g')
            else:
                polygon = plt.Polygon(points[triangle_id],fill='r',color='r')
        else:
            polygon = plt.Polygon(points[triangle_id],fill=None,color='b')
        plt.gca().add_patch(polygon)
    data = np.array(data)
    data = data[data[:,0]>0.98]
    data = data[data[:,2]>0]

    mean = np.mean(data[:,2])
    std  = np.std(data[:,2])

    data = data[data[:,2]>mean - 3*std]
    data = data[data[:,2]<mean + 3*std]
    final_mean = np.mean(data[:,2])
    print final_mean
#    plt.imshow(img)
    plt.savefig("result/result"+str(frame_count) +".png")
    #plt.show()
    plt.close(figure)
    frame_count = frame_count+1
