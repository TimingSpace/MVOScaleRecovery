import numpy as np
from scipy.spatial import Delaunay
import cv2
import math
import sys

import time
from estimate_road_norm import get_pitch
from estimate_road_norm import get_pitch_ransac
from estimate_road_norm import get_pitch_line_ransac
from estimate_road_norm import get_inliers
from estimate_road_norm import get_norm_svd
from estimate_road_norm import estimate

camera_focus=718.856
camera_cx=607.1928
camera_cy=185.2157
#camera_focus=707.0912
#camera_cx=601.8873
#camera_cy=183.1104

print 'python calculate_height_pitch.py image_name_list_file feature_pos_path motion_path pose_path'
#initialization
input_id = sys.argv[1]
input_date = sys.argv[2]
frame_number = int(sys.argv[3])
iter_number = int(sys.argv[4])
#image_name_list_file = 'input/kitti_image_'+input_id+'_list.txt'
feature_pos_path = 'result/kitti_'+input_id+'/kitti_'+input_id+'_feature_'+input_date+'/'
camera_motion_path = 'result/kitti_'+input_id+'/kitti_'+input_id+'_motion_'+input_date+'.txt'
camera_pose_path = 'result/kitti_'+input_id+'/kitti_'+input_id+'_pose_'+input_date+'.txt'
#image_name_list = open(image_name_list_file)
#image_name = image_name_list.readline()

begin_time = time.time()
for case_i in range(0,10):
    camera_motions = np.loadtxt(camera_motion_path)
    camera_motion_ts = camera_motions[:,3::4]

    camera_poses = np.loadtxt(camera_pose_path)
    camera_pose_ts = camera_poses[:,3::4]
#flag
    flag_save_images = False
    flag_save_data = False
    flag_demo = False

    ransac_camera_heights = []
    refined_camera_height_means = []
    refined_camera_height_stds = []
    refined_pitchs=[]
    refined_inlier_number=[]
    refined_camera_height_t_means = []
    inlier_numbers=[]
    norm_prev = np.array((3,1),np.float)
    norm_norm = 1
    norm_norm_prev = 1
    pitch_deg_prev = 1
    height_prev = 1
    frame_count = 1
    while frame_count<frame_number:
        if frame_count%1000==0:
            print '**********',input_id,'***',case_i,'**',frame_count,'*****************'
        # load image and feature points
        start_time = time.time()
        #image_name = image_name_list.readline()
        #image_name=image_name[:-1]
        #img = cv2.imread(image_name)
        #img_inlier_points = img.copy()
        feature_pos_name = feature_pos_path+"/"+str(frame_count)+".txt"
        points3d = np.loadtxt(feature_pos_path+str(frame_count)+".txt")
        if points3d.shape[0]==0:
            ransac_camera_heights.append(0)
            refined_camera_height_means.append(0)
            refined_camera_height_stds.append(0)
            refined_camera_height_t_means.append(0)
            refined_pitchs.append(0)
            inlier_numbers.append(0)
            frame_count= frame_count+1
            continue
        estimated_pitch = get_pitch(camera_motion_ts[0:frame_count+1,0:3])
        estimated_pitch_deg = estimated_pitch*180/3.1415926
        # triangulation
        points = points3d[:,0:2]
        points_3d_array = np.array(points3d)
        points_3d_array[:,0] = points_3d_array[:,2]*(points_3d_array [:,0]-camera_cx)/camera_focus
        points_3d_array[:,1] = points_3d_array[:,2]*(points_3d_array [:,1]-camera_cy)/camera_focus
        tri = Delaunay(points)
        triangle_ids = tri.simplices
        b_matrix = np.matrix(np.ones((3,1),np.float))
        #print("triangluation--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        data = []
        # road detection
        # calculate angles norm and height
        for triangle_id in triangle_ids:
            point_selected = points3d[triangle_id]
            a_array = np.array(point_selected)
            a_array[:,0] = a_array[:,2]*(a_array[:,0]-camera_cx)/camera_focus
            a_array[:,1] = a_array[:,2]*(a_array[:,1]-camera_cy)/camera_focus
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
        data = np.array(data)

        #print("triangle calculate--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        if flag_save_data:
            np.savetxt('result/data'+str(frame_count)+'.txt',data)
        # initial select by prior information
        data_sub = data[data[:,1]>estimated_pitch_deg-95]#>80 deg
        data_sub = data[data[:,1]<estimated_pitch_deg-85]#>80 deg
        data_sub = data_sub[data_sub[:,2]>0]#under
        data_id = []
        # collect suitable points and triangle
        for i in range(0,triangle_ids.shape[0]):
            triangle_id = triangle_ids[i]
            pitch_deg = data[i,1]
            height = data[i,2]
            triangle_points =  np.array(points[triangle_id],np.int32)
            if(pitch_deg>estimated_pitch_deg-95 and pitch_deg<estimated_pitch_deg-85):
                if(height>0):
                    if(1):
                        data_id.append(triangle_id[0])
                        data_id.append(triangle_id[1])
                        data_id.append(triangle_id[2])
                        pts = triangle_points.reshape((-1,1,2))
                        #cv2.polylines(img,[pts],True,(0,255,0))
                        #cv2.fillPoly(img,[pts],(0,255,0))
                    else:
                        pts = triangle_points.reshape((-1,1,2))
                        #cv2.polylines(img,[pts],True,(0,255,255))
                        #cv2.fillPoly(img,[pts],(0,255,255))
                else:
                        pts = triangle_points.reshape((-1,1,2))
                        #cv2.polylines(img,[pts],True,(0,0,255))
                        #cv2.fillPoly(img,[pts],(0,0,255))
            else:
                pts = triangle_points.reshape((-1,1,2))
                #cv2.polylines(img,[pts],True,(255,0,0))
        point_selected = points3d[data_id]
        if flag_save_data:
            np.savetxt('result/selected_points'+str(frame_count)+'.txt',point_selected)
        #print 'suitable triangle',data_sub.shape[0]
        #print 'suitable point :' ,point_selected.shape[0]

        #print("select triangle--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        # initial ransac
        if(point_selected.shape[0]>=12):
            # ransac
            a_array = np.array(point_selected)
            a_array[:,0] = a_array[:,2]*(a_array[:,0]-camera_cx)/camera_focus
            a_array[:,1] = a_array[:,2]*(a_array[:,1]-camera_cy)/camera_focus
            m,b = get_pitch_ransac(a_array,iter_number,0.005)
            #m_2,b_2 = get_pitch_line_ransac(a_array[:,1:],500,0.005)
            #print 'road model ransac',m_2
            #print 'inlier number',b_2
            inlier_id = get_inliers(m,a_array,0.01)
            #print inlier_id
            inliers = a_array[inlier_id,:]
            inlier_id_all = get_inliers(m,points_3d_array,0.01)
            inliers_2d = points[inlier_id_all,:]
            outliers_2d = points[inlier_id_all==False,:]

            road_model_ransac = np.matrix(m)
            norm = road_model_ransac[0,0:-1]
            h_bar = -road_model_ransac[0,-1]
            if norm[0,1]<0:
                norm = -norm
                h_bar = -h_bar
            norm_norm_2 = norm*norm.T
            norm_norm = math.sqrt(norm_norm_2)/h_bar
            norm = norm/h_bar
        else:
            norm = norm_prev.copy()
            norm_norm = norm_norm_prev
        ransac_camera_height = 1/norm_norm
        pitch = math.asin(norm[0,1]/norm_norm)
        pitch_deg = abs(pitch)*180/3.1415926

        ransac_camera_heights.append(ransac_camera_height)

        inlier_number = inliers.shape[0]
        inlier_numbers.append(inlier_number)

        #print("ransac--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        # refine n
        refined_norm = estimate(inliers)
        refined_norm = refined_norm[0:-1]
        if refined_norm[1]<0:
            refined_norm = -refined_norm
        refined_norm_matrix_temp = np.matrix(refined_norm)
        refined_norm_norm_2 = refined_norm_matrix_temp*refined_norm_matrix_temp.T
        refined_norm_norm=np.sqrt(refined_norm_norm_2)
        refined_norm = refined_norm/refined_norm_norm
        refined_norm_matrix = np.matrix(refined_norm)
        #print 'refined norm', refined_norm
        refined_pitch = math.asin(refined_norm[0,1])
        inliers_matrix = np.matrix(inliers)
        refined_pitchs.append(refined_pitch)
        # refine h
        refined_hs = inliers_matrix*refined_norm.T

        refined_h_mean = np.mean(refined_hs)
        refined_h_std = np.std(refined_hs)
        refined_camera_height_means.append(refined_h_mean)
        refined_camera_height_stds.append(refined_h_std)

        #print("refine--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        refined_h_ts = inliers[:,2]*math.sin(estimated_pitch)+inliers[:,1]*math.cos(estimated_pitch)
        refined_h_t_mean = np.mean(refined_h_ts)
        refined_camera_height_t_means.append(refined_h_t_mean)
        #demo
        #if flag_demo:
        #    for point2d in inliers_2d:
        #        cv2.circle(img_inlier_points,(int(point2d[0]),int(point2d[1])),3,(0,255,0),-1)
        #    for point2d in outliers_2d:
        #        cv2.circle(img_inlier_points,(int(point2d[0]),int(point2d[1])),3,(0,0,255),-1)
           # cv2.putText(img,'Road Model :'+str(round(norm[0,0],3))+'X+'+str(round(norm[0,1],4))+'Y+'+str(round(norm[0,2],3))+'Z-1=0',(50,50),1,1,(200,128,0),2)
            #cv2.putText(img,'Camera Height :'+str(round(refined_h_t_mean,3)),(50,80),1,1,(200,128,0),2)
            #cv2.putText(img,'Pitch :'+str(round(refined_pitch*180/3.1415926,3)),(50,110),1,1,(200,128,0),2)
            #cv2.putText(img,'Frame counter :'+str(round(frame_count,3)),(50,130),1,1,(200,128,0),2)

            #cv2.imshow("image",img)
            #cv2.imshow("image_inlier",img_inlier_points)
            #cv2.waitKey(1)
        #if flag_save_images:
            #cv2.imwrite("result/result"+str(frame_count)+".png",img)

        #system
        frame_count = frame_count+1
        norm_prev = norm.copy()
        norm_norm_prev = norm_norm
        height_prev = refined_h_mean
#save data

    ransac_camera_heights = np.array(ransac_camera_heights)
    np.savetxt('eval_ransac/result_heights_plane_ransac_'+input_id+'_'+input_date+'_'+str(iter_number)+'_'+str(case_i)+'.txt',ransac_camera_heights)

    refined_camera_height_means = np.array(refined_camera_height_means)
    np.savetxt('eval_ransac/refined_camera_height_means'+input_id+'_'+input_date+'_'+str(iter_number)+'_'+str(case_i)+'.txt''.txt',refined_camera_height_means)

    refined_camera_height_stds =np.array(refined_camera_height_stds)
    np.savetxt('eval_ransac/refined_camera_height_stds'+input_id+'_'+input_date+'_'+str(iter_number)+'_'+str(case_i)+'.txt''.txt',refined_camera_height_stds)

    refined_camera_height_t_means = np.array(refined_camera_height_t_means)
    np.savetxt('eval_ransac/refined_camera_height_t_means'+input_id+'_'+input_date+'_'+str(iter_number)+'_'+str(case_i)+'.txt''.txt',refined_camera_height_t_means)

    refined_pitchs = np.array(refined_pitchs)
    np.savetxt('eval_ransac/refined_pitch'+input_id+'_'+input_date+'_'+str(iter_number)+'_'+str(case_i)+'.txt''.txt',refined_pitchs)

    inlier_numbers = np.array(inlier_numbers)
    np.savetxt('eval_ransac/inlier_numbers'+input_id+'_'+input_date+'_'+str(iter_number)+'_'+str(case_i)+'.txt''.txt',inlier_numbers)

end_time = time.time()
np.savetxt('time'+input_id++str(iter_number)+'.txt',np.array([(end_time-begin_time)/(frame_number*10)]))
print 'ave_time', (end_time-begin_time)/frame_number
