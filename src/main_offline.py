'''
this code is modified from monovo
the processure is as follows:
1. read image(currently load offline images from the paths stored in sys.argv[1])
2. initial the camera parameter
3. feature detection and tracking 
4. calculate the camera motion and 3d coordinates of the tracked feature points
5. figure out which points is on the road and calculation the road model based on the points
6. got the scale parameter, and smooth it

# 3+4 is done by vo.update
# 5+6 is done by scale_estimator
'''
import sys
import numpy as np
import cv2
from thirdparty.MonocularVO.visual_odometry import PinholeCamera, VisualOdometry
#from scale_calculator import ScaleEstimator
#from rescale_test import ScaleEstimator
from rescale import ScaleEstimator
#from reconstruct import Reconstruct
import param

def main():
    real_scale=None
    data_path = sys.argv[1]
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# do what you did like  
    data = np.load(data_path)
    data = data.item()
    np.load = np_load_old
    tag = '.test_001'
    if len(sys.argv)>2:
        tag = sys.argv[2]
    #    real_scale = np.loadtxt(sys.argv[2])
    res_addr = 'evaluate_result/'+data_path.split('.')[-4].split('/')[-1]+'_'
    scale_estimator = ScaleEstimator(absolute_reference = param.camera_h,window_size=5)
    #reconstructer = Reconstruct(cam)
    image_id = 0
    path=[]
    scales=[]
    error =[]
    scale = 1
    path.append([1,0,0,0,0,1,0,0,0,0,1,0])
    scales.append(0)
    error.append(100)
    begin_id = 0
    
    end_id   =  None
    img_last = []
    motions    = data['motions']
    move_flags = data['move_flags']
    feature2ds = data['feature2ds']
    feature3ds = data['feature3ds']
    print(len(motions),len(move_flags),len(feature2ds))
    for move_flag in move_flags:
        if image_id<begin_id:
            image_id+=1
            continue
        if end_id is not None and image_id>end_id:
            break
        print(move_flag)
        if (not move_flag):
            scales.append(0)
            error.append(0)
            image_id+=1
            continue

        feature3d = feature3ds[image_id]
        feature2d = feature2ds[image_id]
        motion    = motions[image_id]
        if feature3d.shape[0]>param.minimum_feature_for_scale:
            pitch = scale_estimator.initial_estimation(motion[3:12:4].reshape(-1))
            scale,std = scale_estimator.scale_calculation(feature3d,feature2d)
            if real_scale is not None:
                print('predict,real,std',scale,real_scale[image_id-1],std)
            #if(np.abs(scale-real_scale[image_id-1])>0.3 and std<0.3):
            if(False):#  and abs(real_scale[image_id-1]-scale)>0.3):
                scale_estimator.check_full_distribution(feature3d.copy(),feature2d.copy(),real_scale[image_id-1],img_bgr)
                scale_estimator.plot_distribution(str(image_id),img_bgr)
            scales.append(scale)
            error.append(std)
        else:
            scales.append(scales[-1])
            error.append(error[-1])
        print('id  ', image_id,' scale ',scale)
        image_id+=1
    #np.savetxt(res_addr+'features.txt',scale_estimator.all_features)
    np.savetxt(res_addr+'scales.txt'+tag,scales[1:])
    print(res_addr)
    poses = get_path(np.array(motions),np.array(scales[1:]))
    np.savetxt(res_addr+'path.txt'+tag,poses)

def line2mat(line_data):
    mat = np.eye(4)
    mat[0:3,:] = line_data.reshape(3,4)
    return np.matrix(mat)

def motion2pose(data):
    data_size = data.shape[0]
    all_pose = np.zeros((data_size+1,12))
    temp = np.eye(4,4).reshape(1,16)
    all_pose[0,:] = temp[0,0:12]
    pose = np.matrix(np.eye(4,4))
    for i in range(0,data_size):
        data_mat = line2mat(data[i,:])
        pose = pose*data_mat
        pose_line = np.array(pose[0:3,:]).reshape(1,12)
        all_pose[i+1,:] = pose_line
    return all_pose


def get_path(motions,scales):
    motion_trans = motions[:,3:12:4]
    motion_trans = (scales*motion_trans.transpose()).transpose()
    motions[:,3:12:4] = motion_trans
    pose = motion2pose(motions)
    return pose


if __name__ == '__main__':
    main()


'''
 # reprojection error
            ref_warp = intrinsic_m@(vo.motion_R@vo.feature3d.transpose()+vo.motion_t)
            ref_warp = (ref_warp/ref_warp[2,:]).transpose()
            ref_warp = ref_warp[:,0:2]
            reproject_error = np.sum(np.abs(ref_warp - vo.px_ref_selected))/ref_warp.shape[0]
            error.append(reproject_error)
            print(reproject_error)

'''
