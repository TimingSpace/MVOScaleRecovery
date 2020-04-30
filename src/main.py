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
from rescale import ScaleEstimator
#from reconstruct import Reconstruct
import param

def main():
    images_path = sys.argv[1]
    res_addr = 'result/'+images_path.split('.')[-2].split('/')[-1]+'_'
    print(images_path.split('.')[-2].split('/')[-1])
    images      = open(images_path)
    image_name = images.readline() # first line is not pointing to a image, so we read and skip it
    image_names= images.read().split('\n')
    #cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
    #cam = PinholeCamera(640.0, 480.0, 343.8560, 344.8560, 321.1928, 231.2157)
    #cam = PinholeCamera(1920.0, 1080.0, 960.0, 960.0, 960.0, 480.0)
    cam = PinholeCamera(param.img_w, param.img_h, param.img_fx, param.img_fy, param.img_cx, param.img_cy)
    intrinsic_m = np.array([[param.img_fx,0,param.img_cx],[0,param.img_fy,param.img_cy],[0,0,1]])
    vo = VisualOdometry(cam)
    scale_estimator = ScaleEstimator(absolute_reference = param.camera_h,window_size=10)
    #reconstructer = Reconstruct(cam)
    image_id = 0
    path=[]
    scales=[]
    error =[]
    scale = 1
    path.append([1,0,0,0,0,1,0,0,0,0,1,0])
    begin_id = 0
    img_last = []
    for image_name in image_names:
        if image_id<begin_id:
            image_id+=1
            continue
        print(image_name)
        if len(image_name) == 0:
            break
        img = cv2.imread(image_name,0)
        img = cv2.resize(img,(cam.width,cam.height))
        img_bgr = cv2.imread(image_name)
        img_bgr = cv2.resize(img_bgr,(cam.width,cam.height))
        vo.update(img,image_id)
        #print(vo.motion_t)       
        #print(vo.motion_R)
        if image_id>begin_id:
            feature2d = vo.feature3d[:,0:2].copy()
            feature2d[:,0] = feature2d[:,0]*cam.fx/vo.feature3d[:,2]+cam.cx
            feature2d[:,1] = feature2d[:,1]*cam.fx/vo.feature3d[:,2]+cam.cy
            # reprojection error
            ref_warp = intrinsic_m@(vo.motion_R@vo.feature3d.transpose()+vo.motion_t)
            ref_warp = (ref_warp/ref_warp[2,:]).transpose()
            ref_warp = ref_warp[:,0:2]
            reproject_error = np.sum(np.abs(ref_warp - vo.px_ref_selected))/ref_warp.shape[0]
            error.append(reproject_error)
            print(reproject_error)
            #np.savetxt('feature_3d.txt',vo.feature3d)
            print(vo.feature3d.shape)
            if vo.feature3d.shape[0]>param.minimum_feature_for_scale:
                scale = scale_estimator.scale_calculation(vo.feature3d,feature2d)
                # uncomment to visualize the feature and triangle
                #scale_estimator.visualize_distance(vo.feature3d,feature2d,img_bgr)
                #scale_estimator.visualize_distance(vo.feature3d,ref_warp,img_last)
                #re = reconstructer.visualize(vo.feature3d,feature2d,img_bgr)
                #if re==False:
                #    break
                R,t = vo.get_current_state(scale)
                M   = np.zeros((3,4))
                M[:,0:3]=R
                M[:,3]=t.reshape(-1)
                M = M.reshape(-1)
                path.append(M)
                scales.append(scale)
            else:
                path.append(path[-1])
                scales.append(scales[-1])
            print('id  ', image_id,' scale ',scale)
        img_last = img_bgr.copy()
        image_id+=1
        
    np.savetxt(res_addr+'path.txt',path)
    np.savetxt(res_addr+'scales.txt',scales)
    np.savetxt(res_addr+'error.txt',error)


if __name__ == '__main__':
    main()
