import sys
import numpy as np
import cv2
from thirdparty.monovo.visual_odometry import PinholeCamera, VisualOdometry
from rescale import ScaleEstimator

def main():
    images_path = sys.argv[1]
    images      = open(images_path)
    cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
    vo = VisualOdometry(cam)
    scale_estimator = ScaleEstimator(absolute_reference = 1.75)
    image_name = images.readline() # first line is not pointing to a image
    image_id = 0

    bg_img= np.zeros((600,600,3), dtype=np.uint8)
    while image_name!=None:
        image_name = images.readline()
        if len(image_name) == 0:
            break
        image_name=image_name[:-1]
        img = cv2.imread(image_name,0)
        img_bgr = cv2.imread(image_name)
        vo.update(img,image_id)
        
        if image_id>0:
            feature2d = vo.feature3d[:,0:2].copy()
            feature2d[:,0] = feature2d[:,0]*cam.fx/vo.feature3d[:,2]+cam.cx
            feature2d[:,1] = feature2d[:,1]*cam.fx/vo.feature3d[:,2]+cam.cy
            scale = scale_estimator.scale_calculation(vo.feature3d,feature2d)
            cur_R,cur_t = vo.get_current_state(scale)
            for point2d in feature2d:
                cv2.circle(img_bgr,(int(point2d[0]),int(point2d[1])),3,(0,0,255),-1)

            for point2d in scale_estimator.initial_points:
                cv2.circle(img_bgr,(int(point2d[0]),int(point2d[1])),3,(255,0,0),-1)
            for point2d in scale_estimator.inliers:
                cv2.circle(img_bgr,(int(point2d[0]),int(point2d[1])),3,(0,255,0),-1)

        
        
            cv2.circle(bg_img, (int(cur_t[0])+300,-int(cur_t[2])+300), 1, (image_id*255/4540,255-image_id*255/4540,0), 1)
            cv2.imshow('traj',bg_img)
        cv2.imshow('image',img_bgr)
        cv2.waitKey(0)
        image_id+=1
        


if __name__ == '__main__':
    main()
