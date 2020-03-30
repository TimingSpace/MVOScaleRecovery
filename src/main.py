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

def main():
    images_path = sys.argv[1]
    images      = open(images_path)
    cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
    #cam = PinholeCamera(640.0, 480.0, 343.8560, 344.8560, 321.1928, 231.2157)
    vo = VisualOdometry(cam)
    scale_estimator = ScaleEstimator(absolute_reference = 1.75)
    image_name = images.readline() # first line is not pointing to a image, so we read and skip it
    image_id = 0
    path=[]
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
            #np.savetxt('feature_3d.txt',vo.feature3d)
            # uncomment to visualize the feature and triangle
            scale_estimator.visualize(vo.feature3d,feature2d,img_bgr)
            if vo.feature3d.shape[0]>500:
                scale = scale_estimator.scale_calculation(vo.feature3d,feature2d)
                R,t = vo.get_current_state(scale)
                M   = np.zeros((3,4))
                M[:,0:3]=R
                M[:,3]=t.reshape(-1)
                M = M.reshape(-1)
                path.append(M)
            else:
                path.append(path[-1])
            print('scale ',scale)
            # uncomment to visualize the path
            #vo.visualize(bg_img)
        cv2.imshow('image',img_bgr)
        #cv2.imshow('src',img)
        #cv2.imshow('path',bg_img)
        key = cv2.waitKey(1)
        #if(key&255)==ord('q'):
        #    break
        image_id+=1
        
    np.savetxt('path.txt',path)


if __name__ == '__main__':
    main()