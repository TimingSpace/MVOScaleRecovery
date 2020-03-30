import numpy as np 
import cv2
import sys
from matplotlib import pyplot as plt

gt_path    = sys.argv[1]
image_path = sys.argv[2]

from visual_odometry import PinholeCamera, VisualOdometry


cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
vo = VisualOdometry(cam, gt_path)

traj = np.zeros((600,600,3), dtype=np.uint8)

for img_id in range(4541):
    print(img_id)
    img = cv2.imread(sys.argv[2]+str(img_id).zfill(6)+'.png', 0)
    #print(img)

    vo.update(img, img_id)
    img_vis = img.copy()
    if img_id>0:
        print(vo.feature3d.shape)
        feature2d = vo.feature3d[:,0:2]
        feature2d[:,0] = feature2d[:,0]*cam.fx/vo.feature3d[:,2]+cam.cx
        feature2d[:,1] = feature2d[:,1]*cam.fx/vo.feature3d[:,2]+cam.cy
        print(feature2d)
        for point2d in feature2d:
            cv2.circle(img_vis,(int(point2d[0]),int(point2d[1])),3,(0,255,0),-1)


    cur_t = vo.cur_t
    if(img_id > 2):
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
    else:
        x, y, z = 0., 0., 0.
    draw_x, draw_y = int(x)+290, int(z)+90
    true_x, true_y = int(vo.trueX)+290, int(vo.trueZ)+90

    cv2.circle(traj, (draw_x,draw_y), 1, (img_id*255/4540,255-img_id*255/4540,0), 1)
    cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
    cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
    text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
    cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

    cv2.imshow('Road facing camera', img_vis)
    cv2.imshow('Trajectory', traj)
    cv2.waitKey(1)

cv2.imwrite('map.png', traj)
