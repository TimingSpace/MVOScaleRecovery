#coding:utf-8
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('test2.png',0)
delete_freq=5 　　　　　　　　
i=0
akaze = cv2.AKAZE_create(threshold=0.0007)        　　 #smaller, more points
kp_akaze = akaze.detect(img,None)　　　　　　　　　　　　 ＃keypoints of akaze
img_akaze = cv2.drawKeypoints(img,kp_akaze,img,color=(255,0,0))
cv2.imshow('AKAZE',img_akaze)
cv2.waitKey(０)

pts=cv2.KeyPoint_convert(kp_akaze)　　　　　　　　　　　 #positions of keypoints
while i < len(kp_akaze)-6:
    cv2.circle(img,(pts[i][0],pts[i][1]),2, (255, 0, 0), thickness =１) #draw pts[i][0] circle in the image
    i=i+5
cv2.imshow('AKAZE_2',img)
cv2.waitKey(0)
