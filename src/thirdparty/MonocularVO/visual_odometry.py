# modified from https://github.com/uoip/monoVO-python

import numpy as np 
import cv2

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500

lk_params = dict(winSize  = (21, 21), 
                #maxLevel = 3,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def featureTracking(image_ref, image_cur, px_ref):
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2


class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy, 
                k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = int(width)
        self.height = int(height)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
    def __init__(self, cam, annotations=None):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.camera_matrix = np.eye(3)
        self.camera_matrix[0,0] = self.camera_matrix[1,1] = cam.fx
        self.camera_matrix[0,2] = cam.cx
        self.camera_matrix[1,2] = cam.cy

        self.motion_R = None
        self.motion_t = None
        self.feature3d= None
        #print(self.camera_matrix)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        self.annotations = None
        if annotations!=None:
            with open(annotations) as f:
                self.annotations = f.readlines()

    def getAbsoluteScale(self, frame_id):  #specialized for KITTI odometry dataset
        ss = self.annotations[frame_id-1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.annotations[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))

    def processFirstFrame(self):
        self.px_ref = self.detector.detect(self.new_frame)
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.frame_stage = STAGE_SECOND_FRAME

    def processSecondFrame(self):
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)

        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref,cameraMatrix = self.camera_matrix , method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, self.cur_R, self.cur_t, mask,points_3d = cv2.recoverPose(E, self.px_cur,\
             self.px_ref,cameraMatrix=self.camera_matrix,distanceThresh=100)
        mask_bool = np.array(mask>0).reshape(-1)
        points_3d_selected = points_3d[:,mask_bool].T
        #print(points_3d_selected.shape)
        points_3d_selected[:,0] = points_3d_selected[:,0]/points_3d_selected[:,3]
        points_3d_selected[:,1] = points_3d_selected[:,1]/points_3d_selected[:,3]
        points_3d_selected[:,2] = points_3d_selected[:,2]/points_3d_selected[:,3]
        self.motion_R = self.cur_R
        self.motion_t = self.cur_t
        self.feature3d= points_3d_selected[:,0:3]
        #print(points_3d_selected)

        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        self.frame_stage = STAGE_DEFAULT_FRAME 
        self.px_ref = self.px_cur

    def processFrame(self, frame_id):
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        #E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        E, mask_e = cv2.findEssentialMat(self.px_cur, self.px_ref,cameraMatrix = self.camera_matrix , method=cv2.RANSAC,
        prob=0.999, threshold=0.5)
        #_, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
        _, R, t, mask,points_3d = cv2.recoverPose(E, self.px_cur,\
             self.px_ref,cameraMatrix=self.camera_matrix,distanceThresh=100)
        mask_bool = np.array(mask>0).reshape(-1)
        mask_e_bool = np.array(mask_e>0).reshape(-1)
        mask_bool = mask_bool & mask_e_bool
        points_3d_selected = points_3d[:,mask_bool].T
        #print(points_3d_selected.shape)
        points_3d_selected[:,0] = points_3d_selected[:,0]/points_3d_selected[:,3]
        points_3d_selected[:,1] = points_3d_selected[:,1]/points_3d_selected[:,3]
        points_3d_selected[:,2] = points_3d_selected[:,2]/points_3d_selected[:,3]
        self.motion_R = R
        self.motion_t = t
        self.feature3d= points_3d_selected[:,0:3]
        if(self.px_ref.shape[0] < kMinNumFeature):
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
        self.px_ref = self.px_cur
    def get_current_state(self,scale):
        self.cur_t = self.cur_t + scale*self.cur_R.dot(self.motion_t) 
        self.cur_R = self.motion_R.dot(self.cur_R)
        return self.cur_R,self.cur_t
    def visualize(self,img):
        point=(int(img.shape[1]/2+self.cur_t[0,0]),int(img.shape[0]/2-self.cur_t[2,0]))
        cv2.circle(img,point,1,(0,255,0),-1)
    def update(self, img, frame_id):
        assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        self.new_frame = img
        if(self.frame_stage == STAGE_DEFAULT_FRAME):
            self.processFrame(frame_id)
        elif(self.frame_stage == STAGE_SECOND_FRAME):
            self.processSecondFrame()
        elif(self.frame_stage == STAGE_FIRST_FRAME):
            self.processFirstFrame()
        self.last_frame = self.new_frame
