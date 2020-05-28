import cv2
import sys
import numpy as np
lk_params = dict(winSize  = (21, 21), 
                #maxLevel = 3,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

class FeatureDetector:
    def __init__(self,threshold = 0.000007,bucket_size=30,density=2):
        self.akaze = cv2.AKAZE_create(threshold=threshold)
        self.bucket_size = bucket_size
        self.density = density
    def detect(self,image):
        kp_akaze = self.akaze.detect(image,None) 
        px_cur = np.array([x.pt for x in kp_akaze], dtype=np.float32)
        return bucket(px_cur,self.bucket_size,self.density)

    def bucket(self,features,bucket_size=30,density=2):
        u_max,v_max = np.max(features,0)
        u_min,v_min = np.min(features,0)
        print(u_min,v_min,u_max,v_max)
        bucket_x = 1+(u_max )//bucket_size
        bucket_y = 1+(v_max )//bucket_size
        print(bucket_y)
        bucket = []
        for i in range(int(bucket_y)):
            buc = []
            for j in range(int(bucket_x)):
                buc.append([])
            bucket.append(buc)
        print(len(bucket))
        i_feature = 0
        for feature in features:
            u = int(feature[0])//bucket_size
            v = int(feature[1])//bucket_size
            bucket[v][u].append(i_feature)
            i_feature+=1

        #print(bucket)
        new_feature=[]
        for i in range(int(bucket_y)):
            for j in range(int(bucket_x)):
                feature_id = bucket[i][j]
                np.random.shuffle(feature_id)
                for k in range(min(density,len(feature_id))):
                    new_feature.append(features[feature_id[k]])
        
        return np.array(new_feature)


def akaze(image):
    akaze = cv2.AKAZE_create(threshold=0.000007) 
    kp_akaze = akaze.detect(image,None) 
    px_cur = np.array([x.pt for x in kp_akaze], dtype=np.float32)
    return px_cur

def fast(image):
    detector = cv2.FastFeatureDetector_create(10, nonmaxSuppression=True)
    kp_akaze = detector.detect(image,None) 
    #img_akaze = cv2.drawKeypoints(image,kp_akaze,image,color=(255,0,0))
    #cv2.imshow('AKAZE',img_akaze)
    #cv2.waitKey(0)

# ref libviso
def bucket(features,bucket_size=30,density=2):
    u_max,v_max = np.max(features,0)
    u_min,v_min = np.min(features,0)
    print(u_min,v_min,u_max,v_max)
    bucket_x = 1+(u_max )//bucket_size
    bucket_y = 1+(v_max )//bucket_size
    print(bucket_y)
    bucket = []
    for i in range(int(bucket_y)):
        buc = []
        for j in range(int(bucket_x)):
            buc.append([])
        bucket.append(buc)
    print(len(bucket))
    i_feature = 0
    for feature in features:
        u = int(feature[0])//bucket_size
        v = int(feature[1])//bucket_size
        bucket[v][u].append(i_feature)
        i_feature+=1

    #print(bucket)
    new_feature=[]
    for i in range(int(bucket_y)):
        for j in range(int(bucket_x)):
            feature_id = bucket[i][j]
            np.random.shuffle(feature_id)
            for k in range(min(density,len(feature_id))):
                new_feature.append(features[feature_id[k]])
    
    return np.array(new_feature)

def motion_estimarion(feature_ref,feature_target,fx,cx,cy):
    camera_matrix = np.eye(3)
    camera_matrix[0,0] = camera_matrix[1,1] = fx
    camera_matrix[0,2] = cx
    camera_matrix[1,2] = cy


    E, mask = cv2.findEssentialMat(feature_ref, feature_target,cameraMatrix = camera_matrix , method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask,points_3d = cv2.recoverPose(E, feature_ref,\
         feature_target,cameraMatrix=camera_matrix,distanceThresh=100)
    print(t)
    mask_bool = np.array(mask>0).reshape(-1)
    points_3d_selected = points_3d[:,mask_bool].T

def feature_tracking(image_ref, image_cur, px_ref):
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]
    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2



def feature_detection(image,image_show):
    features = akaze(image)
    fast(image)
    features = bucket(features)
    print(features.shape)
    draw_feature(image_show,features)
    return features

def draw_feature(img,feature,color=(255,255,0)):
    for i in range(feature.shape[0]):
        cv2.circle(img,(int(feature[i,0]),int(feature[i,1])),3,color,-1)
def main():
    image_name_file = open(sys.argv[1])
    image_name_file.readline()
    image_names = image_name_file.read().split('\n')
    begin_id = 312
    image_id = 0
    image_last=None
    detector = FeatureDetector()
    for image_name in image_names:
        if image_id< begin_id:
            image_id+=1
            continue
        print(image_name)
        image = cv2.imread(image_name)
        image_show = image.copy()
        #features = feature_detection(image,image_show)
        features = detector.detect(image)
        if(image_last is not None):
            feature_ref,feature_target = feature_tracking(image,image_last,features)
            draw_feature(image_show,feature_ref,(0,255,255))
            draw_feature(image_last,feature_target,(0,255,255))
            motion_estimarion(feature_ref,feature_target,716,607,189)
            print(len(feature_ref))
            cv2.imshow('image_last',image_last)
        cv2.imshow('image',image_show)
        cv2.waitKey(0)
        image_last = image.copy()
        image_id+=1

if __name__ == '__main__':
    main()
