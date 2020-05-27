# recover ego-motion scale by road geometry information
# @author xiangwei(wangxiangwei.cpp@gmail.com) and zhanghui()
# 

# @function: calcualte the road pitch angle from motion matrix
# @input: the tansformation matrix in SE(3) 
# @output:translation angle calculate by t in R^3 and 
# rotarion angle calculate from rotation matrix R in SO(3)



from scipy.spatial import Delaunay
from estimate_road_norm import *
import numpy as np
import math
from collections import deque 
import matplotlib.pyplot as plt



class ScaleEstimator:
    def __init__(self,absolute_reference,window_size=6,vanish=185,focus=718):
        self.absolute_reference = absolute_reference
        self.camera_pitch       = -0.5*np.pi/180
        self.scale  = None
        self.inliers = None
        self.scale_queue = deque()
        self.motion_queue= deque()
        self.window_size = window_size
        self.vanish =  vanish
        self.focus  = focus
        self.b_matrix = np.ones((3,1),np.float)
        self.all_features=[]
        self.correct_distance_features=[]
        self.flat_features=[]
        self.all_feature=[]
        self.correct_distance_feature=[]
        self.flat_feature=[]
        self.flat_feature_2d=[]
        self.img =None
    def initial_estimation(self,motion_t):
        pitch = np.arcsin(motion_t[1])*180/np.pi
        print('initial pitch',pitch)
        self.motion_queue.append(motion_t.reshape(-1))   
        #self.pitch = pitch
        return pitch
    
    # only work when vehicle move forward \frac{fX}{z*(z+1)}>1
    def check_distance(self,feature3d):
        flag_x = (feature3d[:,2]*(feature3d[:,2]+1) - np.abs(self.focus*feature3d[:,0]))<0
        flag_y = (feature3d[:,2]*(feature3d[:,2]+1) - np.abs(self.focus*feature3d[:,1]))<0
        flag = flag_x | flag_y
        print(np.sum(flag),flag.shape)
        return flag
    
    def triangle2region_graph(self,triangle_ids):
        graph  = []
        graph_meta = []
        max_id = np.max(triangle_ids)
        for i in range(0,max_id+1):
            graph_meta.append(set())
        for i in range(0,triangle_ids.shape[0]):
            graph.append([])
        for i in range(0,triangle_ids.shape[0]):
            a,b,c = triangle_ids[i]
            intersect = graph_meta[a].intersection(graph_meta[b])
            if len(intersect)!=0:
                graph[i].append(list(intersect)[0])
                graph[list(intersect)[0]].append(i)
            intersect = graph_meta[a].intersection(graph_meta[c])
            if len(intersect)!=0:
                graph[i].append(list(intersect)[0])
                graph[list(intersect)[0]].append(i)
            intersect = graph_meta[b].intersection(graph_meta[c])
            if len(intersect)!=0:
                graph[i].append(list(intersect)[0])
                graph[list(intersect)[0]].append(i)
            graph_meta[a].add(i)
            graph_meta[b].add(i)
            graph_meta[c].add(i)
        return graph


    # @input  shape n*3 n triangles
    # @output array of set 
    def triangle2graph(self,triangle_ids):
        graph=[]   
        max_id = np.max(triangle_ids)
        for i in range(0,max_id+1):
            graph.append([])
        for triangle_id in triangle_ids:
            triangle_id = np.sort(triangle_id)
            if triangle_id[1]  not in graph[triangle_id[0]]:
                graph[triangle_id[0]].append(triangle_id[1])
            if triangle_id[2]  not in graph[triangle_id[0]]:
                graph[triangle_id[0]].append(triangle_id[2])
            if triangle_id[2]  not in graph[triangle_id[1]]:
                graph[triangle_id[1]].append(triangle_id[2])
        return graph
    '''
    check the three vertice in triangle whether satisfy
    (d_1-d_2)*(v_1-v_2)<=0 if not they are outlier
    True means inlier
    '''
    def check_triangle(self,v,d):
        flag=[False,False,False]
        a = (v[0]-v[1])*(d[0]-d[1])
        b = (v[0]-v[2])*(d[0]-d[2])
        c = (v[1]-v[2])*(d[1]-d[2])
        if a>0:
            flag[0]=True
            flag[1]=True
        if b>0:
            flag[0]=True
            flag[1]=True
        if c>0:
            flag[1]=True
            flag[2]=True
        return np.array(flag)

    def check_depth(self,v,d):
        if (v[0]-v[1])*(d[0]-d[1])>0:
            return True
        else:
            return False
        
    def find_reliability_by_graph(self,feature3d,feature2d,triangle_ids):
        feature_graph = self.triangle2graph(triangle_ids)
        reliability   = 0.8*np.ones((feature3d.shape[0]))
        for i in range(0,len(feature_graph)):
            for j in feature_graph[i]:
                a = reliability[i]*reliability[j]
                b = (1-reliability[i])*reliability[j]
                c = (1-reliability[j])*reliability[i]
                d = (1-reliability[i])*(1-reliability[j])
                # abmormal
                if self.check_depth(feature2d[[i,j],1],feature3d[[i,j],2]):
                    reliability[i] = (0.25*c)/(0.25*(b+c)+0.5*d)
                    reliability[j] = (0.25*b)/(0.25*(b+c)+0.5*d)
                else:
                    #normal
                    reliability[i] = (a+0.25*c)/(a+0.25*(b+c)+0.5*d)
                    reliability[j] = (a+0.25*b)/(a+0.25*(b+c)+0.5*d)

        valid_id = reliability>0.8
        print('reliability',np.min(reliability),np.median(reliability),np.max(reliability))
        print('feature rejected ',np.sum(valid_id==False))
        print('feature left     ',np.sum(valid_id))
        return valid_id

    def find_outliers(self,feature3d,feature2d,triangle_ids):
        # suppose every is inlier
        outliers = np.ones((feature3d.shape[0]))

        for triangle_id in triangle_ids:
            data=[]
            depths   = feature3d[triangle_id,2]
            pixel_vs = feature2d[triangle_id,1]
            flag    = self.check_triangle(pixel_vs,depths)
            outlier = triangle_id[flag] 
            inlier  = triangle_id[~flag] 
            outliers[outlier]-=np.ones(outliers[outlier].shape[0])
            outliers[inlier]+=np.ones(outliers[inlier].shape[0])
        print('feature rejected ',np.sum(outliers<0))
        print('feature left     ',np.sum(outliers>=0))
        valid_id = (outliers>=0)
        return valid_id
    
    def compare(self,a,b,threshold=0.1):
        if a - b< -threshold:
            return -1
        elif a-b> threshold:
            return 1
        else:
            return 0
    
    def feature_selection_by_tri_graph(self,feature3d,triangle_ids):
        graph = self.triangle2region_graph(triangle_ids)
        #calculating the geometry model of each triangle
        triangles = np.array([np.matrix(feature3d[triangle_id]) for triangle_id in triangle_ids])
        triangles_i = np.array([np.matrix(feature3d[triangle_id]).I for triangle_id in triangle_ids])
        normals     = (triangles_i@self.b_matrix).reshape(-1,3)
        normals_len = np.sqrt(np.sum(normals*normals,1)).reshape(-1,1)
        normals     = normals/normals_len
        pitch_deg   = np.arcsin(-normals[:,1])*180/np.pi #[-90,90]
        heights     = np.mean(triangles[:,:,1],1)

        p_road           = (-70 - pitch_deg)/20-0.2
        p_road[p_road<0] = 0
        valid_pitch_id = pitch_deg<-80
        unvalid_pitch_id = pitch_deg>=-80
        valid_triangle_ids = bool2id(valid_pitch_id)
        observation_matrix = np.array([[0.33,0.33,0.33],[0.03,0.07,0.90],[0.90,0.07,0.03],[0.05,0.9,0.05]])
        for valid_triangle_id in valid_triangle_ids:
            mb_ids = np.array(graph[valid_triangle_id])

            #plt.plot(heights[mb_ids],'.-g')
            #plt.plot(p_road[mb_ids],'.-r')
            #plt.plot(heights[valid_triangle_id],'g*')
            #plt.plot(p_road[valid_triangle_id],'r*')
            ha = heights[valid_triangle_id]
            pa = p_road[valid_triangle_id]
            prab=[]
            #print('initial prab',pa)
            for mb_id in  mb_ids:
                hc = heights[mb_id]
                pc = p_road[mb_id]
                cr = self.compare(hc,ha)
                potential_matrix = np.array([(1-pa)*(1-pc),(1-pa)*pc,pa*(1-pc),pa*pc])
                pa = observation_matrix[2:4,cr+1]@potential_matrix[2:4] /(observation_matrix[:,cr+1]@potential_matrix)
                prab.append(pa)
                #print(pa)
            p_road[valid_triangle_id] = pa
            #plt.ylim()
        print('triangle left ',np.sum(valid_pitch_id),'from',valid_pitch_id.shape[0])
        height_level = (np.mean(heights[unvalid_pitch_id]))
        self.height_level = height_level
        print('height level',height_level)
        valid_id = p_road>0.5       #valid_id = valid_pitch_id
        print('triangle left final',np.sum(valid_id),'from',valid_id.shape[0])
        valid_points_id = np.unique(triangle_ids[valid_id].reshape(-1))
        return valid_points_id


    def feature_selection_by_tri(self,feature3d,triangle_ids):

        #calculating the geometry model of each triangle
        triangles = np.array([np.matrix(feature3d[triangle_id]) for triangle_id in triangle_ids])
        triangles_i = np.array([np.matrix(feature3d[triangle_id]).I for triangle_id in triangle_ids])
        normals     = (triangles_i@self.b_matrix).reshape(-1,3)
        normals_len = np.sqrt(np.sum(normals*normals,1)).reshape(-1,1)
        normals     = normals/normals_len
        pitch_deg   = np.arcsin(-normals[:,1])*180/np.pi #[-90,90]
        
        valid_pitch_id = pitch_deg<-80
        
        print('triangle left ',np.sum(valid_pitch_id),'from',valid_pitch_id.shape[0])
        heights     = np.mean(triangles[:,:,1],1)
        unvalid_pitch_id = pitch_deg>=-80
        height_level = (np.mean(heights[unvalid_pitch_id]))
        self.height_level = height_level
        print('height level',height_level)
        valid_height_id = heights>height_level
        valid_id = valid_pitch_id & valid_height_id
        print('triangle left final',np.sum(valid_id),'from',valid_id.shape[0])
        #valid_id = valid_pitch_id
        valid_points_id = np.unique(triangle_ids[valid_id].reshape(-1))
        return valid_points_id

    def feature_selection(self,feature3d,feature2d):
         # 1 select feature below vanish point
        lower_feature_ids = feature2d[:,1]>self.vanish
        feature2d = feature2d[lower_feature_ids,:]
        feature3d = feature3d[lower_feature_ids,:]
        #self.distribution(feature3d)
        # 2. select feature with wrong depth estimation
        tri = Delaunay(feature2d)
        triangle_ids = tri.simplices
        #valid_id = self.find_reliability_by_graph(feature3d,feature2d,triangle_ids)
        valid_id = self.find_outliers(feature3d,feature2d,triangle_ids)
        

        if(valid_id.shape[0]>3):
            feature2d = feature2d[valid_id,:]
            feature3d = feature3d[valid_id,:]
            tri = Delaunay(feature2d)
            triangle_ids = tri.simplices
        else:
            print('no enough feature for triangulation')
            return None

        # 3. select feature with similar model with road
        selected_id = self.feature_selection_by_tri(feature3d,triangle_ids)
        if(len(selected_id)>0):
            self.flat_feature_2d = feature2d[selected_id]
            return  feature3d[selected_id]
        else:
            print('no enough flat feature')
            return None
    
    def road_model_calculation(self,feature3d):
        return self.road_model_calculation_static(feature3d)
    
    def remove_single(self,feature3d,dis,bins):
        flag_single = (dis==1)
        bins_single = bins[1:][flag_single]
        if np.sum(bins_single)>0:
            bin_single = bins_single[0]
            unvalid_flag = (feature3d[:,1]>=bin_single-0.1)&(feature3d[:,1]<=bin_single)
            for bin_single in bins_single[1:]:
                unvalid_flag |= (feature3d[:,1]>bin_single-0.1)&(feature3d[:,1]<=bin_single)
            feature3d=feature3d[unvalid_flag==False,:]
        return feature3d
    def road_model_calculation_static_tri(self,heights):
        print('flat features',len(heights))
        heights_inv = 1/heights
        dis,bins =np.histogram(heights_inv,bins = np.array(range(0,20))*0.1)
        #heights_inv = self.remove_single(heights_inv,dis,bins)
        dis[dis==1]=0
        modes = self.check_mode(dis,bins)
        print('mode number',len(modes))
        if len(modes)==0:
            return np.median(heights_inv),0,1
        else:
            reverse_modes = self.check_reverse_mode(dis)
            mode_right = int(modes[0][-1]*10)
            mode_left  = int(modes[0][0]*10)
            mode = (mode_left+mode_right)/2
            skewness = 0
            return mode/10,0,1;
            if len(modes)>1:
                skewness = self.check_skewness(heights_inv,mode=mode/10)
                print('skewness',skewness,'\n')
            else:
                skewness = self.check_skewness(heights_inv,mode=mode/10,method='p2')
                print('skewness',skewness,'\n')
            if skewness>-0.3:
                return mode/10,0,1
            else:
                reverse_modes_left = reverse_modes[:mode_left]
                left = bins[1:mode_left+1][reverse_modes_left][-1]
                return left,0,1
     
    def road_model_calculation_static(self,feature3d):
        print('flat features',feature3d.shape)
        dis,bins =np.histogram(feature3d[:,1],bins = np.array(range(0,170))*0.1)
        feature3d = self.remove_single(feature3d,dis,bins)
        dis[dis==1]=0
        modes = self.check_mode(dis,bins)
        print('mode number',len(modes))
        if len(modes)==0:
            if feature3d.shape[0]>0:
                return np.median(feature3d[:,1]),0,1
            else:
                return self.height_level,0,1
        else:
            reverse_modes = self.check_reverse_mode(dis)
            mode_right = int(modes[-1][-1]*10)
            mode_left  = int(modes[-1][0]*10)
            mode = (mode_left+mode_right)/2
            reverse_modes_left = reverse_modes[:mode_left]
            reverse_modes_right = reverse_modes[mode_right:]
            left = bins[1:mode_left+1][reverse_modes_left][-1]
            right = bins[mode_right+1:][reverse_modes_right][0]

            skewness = self.check_skewness(feature3d[:,1],mode=mode/10)
            print('skewness',skewness)
            if skewness>0.3:
            # unimodel or multi-model
                    left = right
                    right = 15.1
                    return left,0,1
            else:
                return mode/10,0,1
            '''
            print('mode',mode_left,mode_right,'left',left,'right',right)
            valid_id = (feature3d[:,1]>(left)) & (feature3d[:,1]<right)
            print('valid num',np.sum(valid_id))
            if np.sum(valid_id) >0:
                return  np.median(feature3d[valid_id,1]),0,np.std(feature3d[valid_id,1]+1/np.sum(valid_id))
            else:
                return left,0,1
            '''
            

    def road_model_calculation_ransac(self,feature3d):
            # ransac
        a_array = np.array(feature3d)
        m,b = get_pitch_ransac(a_array,30,0.005)
        inlier_id = get_inliers(m,feature3d[:,:],0.01)
        inliers = feature3d[inlier_id,:]
        road_model_ransac = np.array(m)
        normal = road_model_ransac[0:-1]
        h_bar = -road_model_ransac[-1]
        if normal[1]<0:
            normal = -normal
            h_bar = -h_bar
        

        normal_len = np.sqrt(np.sum(normal*normal)) 
        ransac_camera_height = h_bar/normal_len
        #ransac_camera_height = np.median(feature3d[:,1])
        pitch =  np.arcsin(-normal[1]/normal_len)
        return ransac_camera_height,pitch,inliers

    def distribution(self,feature3d):
        dis,bins =np.histogram(feature3d[:,1],bins = np.array(range(0,100))*0.1)
        plt.plot(dis)

    def feature_remap(self,feature3d):    
        y=feature3d[:,1]*np.cos(self.camera_pitch)-feature3d[:,2]*np.sin(self.camera_pitch)
        z=feature3d[:,1]*np.sin(self.camera_pitch)+feature3d[:,2]*np.cos(self.camera_pitch)
        feature3d[:,1]=y
        feature3d[:,2]=z
    
    def scale_filtering(self,scale):
        self.scale_queue.append(scale)
        if len(self.scale_queue)>self.window_size:
            self.scale_queue.popleft()
        return np.median(self.scale_queue)
    def scale_calculation_static(self,point_selected):
        self.feature_remap(point_selected)
        if(point_selected is not None):
            height,pitch,std= self.road_model_calculation(point_selected)
            scale = self.absolute_reference/height
        else:
            scale = self.absolute_reference/self.height_level
            print('no enough feature on road')
        return self.scale_filtering(scale),std

    def scale_calculation(self,feature3d,feature2d,img=None):
        scale = 0       
        std   = 100
        self.feature_remap(feature3d)
        point_selected = self.feature_selection(feature3d,feature2d)
        self.flat_feature = point_selected
        if(point_selected is not None):
            height,pitch,std= self.road_model_calculation(point_selected)
            scale = self.absolute_reference/height
        else:
            scale = self.absolute_reference/self.height_level
            print('no enough feature on road')
        return self.scale_filtering(scale),std


        #return scale,std

    def check_reverse_mode(self,dis_data):
        dis_data = np.array(dis_data)
        flag   = np.zeros(dis_data.shape[0])
        min_data =  np.min(dis_data)

        if dis_data[0] == min_data:
            flag[0]=1

        if dis_data[-1] == min_data:
            flag[-1]=1
        flag_l = dis_data[1:-1] <=dis_data[0:-2]
        flag_r = dis_data[1:-1] <= dis_data[2:]
        flag_n = (dis_data[1:-1] == dis_data[2:]) & (dis_data[1:-1] == dis_data[0:-2])

        flag[1:-1]   = flag_l & flag_r &(~flag_n)
        return flag>0.5


    def check_mode(self,dis_data,bins):
        dis_data = np.array(dis_data)
        flag   = np.zeros(dis_data.shape[0])
        max_data =  np.max(dis_data)

        if max_data<=2:
            return []

        if dis_data[0] == max_data:
            flag[0]=1

        if dis_data[-1] == max_data:
            flag[-1]=1
        flag_l = dis_data[1:-1] >= dis_data[0:-2]
        flag_r = dis_data[1:-1] >= dis_data[2:]
        flag_v = dis_data[1:-1] >= 0.33* max_data
        flag_n = dis_data[1:-1] >=2
        flag[1:-1]   = flag_l & flag_r & flag_v & flag_n

        modes_flag =  flag>0.5
        modes = bins[1:][modes_flag]
        print(modes)
        modes_final = []
        mode_final=[]
        if np.sum(modes_flag)==1:
            modes_final.append([modes])
        elif np.sum(modes_flag)>1:
            for mode in modes:
                if len(mode_final)==0 or mode - mode_final[-1]<0.11:
                    mode_final.append(mode)
                else:
                    modes_final.append(mode_final[:])
                    mode_final.clear()
                    mode_final.append(mode)
            if len(mode_final)>0:
                modes_final.append(mode_final)

        return modes_final


    def check_skewness(self,data,mode=None,method='p1'):   
        print('mean,meidan,std',np.mean(data),np.median(data),np.std(data))
        if method=='p2':
            skew = 3*(np.mean(data) - np.median(data))/np.std(data)
        elif method =='p1':
            if mode is  None:
                dis,bins =np.histogram(data,bins = np.array(range(0,170))*0.1)
                dis[dis==1]=0
                flag = (dis == np.max(dis))
                mode = np.mean(bins[1:][flag])
            skew = (np.mean(data) - mode)/np.std(data)
        return skew
    def skewness_analysis(self):
        #all_skew = self.check_skewness(self.all_feature[:,1])
        #cor_skew = self.check_skewness(self.correct_distance_feature[:,1])
        flat_skew= self.check_skewness(self.flat_feature[:,1])
        print('skewness',flat_skew)
        return flat_skew

    def mode_analysis(self):
        dis,bins =np.histogram(self.flat_feature[:,1],bins = np.array(range(0,100))*0.1,density=True)
        return self.check_mode(dis,bins)

    def plot_distribution(self,label,img,scale=1):

        '''
        ax = plt.subplot(111)
        all_features = np.array(self.all_features)
        dis,bins =np.histogram(all_features[:,1],bins = np.array(range(0,30))*0.1,density=True)
        ax.plot(bins[:-1],0.1*dis,'r',label='All Features')
        correct_features = np.array(self.correct_distance_features)
        dis,bins =np.histogram(correct_features[:,1],bins = np.array(range(0,30))*0.1,density=True)
        ax.plot(bins[:-1],0.1*dis,'g',label='Depth-correct Features')
        flat_features = np.array(self.flat_features)
        dis,bins =np.histogram(flat_features[:,1],bins = np.array(range(0,30))*0.1,density=True)
        #ax.plot(bins[:-1],dis,'y')
        plt.title('Feature Points Vertical Distribution')
        plt.legend()
        '''
        ax2 = plt.subplot(221)
        if self.all_feature is not None and len(self.all_feature)>0:
            all_feature = np.array(self.all_feature)
            dis,bins =np.histogram(all_feature[:,1],bins = np.array(range(0,50))*0.1)
            #ax2.plot(bins[:-1],0.1*dis,'r-*',label='All Features')
        if len(self.correct_distance_feature)>0:
            correct_feature = np.array(self.correct_distance_feature)
            dis,bins =np.histogram(correct_feature[:,1],bins = np.array(range(0,50))*0.1)
            #ax2.plot(bins[:-1],0.1*dis,'g-*',label='Depth-correct Features')
        if self.flat_feature is not None and len(self.flat_feature)>0:
            flat_feature = np.array(self.flat_feature)
            dis,bins =np.histogram(flat_feature[:,1],bins = np.array(range(0,50))*0.1,density=True)
            ax2.plot(bins[:-1],0.1*dis,'y-*',label='Selected Features')
        ax2.set_title('vertical distribution')
        ax2.set_xlabel('y')
        ax2.set_ylabel('')

        #ax2.legend(loc='upper left')


        ax3 = plt.subplot(222)
        ax3.plot(self.all_feature[:,2],-self.all_feature[:,1],'.r',label='All Features')
        ax3.plot(self.correct_distance_feature[:,2],-self.correct_distance_feature[:,1],'.g',label='Depth-correct Features')
        ax3.set_xlabel('z')
        ax3.set_ylabel('-y')
        ax3.set_title('feature projection')
        if self.flat_feature is not None and len(self.flat_feature)>0:
            ax3.plot(self.flat_feature[:,2],-self.flat_feature[:,1],'.y',label='Selected Features')
        #ax3.legend()
        ax4 = plt.subplot(212)
        #draw_feature(img,self.flat_feature_2d,)
        #draw_feature(img,self.flat_feature_2d[self.flat_feature[:,2]>60,:],(255,0,0))
        ax4.imshow(self.img)
        #plt.savefig('case'+label+'.pdf')

        plt.show()

    def check_full_distribution(self,feature3d,feature2d,scale,img):
        #self.feature_remap(feature3d)
        lower_feature_ids = feature2d[:,1]>self.vanish
        feature3d = feature3d[lower_feature_ids,:]
        feature2d = feature2d[lower_feature_ids,:]
        self.all_feature = feature3d.copy()
        for feature in feature3d:
            self.all_features.append(feature*scale)
        
        draw_feature(img,feature2d,(255,0,0))
        tri = Delaunay(feature2d)
        triangle_ids = tri.simplices
        #valid_id = self.find_outliers(feature3d,feature2d,triangle_ids)
        valid_id = self.find_reliability_by_graph(feature3d,feature2d,triangle_ids)
        if(valid_id.shape[0]>3):
            feature2d = feature2d[valid_id,:]
            feature3d = feature3d[valid_id,:]
            draw_feature(img,feature2d,(0,255,0))
            self.correct_distance_feature =feature3d.copy()
            for feature in feature3d:
                self.correct_distance_features.append(feature*scale)
            tri = Delaunay(feature2d)
            triangle_ids = tri.simplices
        else:
            return 

        selected_id = self.feature_selection_by_tri(feature3d,triangle_ids)
        #selected_id = self.feature_selection_by_tri_graph(feature3d,triangle_ids)
        if(len(selected_id)>0):
            point_selected =   feature3d[selected_id]
            point_selected_2d =   feature2d[selected_id]
            self.flat_feature = point_selected.copy()
            for feature in point_selected:
                self.flat_features.append(feature*scale)


            draw_feature(img,point_selected_2d,(255,255,0))
        self.img = img


def bool2id(flag):
    id_array = np.array(range(0,flag.shape[0]))
    ids      = id_array[flag]
    return ids

def draw_feature(img,feature,color=(255,255,0)):
    for i in range(feature.shape[0]):
        cv2.circle(img,(int(feature[i,0]),int(feature[i,1])),3,color,-1)


def main():
    # get initial motion pitch by motion matrix
    # triangle region norm and height
    # selected
    # calculate the road norm
    # filtering
    # scale recovery
    camera_height = 1.7
    scale_estimator = ScaleEstimator(camera_height)
    scale = scale_estimator.scale_calculation()

if __name__ == '__main__':
    main()


'''
#heights     = (1/normals_len).reshape(-1)
#heights     = np.mean(triangles[:,:,1],1)
#if np.sum(valid_pitch_id)==0:
#    return []
#precomput_h  = np.median(heights[valid_pitch_id])
#height_sorted= np.sort(heights[valid_pitch_id])
#feature_height_sorted = np.sort(feature3d[:,1])
#print(heights)
#precomput_h_lower = height_sorted[3*height_sorted.shape[0]//4]
#precomput_h_lower_feature = feature_height_sorted[3*feature_height_sorted.shape[0]//4]
#valid_height_id= heights> max(precomput_h_lower,precomput_h_lower_feature)


#valid_id = valid_pitch_id & valid_height_id
#print(triangle_ids.shape,valid_id.shape)       

#near_feature_ids = self.check_distance(feature3d)
#feature2d = feature2d[near_feature_ids,:]
#feature3d = feature3d[near_feature_ids,:]
#draw_feature(img,feature2d,(0,255,0))       
#plt.plot(feature3d[:,2],-feature3d[:,1],'.r')
'''
