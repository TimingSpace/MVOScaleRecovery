import matplotlib.pyplot as plt
import numpy as np
import sys
def plot_path():
    label=['ground truth','path 1','path 2']
    name = str(sys.argv[1]).split('.')[-2].split('/')[-1]
    print(name)
    for i in range(1, len(sys.argv)):
        path_vo = np.loadtxt(sys.argv[i]) 
        #plt.plot(path_vo[:,3],path_vo[:,7])
        plt.plot(path_vo[:,3],path_vo[:,11],label=label[i-1])
    min_x = np.min(path_vo[:,3:12:4],0)
    max_x = np.max(path_vo[:,3:12:4],0)
    mean_x= (min_x+max_x)/2
    diff_x = max_x -min_x
    max_diff = np.max(diff_x)
    print(min_x,max_x)
    plt.xlabel('x/m')
    plt.xlim(mean_x[0]-max_diff/2-0.1*max_diff,mean_x[0]+max_diff/2+0.1*max_diff)
    plt.ylim(mean_x[2]-max_diff/2-0.1*max_diff,mean_x[2]+max_diff/2+0.1*max_diff)
    plt.ylabel('y/m')
    plt.title('PATH')
    plt.legend()
    plt.savefig('result/'+name+'.pdf')
    plt.show()

def plot_scale_filter(window=20):
    for i in range(1, min(len(sys.argv),3)):
        data = np.loadtxt(sys.argv[i])
        data_new = []
        for i in range(0,data.shape[0]-window):
            data_new.append(np.mean(data[i:i+window]))
        plt.plot(data_new)
    plt.xlabel('frame')
    plt.ylabel('scale')
    plt.title('SCALE')
    plt.show()


def plot_scale():
    label=['ground truth','path 1','path 2']
    for i in range(1, min(len(sys.argv),4)):
        data = np.loadtxt(sys.argv[i])
        data_sum = [data[0]]
        for j in range(1,data.shape[0]):
            data_sum.append(data_sum[-1]+data[j])
        
        plt.plot(data,label=label[i-1])
    plt.xlabel('frame')
    plt.ylabel('scale')
    plt.legend()
    plt.title('SCALE')
    plt.show()

def plot_motion():
    import transformation as tf
    color=['r','g','b','y']
    label=['ground truth','path 1','path 2']
    for i in range(1, len(sys.argv)):
        path_vo = np.loadtxt(sys.argv[i]) 
        motion_vo = tf.pose2motion(path_vo)
        se_vo     = tf.SEs2ses(path_vo)
        #plt.plot(path_vo[:,3],path_vo[:,7])
        #plt.plot(se_vo[:,3])
        #plt.plot(path_vo[:,7]/1000)
        #plt.plot(motion_vo[:,7])
        #plt.plot(motion_vo[:,11]/10)
        plt.plot(path_vo[:,11],color[i-1],label=label[i-1])
    plt.xlabel('frame')
    plt.ylabel('z/m')
    plt.title('PATH')
    plt.legend()
    plt.show()
def plot_corr():
   gt = np.loadtxt(sys.argv[1])
   re = np.loadtxt(sys.argv[2])
   re = re[1:]
   re_l = re.shape[0]
   er_ = np.abs((gt[:re_l]-re))
   valid_1 = er_ >0.0
   er = np.loadtxt(sys.argv[3])
   er = er
   er = er[1:]
   valid = (er <100) & valid_1
   corr = np.corrcoef(er,er_)
   print(corr)
   plt.plot(er[valid],'g',label='variance')
   plt.plot(er_[valid],'r',label='error')
   plt.legend()
   plt.show()

def main():
    if len(np.loadtxt(sys.argv[1]).shape)>1:
        plot_motion()
        plot_path()
    else:
        plot_scale()

if __name__ == '__main__':
    main()
