import matplotlib.pyplot as plt
import numpy as np
import sys
def plot_path():
    label=['MVOSR','ground truth']
    for i in range(1, len(sys.argv)):
        path_vo = np.loadtxt(sys.argv[i]) 
        #plt.plot(path_vo[:,3],path_vo[:,7])
        plt.plot(path_vo[:,3],path_vo[:,11],label=label[i-1])
    plt.xlabel('x/m')
    plt.ylabel('y/m')
    plt.title('PATH')
    plt.legend()
    plt.show()

def plot_scale():
    for i in range(1, len(sys.argv)):
        plt.plot(np.loadtxt(sys.argv[i]))
    plt.xlabel('frame')
    plt.ylabel('scale')
    plt.title('SCALE')
    plt.show()

def plot_motion():
    import transformation as tf
    color=['r','g']
    label=['ground truth','MVOSR']
    for i in range(1, len(sys.argv)):
        path_vo = np.loadtxt(sys.argv[i]) 
        motion_vo = tf.pose2motion(path_vo)
        se_vo     = tf.SEs2ses(path_vo)
        #plt.plot(path_vo[:,3],path_vo[:,7])
        #plt.plot(se_vo[:,3])
        #plt.plot(path_vo[:,7]/1000)
        #plt.plot(motion_vo[:,7])
        #plt.plot(motion_vo[:,11]/10)
        plt.plot(path_vo[:,7],color[i-1],label=label[i-1])
    plt.xlabel('frame')
    plt.ylabel('z/m')
    plt.title('PATH')
    plt.legend()
    plt.show()
def plot_corr():
   gt = np.loadtxt(sys.argv[1])
   re = np.loadtxt(sys.argv[2])
   er_ = (gt-re)/(gt)
   er_[gt<0.1]=0
   er_ = (er_-np.median(er_))/np.std(er_)
   er = np.loadtxt(sys.argv[3])
   plt.plot(er)
   plt.show()
   er = (er-np.median(er[er>0]))/(np.std(er[er>0]))
   plt.plot(er,'g')
   plt.plot(er_,'r')
   plt.show()

def main():
    if len(np.loadtxt(sys.argv[1]).shape)>1:
        plot_motion()
        plot_path()
    else:
        #plot_corr()
        plot_scale()

if __name__ == '__main__':
    main()
