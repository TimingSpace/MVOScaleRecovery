import matplotlib.pyplot as plt
import numpy as np
import sys
def evaluate_scale(gt,re,vis=True):
    re_l = re.shape[0]
    er_ = np.abs((gt[:re_l]-re))
    print(np.mean(er_),np.max(er_),1-np.sum(er_>0.1)/re_l,1-np.sum(er_>0.2)/re_l,1-np.sum(er_>0.3)/re_l,1-np.sum(er_>0.5)/re_l)

def evaluate(vis=True):
    gt = np.loadtxt(sys.argv[1]) # ground truth
    re = np.loadtxt(sys.argv[2]) # result
    evaluate_scale(gt,re)
    re_f = filter(re,5)
    print(re_f)
    evaluate_scale(gt,re_f)
    if vis:
        plt.plot(re_f)
        plt.show()
def filter(data,window=10):
    data_new = [data[0]]
    for i in range(1,data.shape[0]):
        data_new.append(np.median(data[max(i-window+1,0):i+1]))
    return np.array(data_new)
def main():
    evaluate()

if __name__ == '__main__':
    main()
