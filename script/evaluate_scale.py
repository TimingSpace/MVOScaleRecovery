import matplotlib.pyplot as plt
import numpy as np
import sys
def evaluate_scale(gt,re,vis=True):
    re_l = re.shape[0]
    er_ = np.abs((gt[:re_l]-re))
    print(np.mean(er_),np.max(er_),1-np.sum(er_>0.1)/re_l,1-np.sum(er_>0.2)/re_l,1-np.sum(er_>0.3)/re_l,1-np.sum(er_>0.5)/re_l)
    er_s = gt[:re_l]-re
    er_all = []
    for window in [10,20,50,100,200,300,400,500,600,700,800]:
        er_  = patch(er_s,window,10)
        er_all.append(np.mean(er_))
    print(er_all)

def evaluate(vis=True):
    gt = np.loadtxt(sys.argv[1]) # ground truth
    re = np.loadtxt(sys.argv[2]) # result
    evaluate_scale(gt,re)
def patch(data,window=10,step=2):
    data_new = []
    for i in range(0,data.shape[0]-window,step):
        data_new.append(np.sum(data[i:i+window]))
    return np.abs(np.array(data_new))/window

def filter(data,window=10):
    data_new = [data[0]]
    for i in range(1,data.shape[0]):
        data_new.append(np.median(data[max(i-window+1,0):i+1]))
    return np.array(data_new)
def main():
    evaluate()

if __name__ == '__main__':
    main()
