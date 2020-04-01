import matplotlib.pyplot as plt
import numpy as np
import sys
def plot_path():
    for i in range(1, len(sys.argv)):
        path_vo = np.loadtxt(sys.argv[i]) 
        plt.plot(path_vo[:,3],path_vo[:,11])
    plt.show()

def plot_scale():
    for i in range(1, len(sys.argv)):
        plt.plot(np.loadtxt(sys.argv[i]))
    plt.show()

def main():
    if len(np.loadtxt(sys.argv[1]).shape)>1:
        plot_path()
    else:
        plot_scale()

if __name__ == '__main__':
    main()
