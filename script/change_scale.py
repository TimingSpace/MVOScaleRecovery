import numpy as np
import sys
from transformation import *


def main():
    file_name = sys.argv[2]
    tag = file_name.split('.')[-1]
    data = np.loadtxt(sys.argv[1])
    scale= np.loadtxt(sys.argv[2])
    motion = pose2motion(data)
    motion_trans = motion[:,3:12:4]
    square = np.sum(motion_trans*motion_trans,1)
    speed = np.sqrt(square)
    motion_trans = (scale*motion_trans.transpose()/(speed+0.00001)).transpose()
    motion[:,3:12:4] = motion_trans
    pose = motion2pose(motion)
    np.savetxt('new_pose.txt'+tag,pose)

if __name__ =='__main__':
    main()
