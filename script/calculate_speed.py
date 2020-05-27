import numpy as np
import sys
from transformation import *

name = sys.argv[1]
name = name.split('.')[0]
data = np.loadtxt(sys.argv[1])
motion = pose2motion(data)
motion_trans = motion[:,3:12:4]
square = np.sum(motion_trans*motion_trans,1)
speed = np.sqrt(square)
np.savetxt(name+'_speed.txt',speed)
