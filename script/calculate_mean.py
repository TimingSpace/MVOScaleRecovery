import numpy as np
import sys
data = np.loadtxt(sys.argv[1])
print(np.mean(data,0))
