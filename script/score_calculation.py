import sys
import numpy as np
def get_rpe(file_name):
    file_data = open(file_name)
    data= file_data.read()
    rpe_800 = data.split(',')[-7]
    return float(rpe_800)

def get_scale(file_name):
    file_data = open(file_name)
    data= file_data.read()
    scale_800 = data.split(',')[-1][:-1]
    return float(rpe_800)
def main():
    if sys.argv[2]=='v':
        rpes = []
        for i in range(0,10):
            file_name = sys.argv[1]+str(i)
            rpe = get_rpe(file_name)
            rpes.append(rpe)
        rpes =np.sort(rpes)
        print(np.mean(rpes[1:-1]))
    else:
        scales=[]
        for i in range(0,10):
            file_name = sys.argv[1]+str(i)
            scale = get_scale(file_name)
            scales.append(scale)
        scales =np.sort(scales)
        print(np.mean(scales[1:-1]))


if __name__ == '__main__':
    main()
