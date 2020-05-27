from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import sys

def vis_3d(points):
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    select_id = np.array(points[:,1]>0) & np.array(points[:,1]<2)
    data_x = points[select_id,0]
    data_y = points[select_id,2]
    data_z = -points[select_id,1]
    ax.scatter(data_x,data_y,data_z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
def depth_check(data):
    select_id = np.array(data[:,1]>0) 
    depth = data[select_id,2]
    v = data[select_id,1]
    v_n = v/depth
    matrix = []
    for i in range(v_n.shape[0]):
        v_minus = v_n-v_n[i]
        d_minus = depth-depth[i]
        matrix.append(d_minus/(v_minus+0.000001))
    matrix = np.array(matrix)
    bins = np.array(list(range(-40,20)))*50
    plt.hist(matrix.reshape(-1),bins)
    print(np.max(matrix))
    #plt.imshow(matrix)
    plt.show()
def vis_2d(points,ax2,color='b'):
    select_id = np.array(points[:,1]>0) & np.array(points[:,1]<10)
    data_x = points[select_id,0]
    data_y = points[select_id,2]
    data_z = -points[select_id,1]
    ax2.plot(data_y,data_z,'.'+color)
def main():
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    points_all = np.loadtxt(sys.argv[1])
    points_left = np.loadtxt(sys.argv[2])
    vis_2d(points_all,ax2)
    vis_2d(points_left,ax2,'g')
    plt.show()
    #depth_check(points)
if __name__ == '__main__':
    main()

