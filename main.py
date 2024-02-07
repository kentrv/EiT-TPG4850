from GAN_3D.ShapeNetVoxelizer import ShapeNetVoxelizer
import os
import open3d as o3d
import numpy as np
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":
    voxelizer = ShapeNetVoxelizer(resolution=32)
    obj_path = os.getcwd()+'/Datasets/ShapeNet/model_normalized.obj'
    voxel_array = voxelizer.process_obj_file(obj_path)
    print(voxel_array.shape)  # Should print (32, 32, 32)
    print(np.unique(voxel_array, return_counts=True))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the coordinates of filled voxels
    x, y, z = np.indices(voxel_array.shape)
    x, y, z = x[voxel_array == 1], y[voxel_array == 1], z[voxel_array == 1]

    # Plot filled voxels
    ax.scatter(x, y, z, zdir='z', c='red', marker='s')

    # Set plot labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Voxel Visualization')

    # Adjust the viewing angle if needed
    ax.view_init(elev=20, azim=30)
    
    #plt.savefig('voxel_visualization.png')
    plt.show()