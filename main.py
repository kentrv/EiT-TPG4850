from GAN_3D.ShapeNetVoxelizer import ShapeNetVoxelizer
import os
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



if __name__ == "__main__":
    voxelizer = ShapeNetVoxelizer(resolution=32)
    obj_path = os.getcwd()+'/Datasets/ShapeNet/model_normalized.obj'
    voxel_array = voxelizer.process_obj_file(obj_path)
    print(voxel_array.shape)  # Should print (32, 32, 32)
    
    
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the indices of the filled voxels
    filled_indices = np.argwhere(voxel_array == 1)

    # Plot the filled voxels as a 3D scatter plot
    ax.scatter(filled_indices[:, 0], filled_indices[:, 1], filled_indices[:, 2], c='red', marker='s')

    # Set plot labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Voxel Visualization')

    plt.show()