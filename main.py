from GAN_3D.ShapeNetVoxelizer import ShapeNetVoxelizer
from GAN_3D.Discriminator import Discriminator
from GAN_3D.Generator import Generator
import os
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader



if __name__ == "__main__":
    voxelizer = ShapeNetVoxelizer(resolution=32)
    obj_path = os.getcwd()+'\Datasets+\\1fbb9f70d081630e638b4be15b07b442\models\model_normalized.obj'
    voxel_array = voxelizer.process_obj_file(obj_path)
    voxel_array = np.array([[voxel_array]])
    #print(voxel_array)  # Should print (32, 32, 32)
    voxel_tensor = torch.from_numpy(voxel_array).float()
    dataloader = DataLoader(voxel_array, batch_size=32, shuffle=True)
    (i, data) = zip(*enumerate(dataloader))
    data = data[0]
    generator = Generator(input_size=[64, 1, 32, 32, 32])
    discriminator = Discriminator(input_size=[64, 1, 32, 32, 32])
    output = discriminator(voxel_tensor)
    output = generator(voxel_tensor)
    voxel_array = output.detach().numpy()[0][0]
    print(voxel_array)  
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the indices of the filled voxels
    filled_indices = voxel_array #np.argwhere(voxel_array == 1)

    # Plot the filled voxels as a 3D scatter plot
    ax.scatter(filled_indices[:, 0], filled_indices[:, 1], filled_indices[:, 2], c='red', marker='s')

    # Set plot labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Voxel Visualization')

    plt.show()
