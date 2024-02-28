from GAN_3D.ShapeNetVoxelizer import ShapeNetVoxelizer
from GAN_3D.Discriminator import Discriminator
from GAN_3D.Generator import Generator
from GAN_3D.LRCN import LRCNModel
import os
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import ImageGrid



def preprocess_volume(volume, sequence_length=5):
    """
    Preprocess the volume by slicing the volume, and for each slice, make a sequence of length sequence_length.
    
    Args:
    volume: A numpy array of shape (dl, dl, dl) representing the voxel array.
    sequence_length: How many slices in the sequence per slice.
    
    Returns: A numpy array of shape [dl, sequence_length, dl, dl], that is: a numpy array representing dl sequences of length sequence_length containing a dl x dl 2D voxel array.
    """
    dl = volume.shape[0]
    padded_volume = np.pad(volume, ((sequence_length//2, sequence_length//2), (0,0), (0,0)), mode='constant', constant_values=0)
    sequences = np.zeros((dl, sequence_length, dl, dl))
    
    for i in range(dl):
        start_idx = i
        end_idx = start_idx + sequence_length
        sequences[i] = padded_volume[start_idx:end_idx]
        
    return sequences
            

if __name__ == "__main__":
    voxelizer = ShapeNetVoxelizer(resolution=32)
    obj_path = os.getcwd()+'/Datasets/ShapeNet/model_normalized.obj'
    voxel_array = voxelizer.process_obj_file(obj_path)
    preprocessed_slices = preprocess_volume(voxel_array)
    voxel_array = np.array([[voxel_array]])
    print(voxel_array.shape)  # Should print (32, 32, 32)
    voxel_tensor = torch.from_numpy(voxel_array).float()
    print(preprocessed_slices.shape)
    preprocessed_slices = torch.from_numpy(preprocessed_slices).float().unsqueeze(0)
    dataloader = DataLoader(voxel_array, batch_size=32, shuffle=True)
    (i, data) = zip(*enumerate(dataloader))
    data = data[0]
    generator = Generator(input_size=[64, 1, 32, 32, 32])
    discriminator = Discriminator(input_size=[64, 1, 32, 32, 32])
    lrcn = LRCNModel(dl=32, dh=64)
    output = discriminator(voxel_tensor)
    output = generator(voxel_tensor)
    output = lrcn(preprocessed_slices)
    voxel_array = output.detach().numpy()[0][0]
    print(voxel_array.shape)
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


