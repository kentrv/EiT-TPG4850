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


def preprocess_volume(volume, num_slices=5):
    """
    Preprocess the volume by slicing and normalizing it.
    
    Args:
    volume: A numpy array of shape (32, 32, 32) representing the voxel array.
    num_slices: Number of slices to create. Default is 5,
    
    Returns:
    A numpy array of sliced and normalized volumes.
    """
    # Normalize the volume to be between -1 and 1
    #normalized_volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume)) * 2 - 1
    #slices = volume.transpose(2, 0, 1) 
    
    return volume


if __name__ == "__main__":
    voxelizer = ShapeNetVoxelizer(resolution=32)
    obj_path = os.getcwd()+'/Datasets/ShapeNet/model_normalized.obj'
    voxel_array = voxelizer.process_obj_file(obj_path)
    preprocessed_slices = preprocess_volume(voxel_array)
    preprocessed_slices = torch.tensor(preprocessed_slices, dtype=torch.float).unsqueeze(0).unsqueeze(0)
    voxel_array = np.array([[voxel_array]])
    print(voxel_array.shape)  # Should print (32, 32, 32)
    voxel_tensor = torch.from_numpy(voxel_array).float()
    print(preprocessed_slices.shape)
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


#lager en tilfeldig kube som eksempel-modell
def create_3d_cube(shape=(10, 10, 10)):
    cube = np.zeros(shape)
    cube[3:7, 3:7, 3:7] = 1  # Set a region in the center of the cube to 1
    return cube

#lager batcher med slices
def get_2d_layer_3d_batch(array, layer_index):
    batch = []

    for i in range(layer_index - 2, layer_index + 3):
        if 0 <= i < array.shape[0]:
            layer = array[i, :, :]
        else:
            # If index is out of bounds, create an empty layer
            layer = np.zeros_like(array[0, :, :])
        batch.append(layer)

    return batch

# Create a 3D cube array
cube_array = create_3d_cube()

# Get batches of 5 layers for each layer index
layer_batches = []
for layer_index in range(cube_array.shape[0]):
    batch = get_2d_layer_3d_batch(cube_array, layer_index)
    layer_batches.append(batch)

# Example: Print all batches
for layer_index, batch in enumerate(layer_batches):
    print(f"Batch at Layer Index {layer_index}:")
    for i, layer in enumerate(batch):
        print(f"  Layer {i + (layer_index - 2)}:")
        print(layer)
        print()
