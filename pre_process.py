import os
import numpy as np
from pytorch3d.datasets import ShapeNetCore
from GAN_3D.ShapeNetVoxelizer import ShapeNetVoxelizer

# Initialize the voxelizer
voxelizer = ShapeNetVoxelizer(resolution=32)

# Initialize ShapeNetCore dataset
SHAPENET_PATH = "D:/ShapeNet"
shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=2)

# Define the root directory for the voxelized models
VOXELIZED_PATH = "D:/ShapeNet_Voxelized"

# Iterate over the dataset
for i in range(len(shapenet_dataset)):
    # Get the model data
    model_data = shapenet_dataset[i]

    # Load the obj file
    obj_path = os.path.join(SHAPENET_PATH, model_data['synset_id'], model_data['model_id'], shapenet_dataset.model_dir)

    # Process the Meshes object to get its voxel array representation
    voxel_array = voxelizer.process_obj_file(obj_path)

    # Create a directory for the synset if it doesn't exist
    synset_dir = os.path.join(VOXELIZED_PATH, model_data['synset_id'])
    os.makedirs(synset_dir, exist_ok=True)

    # Create a directory for the model if it doesn't exist
    model_dir = os.path.join(synset_dir, model_data['model_id'])
    os.makedirs(model_dir, exist_ok=True)

    # Save the voxel array to the model directory
    np.save(os.path.join(model_dir, 'voxel.npy'), voxel_array)

    print(f"Processed model {i+1}/{len(shapenet_dataset)}")