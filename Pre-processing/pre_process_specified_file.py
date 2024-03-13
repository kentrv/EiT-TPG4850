import os
import numpy as np
from pytorch3d.datasets import ShapeNetCore
from GAN_3D.ShapeNetVoxelizer import ShapeNetVoxelizer
from voxel_PCA import padded_align_voxel_array

# Initialize the voxelizer
voxelizer = ShapeNetVoxelizer(resolution=32)

# Initialize ShapeNetCore dataset
SHAPENET_PATH = "D:/ShapeNet"
shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=2)

# Define the root directory for the voxelized models
VOXELIZED_PATH = "D:/ShapeNet_Voxelized"

# Specify the synset_id and model_id of the model you want to process
synset_id = "your_synset_id"
model_id = "your_model_id"

# Load the obj file
obj_path = os.path.join(SHAPENET_PATH, synset_id, model_id, shapenet_dataset.model_dir)

# Check if the obj file exists
if not os.path.isfile(obj_path):
    print(f"No obj file found for model {model_id}. Skipping this model.")
else:
    # Process the Meshes object to get its voxel array representation
    voxel_array = voxelizer.process_obj_file(obj_path)

    # Align the voxel array using PCA and pad it to 32x32x32
    aligned_voxel_array = padded_align_voxel_array(voxel_array)

    # Create a directory for the synset if it doesn't exist
    synset_dir = os.path.join(VOXELIZED_PATH, synset_id)
    os.makedirs(synset_dir, exist_ok=True)

    # Create a directory for the model if it doesn't exist
    model_dir = os.path.join(synset_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)

    # Save the voxel array and the aligned voxel array to the model directory
    np.save(os.path.join(model_dir, 'voxel.npy'), voxel_array)
    np.save(os.path.join(model_dir, 'aligned_voxel.npy'), aligned_voxel_array)

    print(f"Processed model {model_id}")