import os
import numpy as np
from pytorch3d.datasets import ShapeNetCore
from GAN_3D.ShapeNetVoxelizer import ShapeNetVoxelizer
from voxel_PCA import padded_align_voxel_array
import PIL
import sys

# Initialize the voxelizer
voxelizer = ShapeNetVoxelizer(resolution=32)

# Initialize ShapeNetCore dataset
if sys.platform == "win32":
    SHAPENET_PATH = "D:/ShapeNet"
    shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=2)
    # Define the root directory for the voxelized models
    VOXELIZED_PATH = "D:/ShapeNet_Voxelized"
    
elif sys.platform == "linux":
    user = os.environ.get('USER')
    SHAPENET_PATH = "/media/"+user+"/Elements/ShapeNet"
    shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=2)
    # Define the root directory for the voxelized models
    VOXELIZED_PATH = "/media/"+user+"/Elements/ShapeNet_Voxelized"
else:
    print("Unsupported OS. Exiting...")
    sys.exit()


# Open the error log file
with open('error_log.txt', 'a') as error_log:
    # Iterate over the dataset
    for i in range(len(shapenet_dataset)):
        try:
            # Get the model data
            model_data = shapenet_dataset[i]

            # Load the obj file
            obj_path = os.path.join(SHAPENET_PATH, model_data['synset_id'], model_data['model_id'], shapenet_dataset.model_dir)

            # Check if the obj file exists
            if not os.path.isfile(obj_path):
                fileNotFound = f"No obj file found for model {i+1}/{len(shapenet_dataset)} at path {obj_path}. Skipping this model."
                print(fileNotFound)
                error_log.write(fileNotFound)
                continue

            # Process the Meshes object to get its voxel array representation
            voxel_array = voxelizer.process_obj_file(obj_path)

            # Align the voxel array using PCA and pad it to 32x32x32
            aligned_voxel_array = padded_align_voxel_array(voxel_array)

            # Create a directory for the synset if it doesn't exist
            synset_dir = os.path.join(VOXELIZED_PATH, model_data['synset_id'])
            os.makedirs(synset_dir, exist_ok=True)

            # Create a directory for the model if it doesn't exist
            model_dir = os.path.join(synset_dir, model_data['model_id'])
            os.makedirs(model_dir, exist_ok=True)

            # Define the paths for the voxel and aligned voxel files
            voxel_file_path = os.path.join(model_dir, 'voxel.npy')
            aligned_voxel_file_path = os.path.join(model_dir, 'aligned_voxel.npy')

            # Check if the voxel and aligned voxel files already exist
            if os.path.isfile(voxel_file_path) and os.path.isfile(aligned_voxel_file_path):
                print(f"Files for model {i+1}/{len(shapenet_dataset)} already exist. Skipping this model.")
                continue

            # Process the Meshes object to get its voxel array representation
            voxel_array = voxelizer.process_obj_file(obj_path)

            # Align the voxel array using PCA and pad it to 32x32x32
            aligned_voxel_array = padded_align_voxel_array(voxel_array)

            # Save the voxel array and the aligned voxel array to the model directory
            np.save(voxel_file_path, voxel_array)
            np.save(aligned_voxel_file_path, aligned_voxel_array)

            print(f"Processed model {i+1}/{len(shapenet_dataset)}")
        except PermissionError:
            error_message = f"Permission denied for model {i+1}/{len(shapenet_dataset)} at path {obj_path}. Skipping this model.\n"
            print(error_message)
            error_log.write(error_message)
        except UnicodeDecodeError:
            error_message = f"UnicodeDecodeError for model {i+1}/{len(shapenet_dataset)} at path {obj_path}. Skipping this model.\n"
            print(error_message)
            error_log.write(error_message)
        except PIL.UnidentifiedImageError:
            error_message = f"Unidentified image error for model {i+1}/{len(shapenet_dataset)} at path {obj_path}. Skipping this model.\n"
            print(error_message)
            error_log.write(error_message)