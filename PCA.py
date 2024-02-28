from pytorch3d.datasets import ShapeNetCore
import os
from open3d import open3d as o3d
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from GAN_3D.ShapeNetVoxelizer import ShapeNetVoxelizer
from sklearn.decomposition import PCA
from scipy.ndimage import affine_transform

# Initialize ShapeNetCore dataset
SHAPENET_PATH = "D:/ShapeNet"
shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=2)

# Initialize the voxelizer
voxelizer = ShapeNetVoxelizer(resolution=32)

model = shapenet_dataset[5000]
obj_path = os.path.join(SHAPENET_PATH, model['synset_id'], model['model_id'], shapenet_dataset.model_dir)

voxel_array = voxelizer.process_obj_file(obj_path)

def voxel_grid_to_point_cloud(voxel_grid):
    points = np.argwhere(voxel_grid > 0)
    return points

def align_voxel_grid(voxel_grid):
    points = voxel_grid_to_point_cloud(voxel_grid)
    pca = PCA(n_components=3)
    pca.fit(points)
    rotation_matrix = pca.components_

    # Calculate the geometric center of the original voxel grid
    original_center = np.mean(points, axis=0)

    # Calculate the new center after applying rotation
    rotated_center = original_center.dot(rotation_matrix.T)

    # Compute the offset needed to re-center the grid
    offset = original_center - rotated_center

    # Create affine transform matrix including the rotation and adjusted offset for centering
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = rotation_matrix
    affine_matrix[:3, 3] = offset  # Apply the offset here

    # Apply the affine transformation with corrected offset
    aligned_grid = affine_transform(voxel_grid, affine_matrix[:3, :3], offset=affine_matrix[:3, 3], order=1, mode='constant', cval=0)
    return aligned_grid

# Create a figure with two 3D subplots
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Get the indices of the filled voxels
filled_indices = np.argwhere(voxel_array == 1)

# Align the voxel grid
aligned_voxel_array = align_voxel_grid(voxel_array)
filled_indices_aligned = np.argwhere(aligned_voxel_array == 1)

# Plot the filled voxels as a 3D scatter plot
ax1.scatter(filled_indices[:, 0], filled_indices[:, 1], filled_indices[:, 2], c='red', marker='s')
ax2.scatter(filled_indices_aligned[:, 0], filled_indices_aligned[:, 1], filled_indices_aligned[:, 2], c='red', marker='s')

# Set plot labels and titles
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Voxel Visualization')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Aligned Voxel Visualization')

# Display the plots
plt.show()