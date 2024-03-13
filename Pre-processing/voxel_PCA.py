import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def align_voxel_array(voxel_array):
    # Extract x, y, z coordinates where voxel exists
    x, y, z = np.where(voxel_array)
    voxel_coords = np.vstack((x, y, z)).T

    # Perform PCA and align the object along the principal axes
    pca = PCA(n_components=3)
    voxel_coords_centered = voxel_coords - np.mean(voxel_coords, axis=0)  # Center the data
    pca.fit(voxel_coords_centered)
    aligned_voxels = pca.transform(voxel_coords_centered)

    # Translate aligned voxels to ensure all coordinates are positive
    min_aligned = aligned_voxels.min(axis=0)
    aligned_voxels -= min_aligned

    # Determine the size of the grid needed for the aligned voxels
    grid_size = np.ceil(aligned_voxels.max(axis=0)).astype(int) + 1
    aligned_voxel_grid = np.zeros(grid_size, dtype=bool)

    # Populate the aligned voxel grid
    for x, y, z in aligned_voxels.astype(int):
        aligned_voxel_grid[x, y, z] = True

    return aligned_voxel_grid

def padded_align_voxel_array(voxel_array):
    # Extract x, y, z coordinates where voxel exists
    x, y, z = np.where(voxel_array)
    voxel_coords = np.vstack((x, y, z)).T

    # Perform PCA and align the object along the principal axes
    pca = PCA(n_components=3)
    voxel_coords_centered = voxel_coords - np.mean(voxel_coords, axis=0)  # Center the data
    pca.fit(voxel_coords_centered)
    aligned_voxels = pca.transform(voxel_coords_centered)

    # Translate aligned voxels to ensure all coordinates are positive
    min_aligned = aligned_voxels.min(axis=0)
    aligned_voxels -= min_aligned

    # Determine the size of the bounding box for the aligned voxels
    aligned_max = aligned_voxels.max(axis=0).astype(int)
    #grid_size = np.max([aligned_max + 1, np.array([32, 32, 32])], axis=0)  # Ensure minimum size is 32x32x32

    # Initialize the aligned voxel grid with padding to make it 32x32x32
    padded_aligned_voxel_grid = np.zeros((32, 32, 32), dtype=bool)

    # Calculate padding offsets to center the object
    offsets = ((np.array([32, 32, 32]) - (aligned_max + 1)) / 2).astype(int)

    # Populate the aligned voxel grid with the object centered
    for voxel in aligned_voxels.astype(int):
        padded_aligned_voxel_grid[
            min(voxel[0] + offsets[0], padded_aligned_voxel_grid.shape[0] - 1),
            min(voxel[1] + offsets[1], padded_aligned_voxel_grid.shape[1] - 1),
            min(voxel[2] + offsets[2], padded_aligned_voxel_grid.shape[2] - 1)
        ] = True

    return padded_aligned_voxel_grid




