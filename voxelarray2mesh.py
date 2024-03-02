import trimesh # for converting voxel_array grids to meshes (to import objects into simulators)
import time # to know how long it takes for the code to run
import os # to walk through directories, to rename files
import numpy as np


from GAN_3D.ShapeNetVoxelizer import ShapeNetVoxelizer

# Parses a file of type BINVOX
# Returns a voxel_array grid, generated using the binvox_rw.py package
def parse_BINVOX_file_into_voxel_grid(voxel_array):
    voxelgrid = voxel_array
    return voxelgrid

if __name__ == "__main__":
        
    voxelizer = ShapeNetVoxelizer(resolution=256)
    obj_path = os.getcwd()+'/Datasets/ShapeNet/model_normalized.obj'
    voxel_array = voxelizer.process_obj_file(obj_path)
    voxel_array = np.array([[voxel_array]]) 
    print(voxel_array)

    # Generate a folder to store the images

    directory = "./vox2mesh"
    if not os.path.exists(directory):
        print("Generating a folder to save the mesh")
        os.makedirs(directory)

    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(matrix=voxel_array[0][0], pitch=1.0)

    print("Merging vertices closer than a pre-set constant...")
    mesh.merge_vertices()
    print("Removing duplicate faces...")
    mesh.remove_duplicate_faces()
    print("Scaling...")
    mesh.apply_scale(scaling=1.0)
    print("Making the mesh watertight...")
    trimesh.repair.fill_holes(mesh)
    print("Fixing inversion and winding...")
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_winding(mesh)
    print("Smoothing the mesh...")
    trimesh.smoothing.filter_humphrey(mesh, alpha=0.01, beta=0.1, iterations=100)

    print("Generating the STL mesh file")
    trimesh.exchange.export.export_mesh(
        mesh=mesh,
        file_obj=directory + f"/mesh_{str(time.time())}.stl",
        file_type="stl"
    )



