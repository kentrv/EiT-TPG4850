import trimesh # for converting voxel_array grids to meshes
import time
import os 

from GAN_3D.ShapeNetVoxelizer import ShapeNetVoxelizer

if __name__ == "__main__":
        
    voxelizer = ShapeNetVoxelizer(resolution=256)
    obj_path = os.getcwd()+'/Datasets/ShapeNet/model_normalized.obj'
    voxel_array = voxelizer.process_obj_file(obj_path)

    # Generate a folder to store the mesh
    directory = "./vox2mesh"
    if not os.path.exists(directory):
        print("Generating a folder to save the mesh")
        os.makedirs(directory)

    # Convert the voxel array to a mesh
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(matrix=voxel_array, pitch=1.0)

    # Adjusting the mesh
    print("Merging vertices closer than a pre-set constant...")
    mesh.merge_vertices()
    print("Removing duplicate faces...")
    mesh.update_faces(mesh.unique_faces())
    print("Scaling...")
    mesh.apply_scale(scaling=0.3)
    print("Making the mesh watertight...")
    trimesh.repair.fill_holes(mesh)
    print("Fixing inversion and winding...")
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_winding(mesh)
    print("Smoothing the mesh...")
    trimesh.smoothing.filter_humphrey(mesh, alpha=0.01, beta=0.1, iterations=100)

    # Export the mesh
    trimesh.exchange.export.export_mesh(
        mesh=mesh,
        file_obj=directory + f"/mesh{str(time.time())}.stl",
        file_type="stl"
    )




