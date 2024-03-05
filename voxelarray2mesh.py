import trimesh # for converting voxel_array grids to meshes
import time
import os 

from GAN_3D.ShapeNetVoxelizer import ShapeNetVoxelizer

class voxel2mesh:
    def __init__(self, voxel_array=0, mesh=0, pitch=1.0, scaling=0.3):
        self.voxel_array = voxel_array
        self.mesh = mesh

    def copy(self):
        return self.mesh.copy()

    def voxel_to_mesh(self, pitch=1.0, scaling=0.3):
        # Convert the voxel array to a mesh
        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(self.voxel_array, pitch=pitch)
        # Adjusting the mesh
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.apply_scale(scaling)
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_winding(mesh)
        self.mesh = mesh        
        return self
    
    def humphrey_smoothing(self, alpha=1, beta=1, iterations=100):
        trimesh.smoothing.filter_humphrey(self.mesh, alpha, beta, iterations)
        return self
    
    def laplacian_smoothing(self, lamb=0.5, iterations=10):
        trimesh.smoothing.filter_laplacian(self.mesh, lamb, iterations)
        return self
    
    def taubin_smoothing(self, lamb=0.5, nu=0.5, iterations=10):
        trimesh.smoothing.filter_taubin(self.mesh, lamb, nu, iterations)
        return self
    
    def subdivide(self, iterations=3):
        self.mesh.subdivide_loop(iterations)
        return self
    
    def translate(self, coordinates):
        self.mesh.apply_translation(coordinates)
        return self
    
    def xbounds(self):
        bounds = self.mesh.bounds
        return [bounds[0][0],bounds[1][0]]
    
    def ybounds(self):
        bounds = self.mesh.bounds
        return [bounds[0][1],bounds[1][1]]
    
    def zbounds(self):
        bounds = self.mesh.bounds
        return [bounds[0][2],bounds[1][2]]

    def view(self):
        self.mesh.show()

    def export_mesh(self, directory, file_name=0):
        if not os.path.exists(directory):
            os.makedirs(directory)
        if file_name:
            file_obj = directory + f"/{file_name}.stl"
        else:
            file_obj = directory + f"/mesh{str(time.time())}.stl"
        trimesh.exchange.export.export_mesh(
            mesh=self.mesh,
            file_obj=file_obj,
            file_type="stl"
        )


if __name__ == "__main__":
        
    voxelizer = ShapeNetVoxelizer(resolution=256)
    obj_path = os.getcwd()+'/Datasets/ShapeNet/model_normalized.obj'
    voxel_array = voxelizer.process_obj_file(obj_path)

    # Generate a folder to store the mesh
    directory = "./vox2mesh"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate mesh
    mesh = voxel2mesh(voxel_array).voxel_to_mesh()

    #Create scene
    scene = trimesh.scene.scene.Scene()
    scene.add_geometry(mesh.mesh)

    # Smooth mesh and add to scene

    # Humphrey smoothing
    """ xbounds = mesh.xbounds()
    ybounds = mesh.ybounds()
    for beta in range(50, 1, -10):
        for alpha in range(50, 1, -10):
            temp_mesh = voxel2mesh(mesh=mesh.copy())
            temp_mesh.humphrey_smoothing(alpha=alpha/100, beta=beta/100, iterations=50)
            temp_mesh.translate([xbounds[1]+20,ybounds[1]+20,0])
            xbounds = temp_mesh.xbounds()
            scene.add_geometry(temp_mesh.mesh)
        xbounds = mesh.xbounds()
        ybounds = temp_mesh.ybounds() """
    
    # Laplacian smoothing
    """ xbounds = mesh.xbounds()
    for lamb in range(1, 10, 1):
        temp_mesh = voxel2mesh(mesh=mesh.copy())
        temp_mesh.laplacian_smoothing(lamb=lamb/10, iterations=50)
        temp_mesh.translate([xbounds[1]+20,0,0])
        xbounds = temp_mesh.xbounds()
        scene.add_geometry(temp_mesh.mesh) """

    # taubin smoothing
    xbounds = mesh.xbounds()
    for lamb in range(1, 10, 1):
        temp_mesh = voxel2mesh(mesh=mesh.copy())
        temp_mesh.taubin_smoothing(lamb=lamb/10, nu=lamb/10, iterations=50)
        temp_mesh.translate([xbounds[1]+20,0,0])
        xbounds = temp_mesh.xbounds()
        scene.add_geometry(temp_mesh.mesh)
    
    # View scene
    scene.show()


