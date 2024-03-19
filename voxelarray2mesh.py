import trimesh # for converting voxel_array grids to meshes
import time
import os 

from GAN_3D.ShapeNetVoxelizer import ShapeNetVoxelizer

class voxel2mesh:
    def __init__(self, voxel_array=0, mesh=0, pitch=1.0, scaling=0.3):
        self.voxel_array = voxel_array
        self.mesh = mesh

    def convert(self, pitch=1.0, scaling=0.003):
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
    
    def copy(self):
        return self.mesh.copy()
    
    def scale(self, scaling=0.3):
        self.mesh.apply_scale(scaling)
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
    
    def to_orgio(self):
        self.translate([-self.xbounds()[0],-self.ybounds()[0],-self.zbounds()[0]])
        return self
    
    def add_to_scene(self, scene):
        scene.add_geometry(self.mesh)
        return self

    def view(self):
        self.mesh.show()

    def export(self, directory, file_name=0, filetype="stl"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        if file_name:
            file_obj = directory + f"/{file_name}.{filetype}"
        else:
            file_obj = directory + f"/mesh{str(time.time())}.{filetype}"
        trimesh.exchange.export.export_mesh(
            mesh=self.mesh,
            file_obj=file_obj,
            file_type=filetype
        )


if __name__ == "__main__":
        
    voxelizer = ShapeNetVoxelizer(resolution=128)
    folder = '1fbb9f70d081630e638b4be15b07b442'
    obj_path = os.getcwd()+f'\Datasets\\{folder}\models\model_normalized.obj'
    voxel_array = voxelizer.process_obj_file(obj_path)

    # Generate a folder to store the mesh
    directory = "./vox2mesh"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate mesh
    mesh = voxel2mesh(voxel_array).convert().to_orgio()

    t = time.time()
    humphrey = voxel2mesh(mesh=mesh.copy()).humphrey_smoothing(alpha=0.5, beta=0.5, iterations=100).translate([mesh.xbounds()[1],0,0])
    print(f"Humphrey: {time.time()-t}")

    """ t = time.time()
    laplacian = voxel2mesh(mesh=mesh.copy()).laplacian_smoothing(lamb=0.9, iterations=15).translate([humphrey.xbounds()[1],0,0])
    print(f"laplacian: {time.time()-t}") """
    
    iterations = 500
    lamb = 0.9
    nu=lamb/(1+0.1*lamb)
    t = time.time()
    taubin = voxel2mesh(mesh=mesh.copy()).taubin_smoothing(lamb=lamb, nu=nu, iterations=iterations).translate([humphrey.xbounds()[1],0,0])
    print(f"Taubin: {time.time()-t}s" )

    ekte = voxel2mesh(mesh=trimesh.exchange.load.load_mesh(obj_path)).to_orgio().translate([taubin.xbounds()[1],0,0])
    #mesh.view()

    #Create scene
    scene = trimesh.scene.scene.Scene()

    # Smooth mesh and add to scene
    mesh.add_to_scene(scene)
    humphrey.add_to_scene(scene)
    #laplacian.add_to_scene(scene)
    taubin.add_to_scene(scene)
    ekte.add_to_scene(scene)

    # Humphrey smoothing
    """ xbounds = mesh.xbounds()
    ybounds = mesh.ybounds()
    for beta in range(100, 0, -20):
        for alpha in range(100, 0, -20):
            temp_mesh = voxel2mesh(mesh=mesh.copy())
            t = time.time()
            temp_mesh.humphrey_smoothing(alpha=alpha/1000, beta=beta/100, iterations=100)
            print(f'alpha: {alpha/100}, beta: {beta/100}, time: {time.time()-t}')
            temp_mesh.translate([xbounds[1],ybounds[1],0])
            xbounds = temp_mesh.xbounds()
            scene.add_geometry(temp_mesh.mesh)
        xbounds = mesh.xbounds()
        ybounds = temp_mesh.ybounds() """
    
    # Laplacian smoothing
    """ xbounds = mesh.xbounds()
    for i in range(10, 100, 10):
        for lamb in range(0, 100, 10):
            temp_mesh = voxel2mesh(mesh=mesh.copy())
            t = time.time()
            temp_mesh.laplacian_smoothing(lamb=lamb/100, iterations=i)
            print(f'lamb: {lamb/100}, iterations: {i}, time: {time.time()-t}')
            temp_mesh.translate([xbounds[1]+20,0,0])
            xbounds = temp_mesh.xbounds()
            scene.add_geometry(temp_mesh.mesh) """
    
    

    # taubin smoothing
    """ xbounds = mesh.xbounds()
    for lamb in range(1, 10, 1):
        temp_mesh = voxel2mesh(mesh=mesh.copy())
        temp_mesh.taubin_smoothing(lamb=lamb/10, nu=lamb/10, iterations=50)
        temp_mesh.translate([xbounds[1]+20,0,0])
        xbounds = temp_mesh.xbounds()
        scene.add_geometry(temp_mesh.mesh) """
    
    """ xbounds = mesh.xbounds()
    ybounds = mesh.ybounds()
    for lamb in range(0, 100, 20):
        for nu in [lamb/100, lamb/(100+0.1*lamb)]:
            for i in range(100, 1000, 200):
                temp_mesh = voxel2mesh(mesh=mesh.copy())
                t = time.time()
                temp_mesh.taubin_smoothing(lamb=lamb/100, nu=nu, iterations=i)
                print(f'lamb: {lamb/100}, nu: {nu}, iterations: {i}, time: {time.time()-t}')
                temp_mesh.translate([xbounds[1],ybounds[1],0])
                xbounds = temp_mesh.xbounds()
                scene.add_geometry(temp_mesh.mesh)
            xbounds = mesh.xbounds()
            ybounds = temp_mesh.ybounds() """
    
    """ xbounds = mesh.xbounds()
    for i in [0, 1]:
        iterations = [500, 2000]
        lamb = [0.9, 0.45]
        nu=lamb[i]/(1+0.1*lamb[i])
        temp_mesh = voxel2mesh(mesh=mesh.copy())
        t = time.time()
        temp_mesh.taubin_smoothing(lamb=lamb[i], nu=nu, iterations=iterations[i])
        print(f'lamb: {lamb[i]}, nu: {nu}, iterations: {iterations[i]}, time: {time.time()-t}')
        temp_mesh.translate([xbounds[1],0,0])
        xbounds = temp_mesh.xbounds()
        scene.add_geometry(temp_mesh.mesh) """
    
    # View scene
    scene.show()

    # Export mesh
    taubin.export(directory, file_name=f"{folder}",filetype="obj")