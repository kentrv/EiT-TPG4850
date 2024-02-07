from open3d import open3d as o3d
import numpy as np
import trimesh

class ShapeNetVoxelizer:
    def __init__(self, resolution=32):
        self.resolution = resolution
        self.voxel_size = 1.0 / self.resolution

    def load_mesh(self, obj_path):
        """
        Load a mesh from an OBJ file, ensuring it is triangulated.
        Handles both single meshes and scenes with multiple meshes.
        """
        loaded = trimesh.load(obj_path)

        # If the loaded object is a Scene, process all geometries
        if isinstance(loaded, trimesh.Scene):
            # Attempt to convert the scene to a single mesh
            # This combines all geometries in the scene into one mesh
            mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        else:
            # If it's already a Trimesh object, use it directly
            mesh = loaded

        # Ensure the mesh is triangulated
        if not mesh.is_empty and hasattr(mesh, 'faces'):
            mesh = mesh.split()[0]  # Split into individual components and take the first, if necessary
            #mesh = mesh.triangulate()

        # Convert to Open3D mesh
        if not mesh.is_empty:
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
            o3d_mesh.compute_vertex_normals()
        else:
            o3d_mesh = o3d.geometry.TriangleMesh()

        return o3d_mesh

    
    def normalize_mesh(self, mesh):
        """
        Normalize the mesh to fit within a unit cube centered at the origin.
        """
        aabb = mesh.get_axis_aligned_bounding_box()
        max_dim = max(aabb.get_extent())
        scale_factor = 1.0 / max_dim
        mesh.scale(scale_factor, center=aabb.get_center())
        mesh.translate(-mesh.get_center())
        print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.triangles)}")
        return mesh

    def mesh_to_voxel_grid(self, mesh):
        """
        Convert a mesh to a voxel grid of the specified resolution.
        """
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
            mesh,
            voxel_size=self.voxel_size
        )
        return voxel_grid

    def voxel_grid_to_array(self, voxel_grid):
        """
        Convert a voxel grid to a numpy array.
        """
        try:
            # Initialize an empty array for the voxel grid
            voxel_array = np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.uint8)

            # Assuming voxel_grid is correctly populated and aligned with the mesh
            for voxel in voxel_grid.get_voxels():
                # Calculate voxel indices based on the voxel grid's resolution and bounds
                index = voxel.grid_index
                if all(0 <= idx < self.resolution for idx in index):
                    voxel_array[index[0], index[1], index[2]] = 1

            return voxel_array
        except Exception as e:
            print(f"Error converting voxel grid to array: {e}")
            return np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.uint8)


    def process_obj_file(self, obj_path):
        """
        Full processing pipeline for converting an OBJ file to a voxel grid array.
        """
        mesh = self.load_mesh(obj_path)
        normalized_mesh = self.normalize_mesh(mesh)
        voxel_grid = self.mesh_to_voxel_grid(normalized_mesh)
        voxel_array = self.voxel_grid_to_array(voxel_grid)
        return voxel_array

