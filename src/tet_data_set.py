import os
import torch
import numpy as np
import miniball
from tqdm import tqdm
from tetrahedralization_with_tetgen import create_simple_tet_mesh_from_off, generate_tet_mesh_in_with_tetgen, generate_tet_mesh_out_with_tetgen


class TetDataSet:
    def __init__(self):
        self.AABB_lower = None
        self.AABB_upper = None
        self.points = None
        self.radii = None # radii are only computed for tetrahedron-based points
        
    def _save_points_to_buffer(self, buffer_path, are_points_inside_tet_mesh):
        """Saves AABB_lower, AABB_upper, points and radii to buffer files"""
        torch.save(self.AABB_lower, os.path.join(buffer_path, "AABB_lower.pt"))
        torch.save(self.AABB_upper, os.path.join(buffer_path, "AABB_upper.pt"))
        torch.save(self.points,     os.path.join(buffer_path, "points.pt"))
        if are_points_inside_tet_mesh:
            torch.save(self.radii, os.path.join(buffer_path, "radii.pt"))
            
    def calculate_total_volume_definition_volume(self):
        """Calculates the total volume of the definition volume defined by the AABB and the offset"""
        offsetted_aabb_lower = self.AABB_lower - self.offset
        offsetted_aabb_upper = self.AABB_upper + self.offset
        dimensions = offsetted_aabb_upper - offsetted_aabb_lower
        total_volume = torch.prod(dimensions).item()
        return total_volume


class TetDataSetTetBased(TetDataSet):
    def __init__(self, mesh_data_path, inside_shape, offset=0.0, shape_name=None, maxvolume=None, max_edge_length=None): 
        super().__init__()
        # cached sphere tree (built lazily)
        self._sphere_tree = None

        # try to load from buffer
        buffer_path = _create_buffer_path_and_create_dir(shape_name, True, False, inside_shape, None, offset, maxvolume, max_edge_length)
        self.AABB_lower, self.AABB_upper, self.points, self.radii = _load_points_from_buffer(buffer_path=buffer_path, points_inside_tet_mesh=True)
        self.offset = offset
        # if successful, return
        if all(x is not None for x in [self.AABB_lower, self.AABB_upper, self.points, self.radii]):
            return
        print(f"Generating tet-based points and saving to buffer at {buffer_path}")

        # otherwise generate tet mesh and compute the points from the it
        if inside_shape:
            tet_mesh = generate_tet_mesh_in_with_tetgen(shape_name, mesh_data_path, max_edge_length)
        else:
            tet_mesh_in_simple = create_simple_tet_mesh_from_off(shape_name, mesh_data_path)
            one_inside_point = tet_mesh_in_simple.get_point_inside_tet_mesh()
            tet_mesh = generate_tet_mesh_out_with_tetgen(shape_name, mesh_data_path, one_inside_point, max_edge_length, offset)
        
        # axis-aligned bounding box
        self.AABB_lower = tet_mesh.lower_point
        self.AABB_upper = tet_mesh.upper_point   
        
        # the grid points are the center of the (potentially split) voxel
        # Calculate the circumcenters of the tetrahedra
        vertices = tet_mesh.vertices.numpy()
        tetrahedra = tet_mesh.tetrahedra.numpy()
        
        points = np.zeros((len(tetrahedra), 3))
        radii_squared = np.zeros((len(tetrahedra)))
        for i, tet in enumerate(tetrahedra):
            p_1 = vertices[tet[0]]
            p_2 = vertices[tet[1]]
            p_3 = vertices[tet[2]]
            p_4 = vertices[tet[3]]
            tet_points = np.stack([p_1, p_2, p_3, p_4])
            points[i], radii_squared[i] = miniball.get_bounding_ball(tet_points)
        
        self.points = torch.tensor(points, dtype=torch.float32)
        radii = np.sqrt(radii_squared)
        self.radii = torch.tensor(radii, dtype=torch.float32) #+ epsilon
        
        # find index of largest radius
        index = torch.argmax(self.radii)
        tet_largest_radius = tet_mesh.tetrahedra[index]
        p_1, p_2, p_3, p_4 = [tet_mesh.vertices[i] for i in tet_largest_radius]
        print("points tetrahedron largest radius", p_1, p_2, p_3, p_4)
        print("center largest radius", self.points[index])
        print("largest radius", self.radii[index])
        
                 
        # set points to dtype=torch.float32
        self.points = self.points.to(torch.float32)

        self._save_points_to_buffer(buffer_path, True)


class TetDataSetRandom(TetDataSet):
    def __init__(self, mesh_data_path, are_points_inside_tet_mesh, n_points, shape_name, offset, train_test_val): 
        super().__init__()
        # try to load from buffer
        buffer_path = _create_buffer_path_and_create_dir(shape_name, are_points_inside_tet_mesh, True, False, n_points, offset, None, None, train_test_val=train_test_val)
        self.AABB_lower, self.AABB_upper, self.points, self.radii = _load_points_from_buffer(buffer_path=buffer_path, points_inside_tet_mesh=False)
        self.offset = offset
        # if successful, return
        if all(x is not None for x in [self.AABB_lower, self.AABB_upper, self.points]):
            return
        print(f"Generating tet-based points and saving to buffer at {buffer_path}")
        
        # otherwise generate tet mesh and compute the points from the it
        tet_mesh_simple = create_simple_tet_mesh_from_off(shape_name, mesh_data_path)
        
        # axis-aligned bounding box
        self.AABB_lower = tet_mesh_simple.lower_point
        self.AABB_upper = tet_mesh_simple.upper_point   
        
        self.points = _find_random_points(self.AABB_lower, self.AABB_upper, tet_mesh_simple, n_points, offset, are_points_inside_tet_mesh)  
        
        self._save_points_to_buffer(buffer_path, are_points_inside_tet_mesh)


def _create_buffer_path_and_create_dir(shape_name, are_points_inside_tet_mesh, random, inside_shape,n_points, offset, maxvolume, max_edge_length, train_test_val=""):
    if train_test_val != "":
        train_test_val = f"_{train_test_val}"
    settings_str = f"points_inside_tet_mesh_{are_points_inside_tet_mesh}_random_{random}_inside_shape_{inside_shape}_n_points_{n_points}_offset_{offset}_maxvolume_{maxvolume}_max_edge_length_{max_edge_length}{train_test_val}"
    buffer_path = f"../point_cache/{shape_name}/{settings_str}"
    # if path does not exist, create it
    if not os.path.exists(buffer_path):
        os.makedirs(buffer_path)
    return buffer_path


def _load_points_from_buffer(buffer_path, points_inside_tet_mesh):
    #print(f"Trying to load points from buffer at {buffer_path}")
    
    AABB_lower_path = os.path.join(buffer_path, "AABB_lower.pt")
    AABB_upper_path = os.path.join(buffer_path, "AABB_upper.pt")
    points_path = os.path.join(buffer_path, "points.pt")
    radii_path = os.path.join(buffer_path, "radii.pt")
    
    paths_to_check = [AABB_lower_path, AABB_upper_path, points_path]
    
    # In the case of tetrahedron-based points, also check for the radii file
    if points_inside_tet_mesh:
        paths_to_check.append(radii_path)

    # if not all paths exist, return None
    if not all(os.path.exists(p) for p in paths_to_check):
        print("No cached points found.")
        return None, None, None, None
    
    AABB_lower = torch.load(os.path.join(buffer_path, "AABB_lower.pt"), weights_only=False)
    AABB_upper = torch.load(os.path.join(buffer_path, "AABB_upper.pt"), weights_only=False)
    points = torch.load(os.path.join(buffer_path, "points.pt"), weights_only=False)
    
    # In the case of tetrahedron-based points, also load the radii
    if points_inside_tet_mesh:
        radii = torch.load(os.path.join(buffer_path, "radii.pt"), weights_only=False)
    else:
        radii = None
    
    #print(f"Loaded points from buffer")
    return AABB_lower, AABB_upper, points, radii


def _find_random_points(AABB_lower, AABB_upper, tet_mesh, n_points_total, offset, are_points_inside_tet_mesh):
        offsetted_aabb_lower = AABB_lower - offset
        offsetted_aabb_upper = AABB_upper + offset
        
        n_points = 0
        calculation_batch_size = 10_000
        points = []
        
        with tqdm(total=n_points_total, desc="Generating random points", unit="pts") as pbar:
            while n_points < n_points_total:
                points_batch = torch.rand((calculation_batch_size, 3)) * (offsetted_aabb_upper - offsetted_aabb_lower) + offsetted_aabb_lower
                are_inside = tet_mesh.are_points_inside(points_batch) 
                if not are_points_inside_tet_mesh:
                    are_inside = are_inside == 1    
                    new_points = are_inside.sum().item()
                    n_points += new_points
                    points.append(points_batch[are_inside])
                    pbar.update(new_points)
                else:
                    are_outside = are_inside == -1
                    new_points = are_outside.sum().item()
                    n_points += new_points
                    points.append(points_batch[are_outside])
                    pbar.update(new_points)
        points = torch.cat(points)
        points = points[:n_points_total]
        points = points.to(torch.float32)
        return points
    

