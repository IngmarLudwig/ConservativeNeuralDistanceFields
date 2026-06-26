import os
import numpy as np
import torch
from inside_checks import are_points_inside_one_of_the_tetrahedra
from util import get_device, ensure_tensor_and_batched


class TetMesh():
    def __init__(self, ele_file_path, node_file_path, in_out_all="all", inside_point=None, indices_start_at_one=False):
        # Relying on the tetgen output format to bes stored in /off/tetgen_data/
        # in_out_all defines wheter the outside, the inside or both are stored. in and out require tetgen to be run with -A.
        
        self.vertices = _read_nodes(node_file_path)
        self.tetrahedra = self._read_tets(ele_file_path , in_out_all, inside_point, indices_start_at_one)

        print(f"Loaded tet mesh with {len(self.tetrahedra)} tetrahedra and {len(self.vertices)} vertices from {node_file_path} and {ele_file_path}. Max index in tets: {self.tetrahedra.max()}, Min index in tets: {self.tetrahedra.min()}")
        
        # store axis-aligned bounding box
        self.lower_point, _ = self.vertices.min(axis=0)
        self.upper_point, _ = self.vertices.max(axis=0)
        
    def are_points_inside(self, points, batch_size_spec=None, epsilon=0):
        """ Returns a tensor of shape (num_points,) with negative values for points inside the tet mesh and positive values for points outside. 
            Points exactly on the surface are considered inside and get value -1. 
        """
        return _are_points_inside_tets(points, self.tetrahedra, self.vertices, batch_size_spec, epsilon)
    
    def get_point_inside_tet_mesh(self):
        """ Returns a point that is guaranteed to be inside the tet mesh. This is done by taking the mean of the vertices of the first tetrahedron."""
        first_tet = self.tetrahedra[0]
        first_tet_vertices = self.vertices[first_tet]
        center = first_tet_vertices.mean(axis=0)
        return center
                
    def calculate_total_volume(self):
        """Calculate the total volume of all tetrahedra in the mesh.
        
        Uses the scalar triple product formula for tetrahedral volume:
        V = |((p2-p1) × (p3-p1)) · (p4-p1)| / 6
        
        This represents 1/6 of the volume of the parallelepiped formed by the three edge vectors.
        """
        # Get all vertices for each tetrahedron
        tet_points = self.vertices[self.tetrahedra]  # Shape: (num_tets, 4, 3)
        
        p1 = tet_points[:, 0]  # Shape: (num_tets, 3)
        p2 = tet_points[:, 1]
        p3 = tet_points[:, 2]
        p4 = tet_points[:, 3]
        
        # Form edge vectors from p1 (the reference vertex)
        v1 = p2 - p1  # Vector from p1 to p2
        v2 = p3 - p1  # Vector from p1 to p3
        v3 = p4 - p1  # Vector from p1 to p4
        
        # Compute cross product: v1 × v2
        cross_product = torch.cross(v1, v2, dim=1)  # Shape: (num_tets, 3)
        
        # Compute dot product: (v1 × v2) · v3 (scalar triple product)
        scalar_triple_product = (cross_product * v3).sum(dim=1)  # Shape: (num_tets,)
        
        # Volume = |scalar_triple_product| / 6
        volumes = torch.abs(scalar_triple_product) / 6.0
        
        # Sum all volumes
        total_volume = volumes.sum().item()
        
        return total_volume

    ##### Internal functions #####
    #########################################################################################

    def _read_tets(self, ele_file_path, in_out_or_all, inside_point, indices_start_at_one):
        # Check if ele_file_path exists
        if not os.path.exists(ele_file_path):
            raise FileNotFoundError(f"Element file {ele_file_path} does not exist.")
        
        # Read the element file
        with open(ele_file_path, 'r') as f:
            lines = f.readlines()

        # The first line of the element file contains the number of tetrahedra
        num_tets = int(lines[0].split()[0])
        
        tets_1 = []
        tets_2 = []
        # start from line 1 since line 0 is the header
        for i in range(1, num_tets + 1):
            line = lines[i].split()
            # Depending on the selection, whether all, only inside or only outside tetrahedra are to be used, we store them in different lists.
            # in the case of "all", we store all tetrahedra in tets_1.
            # in the case of "in" and "out", we store the tetrahedra from one side of the surface in tets_1, the other in tets_2.
            # The side is determined by the 6th column in the .ele file, which is 1 for one side of the surface and 2 for the other side. 
            # This collumn is written be TetGen. We do not know which side is inside and which is outside, so we will check later with a point that is known to be inside.
            if in_out_or_all == "all":
                _append(tets_1, line)
                continue
            elif int(line[5]) == 1:
                _append(tets_1, line)
            elif int(line[5]) == 2:
                _append(tets_2, line)
        
        tets_1 = torch.tensor(tets_1, dtype=torch.int32)
        tets_2 = torch.tensor(tets_2, dtype=torch.int32)
        
        # Tetgen indices start at 1, but we want them to start at 0.
        if indices_start_at_one:
            tets_1 -= 1
            tets_2 -= 1
        
        # In the case of "all", we can return tets_1 directly, which contains all tetrahedra in this case.
        if in_out_or_all == "all":
            return tets_1
        # In the case of "in" or "out", we need to check which side is inside and which is outside.
        elif in_out_or_all == "in" or in_out_or_all == "out":
            # A simple hack to use the are_points_inside function which needs multiple points to work.
            inside_point = torch.stack([inside_point, inside_point])
            
            inside_point = ensure_tensor_and_batched(inside_point)     
            
            # We temporarily set self.tetrahedra to tets_1, so that we can use the are_points_inside function.
            self.tetrahedra = tets_1
            tets_1_inside = self.are_points_inside(inside_point)
            self.tetrahedra = None

            # If the first point is inside of the tets in tets_1, then tets_1 are the inside tetrahedra.
            if tets_1_inside[0] <= 0:
                tets_1_inside = True
            else:
                tets_1_inside = False
            
            # Therefore, if we want the inside tetrahedra, we return tets_1 if tets_1 are inside 
            if  in_out_or_all == "in":
                if tets_1_inside:
                    return tets_1
                else:
                    return tets_2
            # and vice versa for the outside tetrahedra.
            elif in_out_or_all == "out":
                if tets_1_inside:
                    return tets_2
                else:
                    return tets_1       

def _append(tets, line):
    tets.append([int(line[1]), int(line[2]), int(line[3]), int(line[4])]) 

def _read_nodes(node_file_path):
    # Check if node_file_path exists
    if not os.path.exists(node_file_path):
        raise FileNotFoundError(f"Node file {node_file_path} does not exist.")

    # Read the node file
    with open(node_file_path, 'r') as f:
        lines = f.readlines()
    
    # The first line of the node file contains the number of nodes
    num_nodes = int(lines[0].split()[0])
    
    # Initialize the vertices array as a numpy array
    vertices = np.zeros((num_nodes, 3), dtype=np.float32)
    
    # start from line 1 since line 0 is the header
    for i in range(1, num_nodes + 1):
        line = lines[i].split()
        # Use i-1 as index since we want to add the vertex in line 1 to index 0 of the vertices array, since line 0 is the header.
        # The first column is the index of the vertex, which we ignore since we assume that the vertices are ordered by their index.
        # The next three columns are the x, y, z coordinates of the vertex.
        vertices[i - 1] = [float(line[1]), float(line[2]), float(line[3])]
        
    vertices = np.array(vertices, dtype=np.float32)
    vertices = torch.tensor(vertices)
    return vertices


def _are_points_inside_tets(points, tets, vertices, batch_size_spec=None, epsilon=0):
    points = ensure_tensor_and_batched(points)
    device = get_device()

    # To improve performance, we first check if the points are inside the bounding box 
    lower_point  = vertices.min(axis=0)
    upper_point  = vertices.max(axis=0)
    
    # check if lower_point and upper_point are tuples or tensors
    if isinstance(lower_point, tuple):
        lower_point = lower_point[0]
    if isinstance(upper_point, tuple):
        upper_point = upper_point[0]
    
    # Check for each dimension if the points are outside the bounding box
    are_outside_x = (points[:, 0] < lower_point[0]) | (points[:, 0] > upper_point[0])
    are_outside_y = (points[:, 1] < lower_point[1]) | (points[:, 1] > upper_point[1])
    are_outside_z = (points[:, 2] < lower_point[2]) | (points[:, 2] > upper_point[2])
    are_outside = are_outside_x | are_outside_y | are_outside_z
    
    # Points that are outside the bounding box are definitely outside the tet mesh
    inside_points = points[~are_outside]
    
    # For the remaining points, we check if they are inside the tetrahedra
    
    # get the corner vertices of the tetrahedra, 
    # the first point of each tet in one array, the second in another array, etc.
    tet_points = vertices[tets]
    p_1s = tet_points[:, 0]
    p_2s = tet_points[:, 1]
    p_3s = tet_points[:, 2]
    p_4s = tet_points[:, 3]
    
    # use a parrallelized function to check if the points are inside one of the tetrahedra
    are_inside_tets = are_points_inside_one_of_the_tetrahedra(inside_points, p_1s, p_2s, p_3s, p_4s, batch_size_spec, epsilon)        
    are_inside_tets = are_inside_tets.to(device)
    
    # Then we create the return values, positive values are outside
    are_inside = torch.ones(len(points), dtype=torch.float32).to(device)
    are_inside[~are_outside] = are_inside_tets  
    return are_inside.cpu()
