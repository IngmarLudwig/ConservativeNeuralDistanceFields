import os
import numpy as np
import torch
from tet_mesh import TetMesh
from generate_smesh_and_mtr import generate_smesh_and_mtr_file_from_off


######## Create tet meshes with tetgen #######

def create_simple_tet_mesh_from_off(shape, mesh_data_path):
    # create paths and names
    name_for_simple_tet_mesh = shape + "_scaled_for_simple"   
    path_to_triangle_mesh_file = _get_path_to_triangle_mesh_file(shape, mesh_data_path)
    off_path_scaled = _get_off_path_scaled(mesh_data_path, name_for_simple_tet_mesh)
    
    # Scale the mesh to fit in the unit cube
    scale_off_file_to_range_minus_1_to_1(path_to_triangle_mesh_file, off_path_scaled ) 
    
    #generate a simple tet mesh from the scaled mesh for inside checks. 
    # Use -p option to receive a .ele and .node file with the tets and nodes of the tet mesh.
    # Use -o option for using an input .off file.
    tetgen_simple_command = '../tetgen1.6.0/build/tetgen -po ' + off_path_scaled 
    print(tetgen_simple_command)
    os.system(tetgen_simple_command)
    
    ele_file_path  = mesh_data_path + name_for_simple_tet_mesh + ".1.ele"
    node_file_path = mesh_data_path + name_for_simple_tet_mesh + ".1.node"
    if not os.path.exists(ele_file_path) or not os.path.exists(node_file_path):
        raise FileNotFoundError(f"Tetgen command did not create .ele or .node file. Please check if tetgen is correctly installed and the command is correct.")
    
    # Create the TetMesh object from the generated .ele and .node file.
    tet_mesh_in_simple = TetMesh(ele_file_path, node_file_path, "all")
    
    file_endings_to_remove = [".1.edge", ".1.ele", ".1.face", ".1.node", ".1.smesh", ".off"]
    _remove_files(name_for_simple_tet_mesh, mesh_data_path, file_endings_to_remove)

    return tet_mesh_in_simple


def generate_tet_mesh_in_with_tetgen(shape, mesh_data_path, max_edge_length_inside):
    # create names and paths
    scaled_shape_name = shape + "_scaled_for_inside"    
    path_to_triangle_mesh = _get_path_to_triangle_mesh_file(shape, mesh_data_path)
    off_path_scaled = _get_off_path_scaled(mesh_data_path, scaled_shape_name)
    smesh_path_scaled = mesh_data_path + scaled_shape_name + ".smesh"
    
    # Scale the mesh to fit in the unit cube
    scale_off_file_to_range_minus_1_to_1(path_to_triangle_mesh, off_path_scaled) 
    
    # Create input for tetgen with max_edge_length. 
    # This creates a .smesh with the mesh data and an .mtr file setting the same max edge length for every vertex.
    generate_smesh_and_mtr_file_from_off(off_path_scaled, max_edge_length_inside)

    tetgen_inside_command = f'../tetgen1.6.0/build/tetgen -pm {smesh_path_scaled} -A -q1.4/10'
    print(tetgen_inside_command)
    os.system(tetgen_inside_command)
    
    ele_file_path_in  = mesh_data_path + scaled_shape_name + ".1.ele"
    node_file_path_in = mesh_data_path + scaled_shape_name + ".1.node"
    tet_mesh_in = TetMesh(ele_file_path_in, node_file_path_in, "all", indices_start_at_one=True)

    # Clean up intermediate files created by tetgen
    _remove_files(scaled_shape_name, mesh_data_path, [".1.edge", ".1.ele", ".1.face", ".1.node", ".1.mtr", ".1.node", ".1.p2t", ".mtr", ".off", ".smesh"])

    return tet_mesh_in

def generate_tet_mesh_out_with_tetgen(shape, mesh_data_path, one_inside_point, max_edge_length_outside, offset):
    scaled_shape_name = shape + "_scaled_for_outside"
    combined_shape_name = shape + "_combined"
    
    path_to_triangle_mesh = _get_path_to_triangle_mesh_file(shape, mesh_data_path)
    off_path_scaled = _get_off_path_scaled(mesh_data_path, scaled_shape_name)
    off_path_combined_meshes   = mesh_data_path + combined_shape_name + ".off"
    smesh_path_combined_meshes = mesh_data_path + combined_shape_name + ".smesh"
    
    #Scale the mesh to fit in the unit cube
    scale_off_file_to_range_minus_1_to_1(path_to_triangle_mesh, off_path_scaled ) 
        
    # Adds a simple bounding box around the shape with a certain offset.
    # This area between the shape and the bounding box is then filled with tets by tetgen.
    generate_triangle_mesh_with_bounding_box(off_path_scaled, off_path_combined_meshes, offset=offset)

    # Create input for tetgen with max_edge_length. 
    # This creates a .smesh with the mesh data and an .mtr file setting the same max edge length for every vertex.
    generate_smesh_and_mtr_file_from_off(off_path_combined_meshes, max_edge_length_outside)

    tetgen_outside_command = f'../tetgen1.6.0/build/tetgen -pqm {smesh_path_combined_meshes} -A -q1.4/10'
    print(tetgen_outside_command)
    os.system(tetgen_outside_command)
    
    ele_file_path_combined  = mesh_data_path + combined_shape_name + ".1.ele"
    node_file_path_combined = mesh_data_path + combined_shape_name + ".1.node"
    # Create the TetMesh object from the generated .ele and .node file, only keeping the outside tets by passing "out" as argument.
    # one_inside_point is used to determine which tets are inside and which are outside.
    # Tetgen starts indexing at 1, so we set indices_start_at_one to True.
    tet_mesh_out = TetMesh(ele_file_path_combined, node_file_path_combined, "out", one_inside_point, indices_start_at_one=True)

    # Clean up intermediate files created by tetgen
    _remove_files(scaled_shape_name, mesh_data_path, [".off"])
    _remove_files(combined_shape_name, mesh_data_path, [".1.edge", ".1.ele", ".1.face", ".1.node", ".1.mtr", ".1.node", ".1.p2t", ".mtr", ".off", ".smesh"])
   
    return tet_mesh_out



####### Paths #######

def _get_off_path_scaled(mesh_data_path, scaled_shape_name):
    return mesh_data_path + scaled_shape_name + ".off"


def _get_path_to_triangle_mesh_file(shape, mesh_data_path):
    return mesh_data_path + shape + ".off"



####### Clean up #######

def _remove_files(name, path, extensions):
    for ext in extensions:
        file_path = os.path.join(path, name + ext)
        if os.path.exists(file_path):
            os.remove(file_path)



####### Scaling #######

def scale_off_file_to_range_minus_1_to_1(off_path, off_path_scaled):
    min_point, max_point = _get_max_min_from_off_file(off_path)
    
    min_point = np.array(min_point)
    max_point = np.array(max_point)
    
    centers = (max_point + min_point) / 2.0
    scale = max(max_point - min_point) / 2.0
    
    with open(off_path, 'r') as f:
        lines = f.readlines()
    
    with open(off_path_scaled, 'w') as f:
        num_vertices = int(lines[1].split()[0])
        f.write(lines[0])
        f.write(lines[1])
        cnt = 0
        for line in lines[2:]:
            if cnt < num_vertices:
                points = line.split()

                x, y, z = map(float, points)
                x = (x - centers[0]) / scale
                y = (y - centers[1]) / scale
                z = (z - centers[2]) / scale
                f.write(f"{x} {y} {z}\n")
                cnt += 1
            else:   
                f.write(line)


######## Generate bounding box #######

def generate_triangle_mesh_with_bounding_box(off_path, target_path, offset):
    min_point, max_point = _get_max_min_from_off_file(off_path)
    
    min_point = torch.tensor(min_point)
    max_point = torch.tensor(max_point)

    # add offset to the bounding box
    min_point -= offset
    max_point += offset

    center = (max_point + min_point) / 2

    #find longest side length of the bounding box
    length_x = max_point[0] - min_point[0]
    length_y = max_point[1] - min_point[1]
    length_z = max_point[2] - min_point[2]
    side_lengths = torch.tensor([length_x, length_y, length_z])
    
    # find the corners of the bounding box
    bb_corners = _find_bounding_box_corners(center, side_lengths)

    # We now add additional triangles to the off file to represent the bounding box.
    # Each face of the bounding box is represented by two triangles.
    bb_indices = [[0, 1, 2], [2, 1, 3], [0, 4, 1], [1, 4, 5], [0, 2, 4], [4, 2, 6], [7, 1, 5], [7, 5, 4], [7, 4, 6], [7, 6, 2], [7, 2, 3], [7, 3, 1]] 
    
    # Read the original off file. 
    with open(off_path, 'r') as f:
        lines = f.readlines()

    # Calculate the new number of vertices and triangles.
    num_vertices = int(lines[1].split()[0])
    num_tets = int(lines[1].split()[1])
    third_number = int(lines[1].split()[2])

    new_num_vertices = num_vertices + len(bb_corners)
    new_num_tets = num_tets + len(bb_indices)

    # Write the new off file with the bounding box included.
    with open(target_path, 'w') as f:
        # Write the first line of the Header which is always "OFF"
        f.write(lines[0])
        # Write the second line of the Header with the new number of vertices and triangles
        f.write(str(new_num_vertices) + " " + str(new_num_tets) + " " + str(third_number) + "\n")
        
        # Write the original vertices. i starts from line 2 since line 0 is "OFF" and line 1 is the header with numbers.
        for i in range(num_vertices):
            f.write(lines[i + 2])
        # Write the corner vertices of the bounding box.
        for i in range(len(bb_corners)):
            f.write(str(bb_corners[i][0].item()) + " " + str(bb_corners[i][1].item()) + " " + str(bb_corners[i][2].item()) + "\n")
        # Write the original tetrahedra. i starts from line 2 + num_vertices since line 0 is "OFF" and line 1 is the header with numbers and the next num_vertices lines are in front of the vertices.
        for i in range(num_tets):
            f.write(lines[i + 2 + num_vertices])
        # Write the triangles representing the faces of the bounding box.
        # The first number is always 3 since each face is represented by a triangle.
        # The indices of the vertices of the bounding box start after the original vertices, hence the addition of num_vertices to the indices.
        for i in range(len(bb_indices)):
            f.write("3 " + str(bb_indices[i][0] + num_vertices) + " " + str(bb_indices[i][1] + num_vertices) + " " + str(bb_indices[i][2] + num_vertices) + "\n")  


def _find_bounding_box_corners(center, side_lengths):
    # There are 8 corners in a bounding box, each corner can be represented by three numbers.
    bb_positions = torch.zeros((8, 3))
    
    # Iterate over all 8 corners.
    for i in range(8):
        # Start from the center of the bounding box. We do not want to modify the center itself, so we clone it.
        p = center.clone()
        # For each corner, we check the bits of the index i to determine whether to add or subtract half the side length in each dimension.
        # E.g. for i=5 (binary 101), we add half the side length in x and z direction, but subtract it in y direction.
        for j in range(3):
            if ((i >> (2 - j)) & 1):
                p[j] += (side_lengths[j] / 2) 
            else:
                p[j] += (-side_lengths[j] / 2)
        bb_positions[i] = p
    return bb_positions   


####### Get max and min from off file #######
  
def _get_max_min_from_off_file(off_path):
    with open(off_path, 'r') as f:
        lines = f.readlines()
        
    min_point = [float('inf'), float('inf'), float('inf')]
    max_point = [-float('inf'), -float('inf'), -float('inf')]
    for line in lines[2:]:
        points = line.split()
        
        # vertices have 3 coordinates, faces have more than 3
        if len(points) == 3:
            x, y, z = map(float, points)
            if x < min_point[0]:
                min_point[0] = x
            if y < min_point[1]:
                min_point[1] = y
            if z < min_point[2]:
                min_point[2] = z
                
            if x > max_point[0]:
                max_point[0] = x
            if y > max_point[1]:
                max_point[1] = y
            if z > max_point[2]:
                max_point[2] = z
    
    return min_point, max_point  
       
       

