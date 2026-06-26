def generate_smesh_and_mtr_file_from_off(off_filename, max_edge_length):    
    vertices, faces = _read_off_file(off_filename)
    _generate_smesh_and_mtr(vertices, faces, off_filename, max_edge_length)


def _generate_smesh_and_mtr(vertices, faces, filename, max_edge_length):
    # get name without extension
    base_name = filename.rsplit('.', 1)[0]
    
    _write_smesh_file(vertices, faces, base_name)
    _write_mtr_file(base_name, vertices, max_edge_length)


def _read_off_file(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        if lines[0].strip() != 'OFF':
            raise ValueError("File is not in OFF format")
        
        # Read number of vertices, edges, and faces
        n_vertices, n_faces, n_edges = map(int, lines[1].strip().split())
        
        # Read vertices
        for i in range(2, 2 + n_vertices):
            parts = lines[i].strip().split()
            vertex = list(map(float, parts[:3]))
            vertices.append(vertex)
        
        # Read faces
        for i in range(2 + n_vertices, 2 + n_vertices + n_faces):
            parts = list(map(int, lines[i].strip().split()))
            parts = parts[1:]  # The first number is the count of vertices in the face
            parts = parts[:3]  # We only take the first three vertices for triangular faces
            faces.append(parts)
            
        #add 1 to each vertex index to convert from 0-based to 1-based indexing
        faces = [[v + 1 for v in face] for face in faces]

    return vertices, faces


def _write_smesh_file(vertices, faces, base_name):
    smash_file_name = f"{base_name}.smesh"
    with open(smash_file_name, 'w') as f:
        f.write(f"{len(vertices)} 3 0 1\n") # <# of points> <dimension (must be 3)> <# of attributes> <# of boundary markers (0 or 1)>

        for i, vertex in enumerate(vertices, start=1):
            f.write(f"{i} {vertex[0]} {vertex[1]} {vertex[2]} 1\n") # <point #> <x> <y> <z>[attributes] [boundary marker]

        f.write(f"{len(faces)} 1\n") #<# of facets> <boundary markers (0 or 1)>
        for face in faces:
            f.write("3 " + " ".join(map(str, face)) + " 1\n") # <# of corners> <corner 1> <corner 2> ... <corner #> [boundary marker]

        f.write("0\n")  # <# of holes>
        f.write("0")  # <# of region>


def _write_mtr_file(base_name, vertices, max_edge_length):
    """ An mtr file specifies the desired edge lengths at each vertex. 
        Here we set a uniform edge length for all vertices.
        The mtr file is then used by tetgen when generating the tetrahedral mesh.
    """
    mtr_file_name = f"{base_name}.mtr"
    with open(mtr_file_name, 'w') as f:
        f.write(f"{len(vertices)} 1\n") # <# of nodes> <size of metric (always 1)>
        for _ in range(len(vertices)):
            f.write(f"{max_edge_length}\n") # < length for all edges connecting to the node>
