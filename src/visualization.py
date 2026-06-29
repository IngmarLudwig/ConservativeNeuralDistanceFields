import dataclasses
import matplotlib.pyplot as plt
import mouette as M
import numpy as np
import os
import torch
from tqdm import tqdm
from skimage.measure import marching_cubes
from torch.utils.data import DataLoader

@dataclasses.dataclass
class Limits:
    """ A class that represents the limits of a 3D space."""
    x_limits: tuple = (-1, 1)
    y_limits: tuple = (-1, 1)
    z_limits: tuple = (-1, 1)


def create_plt_axes(upper_point=None, lower_point=None):
    """ Creates a matplotlib.pyplot axis with 3D projection."""
    fig = plt.figure(figsize=(10, 10))
    plt_ax = fig.add_subplot(111, projection='3d')
    plt_ax.set_xlim(xmin=-1, xmax=1)
    plt_ax.set_ylim(ymin=-1, ymax=1)
    plt_ax.set_zlim(zmin=-1, zmax=1)

    if upper_point is not None and lower_point is not None:
        plt_ax.set_xlim(xmin=lower_point[0], xmax=upper_point[0])
        plt_ax.set_ylim(ymin=lower_point[1], ymax=upper_point[1])
        plt_ax.set_zlim(zmin=lower_point[2], zmax=upper_point[2])
        print(f"Setting limits to {lower_point} and {upper_point}")

    plt_ax.set_xlabel('x')
    plt_ax.set_ylabel('y')
    plt_ax.set_zlabel('z')
    return plt_ax

def forward_in_batches(evaluator, evaluation_func_name, inputs, compute_grad=False, batch_size=5000):
    """
    Generic function to evaluate inputs in batches using a specified evaluation function.
    
    Args:
        evaluator: The object containing the evaluation function (model or dataset)
        evaluation_func_name: String name of the method to call on evaluator
        inputs: Input points to evaluate
        compute_grad: Whether to compute gradients
        batch_size: Size of batches for processing
    
    Returns:
        numpy array of outputs, and optionally gradients if compute_grad=True
    """
    inputs = torch.Tensor(inputs)
    inputs = DataLoader(inputs, batch_size=batch_size, shuffle=False)
    outputs = []
    grads = []
    
    # Get the evaluation function
    eval_func = getattr(evaluator, evaluation_func_name)
    
    for batch in tqdm(inputs, total=len(inputs)):
        batch.requires_grad = compute_grad
        v_batch = eval_func(batch)
        if compute_grad:
            torch.sum(v_batch).backward()
            grads.append(batch.grad.detach().cpu().numpy())
        outputs.append(v_batch.detach().cpu().numpy())
    
    if compute_grad:
        return np.concatenate(outputs), np.concatenate(grads)
    else:
        return np.concatenate(outputs)

# Based on https://github.com/GCoiffier/1-Lipschitz-Neural-Distance-Fields/blob/main/common/visualize.py
def reconstruct_surface_marching_cubes(save_path, evaluator, evaluation_func_name, domain, isovalues=0, res=100, batch_size=5000, handle_errors=False):
    """
    Reconstructs surface meshes using marching cubes algorithm.
    
    Args:
        evaluator: The object containing the evaluation function (model or dataset)
        evaluation_func_name: String name of the method to call on evaluator
        domain: M.geometry.AABB domain for reconstruction
        isovalues: Isovalue(s) for marching cubes
        res: Resolution of the grid
        batch_size: Batch size for evaluation
        handle_errors: Whether to handle ValueError exceptions during marching cubes
        
    Returns:
        Dictionary of meshes keyed by (index, isovalue) tuples
    """
    if isinstance(isovalues, (int, float)): 
        isovalues = [isovalues]
    
    ### Feed grid to evaluator
    L = [np.linspace(domain.mini[i], domain.maxi[i], res) for i in range(3)]
    pts = np.hstack((np.meshgrid(*L))).swapaxes(0, 1).reshape(3, -1).T
    dist_values = forward_in_batches(evaluator, evaluation_func_name, pts, compute_grad=False, batch_size=batch_size)
    dist_values = dist_values.reshape((res, res, res))

    ### Call marching cubes
    # Precompute vectorized index→world transform for reprojection
    origins = np.array([L[i][0] for i in range(3)])
    steps = np.array([L[i][1] - L[i][0] for i in range(3)])

    to_save = dict()
    for ioff, off in enumerate(isovalues):
        try:
            verts, faces, normals, values = marching_cubes(dist_values, level=off)
            # Vectorized reprojection: transform index-space verts to world coordinates
            verts = origins + verts * steps
            values = values[:, np.newaxis]
            m = M.mesh.RawMeshData()
            m.vertices += list(verts)
            m.faces += list(faces)
            m = M.mesh.SurfaceMesh(m)
            normal_attr = m.vertices.create_attribute("normals", float, 3, dense=True)
            normal_attr._data = normals
            values_attr = m.vertices.create_attribute("values", float, 1, dense=True)
            values_attr._data = values
            to_save[(ioff, off)] = m
        except ValueError:
            if not handle_errors:
                raise
            continue
        
    for (n,off),mesh in to_save.items():
        print(f"Saving mesh with isovalue {off} to {save_path}")
        M.mesh.save(mesh, save_path)


def visualize_distance_field_cross_section(model, lower_point, upper_point, axis='y', axis_value=0.0, n_points=50000, batch_size=5000):
    """
    Visualizes the model's distance field as a 2D cross-section scatter plot.

    Samples random points in the specified cross-section plane within the bounding box,
    evaluates the model DF on them, and renders a scatter plot colored by DF value.

    Args:
        model:        Neural network model (callable, returns DF values).
        lower_point:  Lower AABB bound (torch.Tensor or array-like, shape [3]).
        upper_point:  Upper AABB bound (torch.Tensor or array-like, shape [3]).
        axis:         Axis normal to the cross-section plane ('x', 'y', or 'z').
        axis_value:   Position of the cross-section along the chosen axis.
        n_points:     Number of random points to sample in the plane.
        batch_size:   Batch size used when evaluating the model.
    """
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    lower = lower_point.cpu().numpy() if hasattr(lower_point, 'cpu') else np.asarray(lower_point)
    upper = upper_point.cpu().numpy() if hasattr(upper_point, 'cpu') else np.asarray(upper_point)

    axes_2d = [i for i in range(3) if i != axis_idx]
    pts_2d = np.random.uniform(low=lower[axes_2d], high=upper[axes_2d], size=(n_points, 2)).astype(np.float32)

    pts_3d = np.zeros((n_points, 3), dtype=np.float32)
    pts_3d[:, axis_idx] = axis_value
    pts_3d[:, axes_2d[0]] = pts_2d[:, 0]
    pts_3d[:, axes_2d[1]] = pts_2d[:, 1]

    pts_tensor = torch.tensor(pts_3d)
    df_values = []
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            df_values.append(model(pts_tensor[i:i + batch_size]).cpu().numpy().flatten())
    df_values = np.concatenate(df_values)

    axis_labels = ['x', 'y', 'z']
    xlabel = axis_labels[axes_2d[0]]
    ylabel = axis_labels[axes_2d[1]]
    vmax = float(np.abs(df_values).max())

    outside = df_values >= 0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('white')
    sc = ax.scatter(pts_2d[outside, 0], pts_2d[outside, 1], c=df_values[outside], cmap="YlGn", s=1, vmin=0, vmax=vmax)
    plt.colorbar(sc, ax=ax, label='DF')

    # Contour lines on scattered data via triangulation (all points for correct topology)
    triang = plt.matplotlib.tri.Triangulation(pts_2d[:, 0], pts_2d[:, 1])
    positive_levels = np.linspace(0, vmax, 16)[1:]  # exclude 0, only above zero
    ax.tricontour(triang, df_values, levels=positive_levels, colors='gray', linewidths=0.5, alpha=0.6)
    ax.tricontour(triang, df_values, levels=[0], colors='blue', linewidths=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Distance Field  ({axis} = {axis_value})')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

