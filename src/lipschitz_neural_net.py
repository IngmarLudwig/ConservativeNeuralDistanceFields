# based on https://github.com/GCoiffier/1-Lipschitz-Neural-Distance-Fields/blob/main/common/models/sll.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from deel import torchlip
from util import get_device
import mouette as M
from visualization import reconstruct_surface_marching_cubes

def safe_inv(x):
    mask = x == 0
    x_inv = x ** (-1)
    x_inv[mask] = 0
    return x_inv

class SDPBasedLipschitzDense(nn.Module):

    def __init__(self, in_features, inner_dim=-1):
        super().__init__()

        inner_dim = inner_dim if inner_dim != -1 else in_features
        self.activation = nn.ReLU()

        self.weight = nn.Parameter(torch.empty(inner_dim, in_features))
        self.bias = nn.Parameter(torch.empty(1, inner_dim))
        self.q = nn.Parameter(torch.randn(inner_dim))

        nn.init.xavier_normal_(self.weight)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def compute_t(self):
        q = torch.exp(self.q)
        q_inv = torch.exp(-self.q)
        t = torch.abs(torch.einsum('i,ik,kj,j -> ij', q_inv, self.weight, self.weight.T, q)).sum(1)
        t = safe_inv(t)
        return t

    def forward(self, x):
        t = self.compute_t()
        res = F.linear(x, self.weight)
        res = res + self.bias
        res = t * self.activation(res)
        res = 2 * F.linear(res, self.weight.T)
        out = x - res
        return out
    
class LipschitzNeuralNet(nn.Module):
    def __init__(self, layer_width, num_hidden_layer):
        super().__init__()
        
        cnt = 0
        
        layers = []
        layers.append(nn.ZeroPad1d((0, layer_width-3)))
        for _ in range(num_hidden_layer):
            layers.append(SDPBasedLipschitzDense(layer_width))
            cnt += 1
        layers.append(torchlip.FrobeniusLinear(layer_width,1))
        cnt += 1
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        device = get_device()
        self.to(device)
        x = x.to(device)
        return self.model(x).squeeze()
    
    def add_to_last_bias(self, value):
        self.model[-1].bias.data += value

    def determine_max_radius_difference(self, inside_dataloader):
        device = get_device()
        self.to(device)

        max_difference = float('-inf')
        for points, radii in inside_dataloader:
            points = points.to(device)
            with torch.no_grad():
                output = self(points)
                
            output = output.cpu()
            # output must be negative for inside points, radius is positive so output + radius should be negative
            diff = output + radii
            max = torch.max(diff)
            
            if max > max_difference:
                max_difference = max
        return max_difference.item()

    def inflate(self, inside_dataloader, epsilon):
        max_difference = self.determine_max_radius_difference(inside_dataloader)
        print("max difference output radius: ", max_difference)

        if max_difference > 0:
            self.add_to_last_bias(-max_difference-epsilon)
            max_difference = self.determine_max_radius_difference(inside_dataloader)
            print("Radius adjusted. New max difference: ", max_difference)
    
    def render_with_points(self, plt_ax, n_points, batch_size, upper_point, lower_point):
        # classify points
        self.eval()
        
        # use batching to classify points to avoid memory issues
        with torch.no_grad():
            random_points = []
            classifications = []
            for i in range(0, n_points, batch_size):
                random_points_batch = torch.rand(batch_size, 3) * (upper_point - lower_point) + lower_point
                random_points.append(random_points_batch)
                classifications.append(self(random_points_batch))

        random_points = torch.cat(random_points, dim=0)
        classifications = torch.cat(classifications, dim=0)
        
        classifications = classifications.cpu()


        # remove points that are classified as outside
        random_points   = random_points  [classifications <= 0]
        classifications = classifications[classifications <= 0]

        random_points = random_points.cpu()
        classifications = classifications.cpu()

        if len(random_points) <= 0:
            print("No points to render.")

        # render using classifications for color with heatmap
        random_points = random_points.detach().numpy()
        classifications = classifications.detach().numpy()
        im = plt_ax.scatter(random_points[:, 0], random_points[:, 1], random_points[:,2], c=classifications, s=1, cmap=matplotlib.cm.jet)
        # add colorbar
        plt.colorbar(im, fraction=0.025, pad=0.04)


    def render_cross_section(self, plt_axes, n_points, upper_point, lower_point, axis, axis_value, remove_inside=False):
        assert axis in ["x", "y", "z"]
        
        random_points = torch.rand(n_points, 3) * (upper_point - lower_point) + lower_point

        if axis == "x":
            random_points[:, 0] = axis_value
            plt_axes.view_init(elev=0, azim=0)
        elif axis == "y":
            random_points[:, 1] = axis_value
            plt_axes.view_init(elev=0, azim=90)
        elif axis == "z":
            random_points[:, 2] = axis_value
            plt_axes.view_init(elev=90, azim=0)

        self = self.cpu()
        self.eval()
        classifications = self(random_points)

        if len(random_points) <= 0:
            print("No points to render.")
            return
        
        # render using classifications for color with heatmap
        random_points = random_points.cpu().detach().numpy()
        classifications = classifications.cpu().detach().numpy()

        if remove_inside:
            # remove points that are classified as inside
            random_points   = random_points  [classifications > 0]
            classifications = classifications[classifications > 0]
            
        im = plt_axes.scatter(random_points[:, 0], random_points[:, 1], random_points[:,2], c=classifications, s=1, cmap=matplotlib.cm.jet)
        # add colorbar
        plt.colorbar(im, fraction=0.025, pad=0.04)

    def reconstruct_surface_mesh(self, save_path, resolution, batch_size, upper_point, lower_point):
        domain = M.geometry.AABB(lower_point, upper_point)
        reconstruct_surface_marching_cubes(save_path, self, "forward", domain, res=resolution, batch_size=batch_size, handle_errors=False)
