from collections import namedtuple
import torch
import torch.nn.functional as F

SphereData = namedtuple('SphereData', ['V_R_in_i', 'V_R_in', 'V_R_out_j', 'V_R_out', 'r_in_i', 'r_out_j'])

class BinaryCrossEntropy(torch.nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, lbda, f_C_in_i, f_C_out_j, r_in_i, r_out_j=None, gamma=None):
        (V_R_in_i, V_R_in, V_R_out_j, V_R_out, r_in_i, r_out_j) = calculate_or_set_volumes_and_radii(f_C_out_j.shape[0], r_in_i, r_out_j)
        
        bce_in  = -1.0 * lbda * 1./V_R_in  * torch.sum(V_R_in_i  * F.logsigmoid(-1.0 * self.temperature * (f_C_in_i  + r_in_i)))
        bce_out = -1.0 *        1./V_R_out * torch.sum(V_R_out_j * F.logsigmoid(       self.temperature * (f_C_out_j - r_out_j)))
        bce_loss = bce_in + bce_out
        return bce_loss, bce_loss, torch.tensor(0)


class HKR_Loss(torch.nn.Module):
    def __init__(self, margin_in, margin_out):
        super().__init__()
        self.kr_loss_fn = KR_Loss()
        self.hinge_loss_fn = Hinge_Loss(margin_in=margin_in, margin_out=margin_out)
        
    def forward(self, lbda, f_C_in_i, f_C_out_j, r_in_i, r_out_j=None, gamma=None):
        # Calculate volumes once and reuse for both losses
        sphere_data = calculate_or_set_volumes_and_radii(f_C_out_j.shape[0], r_in_i, r_out_j)
        kr_loss    = self.kr_loss_fn.forward(         f_C_in_i, f_C_out_j, sphere_data)
        hinge_loss = self.hinge_loss_fn.forward(lbda, f_C_in_i, f_C_out_j, sphere_data)
        hKR_loss = hinge_loss + gamma * kr_loss
        return hKR_loss, hinge_loss, kr_loss


class Hinge_Loss(torch.nn.Module):
    def __init__(self, margin_in, margin_out):
        super().__init__()
        self.margin_in  = margin_in
        self.margin_out = margin_out
        
    def forward(self, lbda, f_C_in_i, f_C_out_j, sphere_data):
        V_R_in_i, V_R_in, V_R_out_j, V_R_out, r_in_i, r_out_j = sphere_data
        
        hinge_in  = lbda * 1./V_R_in  * torch.sum(V_R_in_i  * F.relu(self.margin_in  + r_in_i  + f_C_in_i ))
        hinge_out =        1./V_R_out * torch.sum(V_R_out_j * F.relu(self.margin_out + r_out_j - f_C_out_j))
        hinge_loss = hinge_in + hinge_out
        return hinge_loss


class KR_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f_C_in_i, f_C_out_j, sphere_data):
        V_R_in_i, V_R_in, V_R_out_j, V_R_out, r_in_i, r_out_j = sphere_data
        
        kr_in  =   1./V_R_in  * torch.sum(V_R_in_i  * f_C_in_i)
        kr_out =  -1./V_R_out * torch.sum(V_R_out_j * f_C_out_j)
        kr_loss = kr_in + kr_out
        return kr_loss


def calculate_or_set_volumes_and_radii(num_output_values, r_in_i, r_out_j):
    V_R_in_i, V_R_in  = _calculate_volumes(r_in_i)
    if r_out_j is None:
        r_out_j   = torch.zeros(num_output_values, device=r_in_i.device, dtype=r_in_i.dtype)
        V_R_out_j = torch.ones (num_output_values, device=r_in_i.device, dtype=r_in_i.dtype)
        V_R_out = torch.tensor(num_output_values, device=r_in_i.device, dtype=r_in_i.dtype)
    else:
        V_R_out_j, V_R_out = _calculate_volumes(r_out_j)
    return SphereData(V_R_in_i, V_R_in, V_R_out_j, V_R_out, r_in_i, r_out_j)


def _calculate_volumes(radii):
    V_R_i = 4./3.*torch.pi * torch.pow(radii, 3)
    V_R = torch.sum(V_R_i)
    return V_R_i, V_R
