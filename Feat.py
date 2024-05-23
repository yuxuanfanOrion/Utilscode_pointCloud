import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F







# ························································································
# :               _        _   _                                                         :
# :     _ __ ___ | |_ __ _| |_(_) ___  _ __                                              :
# :    | '__/ _ \| __/ _` | __| |/ _ \| '_ \                                             :
# :    | | | (_) | || (_| | |_| | (_) | | | |                                            :
# :    |_|  \___/ \__\__,_|\__|_|\___/|_| |_|_      __            _                      :
# :    (_)_ ____   ____ _ _ __(_) __ _ _ __ | |_   / _| ___  __ _| |_ _   _ _ __ ___     :
# :    | | '_ \ \ / / _` | '__| |/ _` | '_ \| __| | |_ / _ \/ _` | __| | | | '__/ _ \    :
# :    | | | | \ V / (_| | |  | | (_| | | | | |_  |  _|  __/ (_| | |_| |_| | | |  __/    :
# :    |_|_| |_|\_/ \__,_|_|  |_|\__,_|_| |_|\__| |_|  \___|\__,_|\__|\__,_|_|  \___|    :
# ························································································
from Coordinates import cartesan2spherical
from Rotation import rotm
def rifeat(points_r, points_s):
    """generate rotation invariant features

    Args:
        points_r (B x N x K x 3): 
        points_s (B x N x 1 x 3): 
    """

    # [*, 3] -> [*, 8] with compatible intra-shapes
    if points_r.shape[1] != points_s.shape[1]:
        points_r = points_r.expand(-1, points_s.shape[1], -1, -1)
    
    r_mean = torch.mean(points_r, -2, keepdim=True)
    l1, l2, l3 = r_mean - points_r, points_r - points_s, points_s - r_mean
    l1_norm = torch.norm(l1, 'fro', -1, True)
    l2_norm = torch.norm(l2, 'fro', -1, True)
    l3_norm = torch.norm(l3, 'fro', -1, True).expand_as(l2_norm)
    theta1 = (l1 * l2).sum(-1, keepdim=True) / (l1_norm * l2_norm + 1e-7)
    theta2 = (l2 * l3).sum(-1, keepdim=True) / (l2_norm * l3_norm + 1e-7)
    theta3 = (l3 * l1).sum(-1, keepdim=True) / (l3_norm * l1_norm + 1e-7)
    
    # spherical mapping
    sx = cartesan2spherical(points_s)
    sx[..., [0, 2]] = sx[..., [2, 0]]
    sx *= -1
    m = rotm(sx) # B x N x 1 x 3 x 3
    h = torch.norm(points_r, dim=-1, keepdim=True)
    r_s2 = points_r / h
    res = torch.einsum('bnxy,bnky->bnkx', m[:, :, 0], r_s2.expand(m.shape[0], m.shape[1], -1, -1))
    txj_inv_xi = torch.acos(torch.clamp(res[..., 2:3], -1., 1.)) / np.pi
    return torch.cat([txj_inv_xi, h, l1_norm, l2_norm, l3_norm, theta1, theta2, theta3], dim=-1)