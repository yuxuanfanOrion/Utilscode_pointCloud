import numpy as np
import torch
import torch.nn as nn


def zm(a):
    '''
    rotation matrix around z axis:
    [cos(a), -sin(a), 0]
    [sin(a),  cos(a), 0]
    [     0,       0, 1]
    '''
    zeros = torch.zeros_like(a)
    ones = torch.ones_like(a)
    return torch.stack([torch.cos(a), -torch.sin(a), zeros, torch.sin(a), torch.cos(a), zeros, zeros, zeros, ones], -1).reshape(*a.shape, 3, 3)


def ym(a):
    '''
    rotation matrix around y axis:
    [cos(a),  0, sin(a)]
    [     0,  1,      0]
    [-sin(a), 0, cos(a)]
    '''
    zeros = torch.zeros_like(a)
    ones = torch.ones_like(a)
    return torch.stack([torch.cos(a), zeros, torch.sin(a), zeros, ones, zeros, -torch.sin(a), zeros, torch.cos(a)], -1).reshape(*a.shape, 3, 3)

def xm(a):
    '''
    rotation matrix around x axis:
    [1,       0,       0]
    [0, cos(a), -sin(a)]
    [0, sin(a),  cos(a)]
    '''
    zeros = torch.zeros_like(a)
    ones = torch.ones_like(a)
    return torch.stack([ones, zeros, zeros, zeros, torch.cos(a), -torch.sin(a), zeros, torch.sin(a), torch.cos(a)], -1).reshape(*a.shape, 3, 3)


def rotm(x):
    return zm(x[..., 0]) @ ym(x[..., 1]) @ zm(x[..., 2])