import torch
from torch.autograd import Variable


def pad(tensor, length):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
        else:
            return var
    else:
        if length > tensor.size(0):
            return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
        else:
            return tensor
