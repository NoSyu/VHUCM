import torch


def to_var(x, on_cpu=False, gpu_id=None, async_var=False):
    if torch.cuda.is_available() and not on_cpu:
        x = x.cuda(gpu_id, async_var)
    return x


def to_tensor(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data
