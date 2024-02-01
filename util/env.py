import torch
import numpy as np

_device = None 

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #return _device

def set_device(dev):
    global _device
    _device = dev

def init_work(worker_id, seed):
    np.random.seed(seed + worker_id)

def pretty_device_name(device):
    dev_type = device.type

    dev_idx = (
        f',{device.index}'
        if (device.index is not None)
        else ''
    )

    dev_cname = (
        f' ({torch.cuda.get_device_name(device)})'
        if (dev_type == 'cuda')
        else ''
    )

    return f'{dev_type}{dev_idx}{dev_cname}'
