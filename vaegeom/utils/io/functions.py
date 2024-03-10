import math

import numpy as np
import torch
from torch import Tensor


def get_memory_str(mem: int) -> str:
    units = ('B', 'KB', 'MB', 'GB', 'TB')
    power = 0
    base = 1024
    if mem > 0:
        power = math.floor(math.log(mem, base))
    out = mem / (base ** power)

    return f'{out :.3f} {units[power]}'

def head_npy(array: np.ndarray) -> str:
    mem = get_memory_str(array.nbytes)
    return f'{array.shape}, {array.dtype}, {mem}'

def head_tnsr(tnsr: Tensor) -> str:
    mem = get_memory_str(tnsr.nbytes)
    is_grad = tnsr.requires_grad
    return f'{tnsr.shape}, {tnsr.dtype}, {tnsr.device}, {mem}, grad={is_grad}'

def head_array(
    array: np.ndarray | Tensor,
    name: str | None = None) -> str:
    if isinstance(array, np.ndarray):
        msg = head_npy(array)
    elif isinstance(array, Tensor):
        msg = head_tnsr(array)
    else:
        raise ValueError('Input must be either np.ndaray or torch.Tensor.')
    if name is None:
        return msg
    else:
        return f'{name}: {msg}'
