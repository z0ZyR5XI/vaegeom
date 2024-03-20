import torch
from torch import Tensor
import torch.nn.functional as F

###############################################################################
#### Functions ################################################################

def safe_softmax(x: Tensor, dim: int = 0) -> Tensor:
    x_max = x.max(dim=dim)[0]
    return F.softmax(x - x_max, dim=dim)

def square_cdist(x: Tensor, y: Tensor) -> Tensor:
    """
    Square Euclidean distance matrix.

    Parameters (required)
    ---------------------
    x: (*, M, D)
    y: (*, N, D)

    Returns
    -------
    out: (*, M, N)
    """
    xx = x.square().sum(dim=-1) # (*, M)
    yy = y.square().sum(dim=-1) # (*, N)
    xy = x.matmul(y.transpose(-2, -1)) # (*, M, N)
    out = xx.unsqueeze(-1) + yy.unsqueeze(-2) - 2.0 * xy
    return out.clamp(min=0.0)