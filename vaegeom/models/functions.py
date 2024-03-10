import torch
from torch import Tensor
import torch.nn.functional as F

###############################################################################
#### Functions ################################################################

def safe_softmax(x: Tensor, dim: int = 0) -> Tensor:
    x_max = x.max(dim=dim)[0]
    return F.softmax(x - x_max, dim=dim)