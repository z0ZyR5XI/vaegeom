from dataclasses import dataclass

import torch
from torch import nn
from torch import Tensor
from torch.distributions.distribution import Distribution


@dataclass
class VAEOutput:
    p_xgivenz: Distribution
    q_z: Distribution
    z: Tensor
