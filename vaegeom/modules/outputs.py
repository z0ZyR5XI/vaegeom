from dataclasses import dataclass
from dataclasses import InitVar

import torch
from torch import nn
from torch import Tensor
from torch.distributions.distribution import Distribution

from vaegeom.models.outputs import VAEOutput


@dataclass
class VAEModuleOutput:
    output: InitVar[VAEOutput]
    p_z: Distribution

    def __post_init__(self, output: VAEOutput):
        self.p_xgivenz = output.p_xgivenz
        self.q_z = output.q_z
        self.z = output.z