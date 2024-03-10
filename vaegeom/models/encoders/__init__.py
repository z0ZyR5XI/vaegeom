import torch
from torch import nn

from .gaussian import GaussianEncoder
from .isotropic_gaussian import IsotropicGaussianEncoder

ENCODERS = {
    'GaussianEncoder': GaussianEncoder,
    'IsotropicGaussianEncoder': IsotropicGaussianEncoder,
    'Encoder': IsotropicGaussianEncoder
}

def build_encoder(
    name: str,
    dim_input: int,
    **kwargs) -> nn.Module:
    return ENCODERS[name](dim_input, **kwargs)