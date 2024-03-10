import torch
from torch import nn

from .gaussian import GaussianDecoder
from .gaussian_sharedvar import GaussianSharedVarDecoder
from .zinb import ZINBDecoder

DECODERS = {
    'ZINBDecoder': ZINBDecoder,
    'GaussianSharedVarDecoder': GaussianSharedVarDecoder,
    'GaussianDecoder': GaussianDecoder,
    'Decoder': GaussianDecoder
}

def build_decoder(
    name: str,
    dim_output: int,
    **kwargs) -> nn.Module:
    return DECODERS[name](dim_output, **kwargs)