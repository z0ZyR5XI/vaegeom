import lightning as L
import torch
from torch import nn

from .vae_module import VAEConfig
from .vae_module import VAEModule
from .vae_updateprior import VAEUpdatePrior

MODULES = {
    'VAEUpdatePrior': VAEUpdatePrior,
    'VAEModule': VAEModule
}

def build_module(
    module: str,
    model: nn.Module,
    **kwargs) -> L.LightningModule:
    config = VAEConfig(**kwargs)
    return MODULES[module](model=model, config=config)