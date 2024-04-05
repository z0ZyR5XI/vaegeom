import lightning as L
import torch
from torch import nn

from .vae_module import VAEConfig
from .vae_module import VAEModule
from .vae_mogprior import VAEMoGPrior
from .vae_updateprior import VAEUpdatePrior
from .vae_vampprior import VAEVampPrior

MODULES = {
    'VAEMoGPrior': VAEMoGPrior,
    'VAEVampPrior': VAEVampPrior,
    'VAEUpdatePrior': VAEUpdatePrior,
    'VAEModule': VAEModule
}

def build_module(
    module: str,
    model: nn.Module,
    **kwargs) -> L.LightningModule:
    cls_module = MODULES[module]
    config = cls_module.create_config(**kwargs)
    #config = VAEConfig(**kwargs)
    return cls_module(model=model, config=config)
    #return MODULES[module](model=model, config=config)