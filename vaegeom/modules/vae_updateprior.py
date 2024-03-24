import lightning as L
import torch
from torch import nn
from torch import Tensor
from torch.distributions.distribution import Distribution

from .vae_module import VAEConfig
from .vae_module import VAEModule

###############################################################################
#### Define Class #############################################################

class UpdatePriorCallback(L.Callback):

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.update_p_z_()


class VAEUpdatePrior(VAEModule):

    def __init__(
        self,
        model: nn.Module,
        config: VAEConfig,
        *args, **kwargs):
        super().__init__(model, config)
        self._prev_encoder = self.model.build_encoder()
        self.update_p_z_()

    def update_p_z_(self):
        self._prev_encoder.load_state_dict(self.model.encoder.state_dict())
        self._prev_encoder.requires_grad_(False)

    def create_p_z(self, x: Tensor, q: Distribution) -> Distribution:
    #def create_p_z(self, q: Distribution) -> Distribution:
        """
        Create distribution p(z) from encoder of previous epoch.
        """
        self._prev_encoder.eval()
        return self._prev_encoder(x)

    def configure_callbacks(self):
        return [UpdatePriorCallback()]