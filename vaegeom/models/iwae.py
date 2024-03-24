#from dataclasses import dataclass

import torch
from torch import nn
from torch import Tensor
#from torch.distributions.distribution import Distribution
from torch.distributions.kl import kl_divergence

from .functions import safe_softmax
from .vae import VAE
from vaegeom.modules.outputs import VAEModuleOutput
#from vaegeom.modules.vae_module import VAEOutput

###############################################################################
#### Define Class #############################################################

class IWAE(VAE):
    """
    Importance weighted autoencoder (IWAE)
    """

    def __init__(
        self,
        dim_input: int,
        encoder_kw: dict,
        decoder_kw: dict,
        #encoder: nn.Module,
        #decoder: nn.Module,
        *args,
        sampling: int = 1,
        **kwargs):
        super().__init__(
            dim_input=dim_input,
            encoder_kw=encoder_kw,
            decoder_kw=decoder_kw,
            #encoder=encoder,
            #decoder=decoder,
            sampling=sampling,
            **kwargs
        )

    def calc_loss(self, x: Tensor, output: VAEModuleOutput) -> Tensor:
    #def calc_loss(self, x: Tensor, output: VAEOutput) -> Tensor:
        """
        Returns
        -------
        loss: (2,), [Reconstruction loss, Regularization loss]
        """
        log_p_xgivenz = output.p_xgivenz.log_prob(x) # (S, B)
        kl_q_p = kl_divergence(output.q_z, output.p_z) # (B,)
        log_w = log_p_xgivenz - kl_q_p # (S, B)
        weight = safe_softmax(log_w.clone().detach(), dim=0)
        # Reconstruction loss: -log(p(x|z))
        loss_recon = weight.mul(log_p_xgivenz.neg()).sum(dim=0).mean(dim=0) # (,)
        # Regularization loss: KL(q(z)||p(z))
        loss_reg = kl_q_p.mean(dim=0) # (,)
        
        return torch.stack((loss_recon, loss_reg))