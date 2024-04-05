#from dataclasses import dataclass

import torch
#from torch import nn
from torch import Tensor

from .vae import VAE
from vaegeom.modules.outputs import VAEModuleOutput

###############################################################################
#### Define Class #############################################################

class VAENoKL(VAE):
    """
    Variational Autoencoder (VAE)
    without using kl_divergence
    """

    def __init__(
        self,
        dim_input: int,
        encoder_kw: dict,
        decoder_kw: dict,
        *args,
        sampling: int = 1,
        **kwargs):
        super().__init__(
            dim_input=dim_input,
            encoder_kw=encoder_kw,
            decoder_kw=decoder_kw,
            sampling=sampling,
            **kwargs
        )

    def calc_loss(self, x: Tensor, output: VAEModuleOutput) -> Tensor:
        """
        Returns
        -------
        loss: (2,), [Reconstruction loss, Regularization loss]
        """
        # Reconstruction loss: -log(p(x|z))
        log_p_xgivenz = output.p_xgivenz.log_prob(x) # (S, B)
        loss_recon = log_p_xgivenz.neg().mean() # (,)
        # Regularization loss: KL(q(z)||p(z))
        log_q_z = output.q_z.log_prob(output.z) # (S, B)
        log_p_z = output.p_z.log_prob(output.z) # (S, B)
        kl_q_p = log_q_z.sub(log_p_z).mean(dim=0) # (B,)
        loss_reg = kl_q_p.mean(dim=0) # (,)
        
        return torch.stack((loss_recon, loss_reg))