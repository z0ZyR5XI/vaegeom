from dataclasses import dataclass

import torch
from torch import nn
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.kl import kl_divergence

from .decoders import build_decoder
from .encoders import build_encoder
from .outputs import VAEOutput
from vaegeom.modules.outputs import VAEModuleOutput
#from vaegeom.modules.vae_module import VAEOutput

###############################################################################
#### Define Class #############################################################

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE)
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
        super().__init__()
        self.dim_input = dim_input
        self.encoder_kw = encoder_kw
        self.decoder_kw = decoder_kw
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        #self.encoder = encoder
        #self.decoder = decoder
        self.sampling = sampling
        self.keys_loss = ('loss_recon', 'loss_reg')

    def build_encoder(self):
        return build_encoder(dim_input=self.dim_input, **self.encoder_kw)

    def build_decoder(self):
        return build_decoder(dim_input=self.dim_input, **self.decoder_kw)

    def forward(
        self,
        x: Tensor,
        sampling: int | None = None) -> VAEOutput:
        #sampling: int | None = None) -> tuple[Distribution, Distribution]:
    #def forward(self, x: Tensor, sampling: int | None = None) -> VAEOutput:
        """
        Parameters (required)
        ---------------------
        x: (B, D), Input tensor
        """
        if sampling is None:
            sampling = self.sampling
        q_z = self.encoder(x)
        #distr_q_z = self.encoder(x)
        #enc = self.encoder(x)
        #distr_q_z = self.encoder.create_q_z(enc)
        z = q_z.rsample([sampling]) # (S, B, D)
        S, B, D = z.shape
        p_xgivenz = self.decoder(z.view(S * B, D), S)
        #dec = self.decoder(z.view(S * B, D), S)
        #distr_p_xgivenz = self.decoder.create_p_xgivenz(dec)
        #distr_p_z = self.create_p_z(distr_q_z)
        
        return VAEOutput(
            p_xgivenz=p_xgivenz,
            q_z=q_z,
            z=z
        )
        #return p_xgivenz, q_z
        #return VAEOutput(
        #    distr_p_xgivenz,
        #    distr_q_z,
        #    distr_p_z
        #)

    def calc_loss(self, x: Tensor, output: VAEModuleOutput) -> Tensor:
    #def calc_loss(self, x: Tensor, output: VAEOutput) -> Tensor:
        """
        Returns
        -------
        loss: (2,), [Reconstruction loss, Regularization loss]
        """
        # Reconstruction loss: -log(p(x|z))
        log_p_xgivenz = output.p_xgivenz.log_prob(x) # (S, B)
        loss_recon = log_p_xgivenz.logsumexp(dim=0).neg().mean(dim=0) # (,)
        #loss_recon = log_p_xgivenz.neg().logsumexp(dim=0).mean(dim=0) # (,)
        # Regularization loss: KL(q(z)||p(z))
        kl_q_p = kl_divergence(output.q_z, output.p_z) # (B,)
        loss_reg = kl_q_p.mean(dim=0) # (,)
        
        return torch.stack((loss_recon, loss_reg))