from dataclasses import dataclass

import torch
from torch import nn
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.kl import kl_divergence

###############################################################################
#### Define Class #############################################################

@dataclass
class VAEOutput:
    distr_p_xgivenz: Distribution
    distr_q_z: Distribution
    distr_p_z: Distribution

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE)
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        *args,
        sampling: int = 1,
        **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sampling = sampling
        self.keys_loss = ('loss_recon', 'loss_reg')

    def forward(self, x: Tensor, sampling: int | None = None):
        """
        Parameters (required)
        ---------------------
        x: (B, D), Input tensor
        """
        if sampling is None:
            sampling = self.sampling
        enc = self.encoder(x)
        distr_q_z = self.encoder.create_q_z(enc)
        z = distr_q_z.rsample([sampling]) # (S, B, D)
        S, B, D = z.shape
        dec = self.decoder(z.view(S * B, D), S)
        distr_p_xgivenz = self.decoder.create_p_xgivenz(dec)
        distr_p_z = self.create_p_z(distr_q_z)
        
        return VAEOutput(
            distr_p_xgivenz,
            distr_q_z,
            distr_p_z
        )

    def create_p_z(self, q: Distribution) -> Distribution:
        """
        Create distribution p(z) from q(z|x).
        """
        loc = torch.zeros_like(q.mean) # (N, K)
        cov = torch.diag_embed(torch.ones_like(loc))
        out = self.encoder.create_q_z((loc, cov))
        return out

    def calc_loss(self, x: Tensor, output: VAEOutput) -> Tensor:
        """
        Returns
        -------
        loss: (2,), [Reconstruction loss, Regularization loss]
        """
        # Reconstruction loss: -log(p(x|z))
        log_p_xgivenz = output.distr_p_xgivenz.log_prob(x) # (S, B)
        loss_recon = log_p_xgivenz.neg().logsumexp(dim=0).mean(dim=0) # (,)
        # Regularization loss: KL(q(z)||p(z))
        kl_q_p = kl_divergence(output.distr_q_z, output.distr_p_z) # (B,)
        loss_reg = kl_q_p.mean(dim=0) # (,)
        
        return torch.stack((loss_recon, loss_reg))