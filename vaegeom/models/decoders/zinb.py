from pyro.distributions import ZeroInflatedNegativeBinomial
import torch
from torch import nn
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.independent import Independent

from vaegeom.models.linear_layer import LinearLayers
from vaegeom.models.linear_layer import LinearLayerConfig

###############################################################################
#### Define Class #############################################################

class ZINBDecoder(nn.Module):
    """
    Decoder of Variational Autoencoder (VAE)
    Modeling likelihood by Zero-inflated Negative Binomial (ZINB)
    p(x | z) = ZINB(x | pi(z), mu(z), theta)

    - pi: dropout rate, parameterized by MLP
    - mu: mean of NB, parameterized by MLP
    - theta: dispersion of NB, trainable
    """

    def __init__(
        self,
        dim_output: int,
        hidden: dict,
        *args,
        output: dict | None = None,
        output_log_mu: dict | None = None,
        output_pi: dict | None = None,
        **kwargs):
        super().__init__()
        self._cfg_hidden = LinearLayerConfig(**hidden)
        self.dim_latent = self._cfg_hidden.dims[0]
        self.dims = [*self._cfg_hidden.dims, dim_output]
        self.hidden = LinearLayers(
            self.dim_latent,
            self._cfg_hidden,
            dims=self._cfg_hidden.dims[1:]
        )
        self.dim_output = dim_output
        if output is None:
            self._cfg_output_log_mu = LinearLayerConfig(**output_log_mu)
        else:
            self._cfg_output_log_mu = LinearLayerConfig(**output)
        self.output_log_mu = LinearLayers(
            self._cfg_hidden.dims[-1],
            self._cfg_output_log_mu,
            dims=[dim_output]
        )
        if output_pi is None:
            self._cfg_output_pi = self._cfg_output_mu
        else:
            self._cfg_output_pi = LinearLayerConfig(**output_pi)
        self.output_pi = LinearLayers(
            self._cfg_hidden.dims[-1],
            self._cfg_output_pi,
            dims=[dim_output]
        )
        self.log_theta = nn.Parameter(torch.rand(dim_output) - 0.5)

    def forward(self, z: Tensor, sampling: int) -> Distribution:
        """
        Parameters (required)
        ---------------------
        z: Tensor, (S * B, D)

        Parameters (optional)
        ---------------------
        sampling: int, Default = 1
        """
        z = self.hidden(z)
        log_mu = self.output_log_mu(z)
        log_mu = log_mu.view(sampling, -1, log_mu.shape[-1]).squeeze(0)
        pi = self.output_pi(z)
        pi = pi.view(sampling, -1, pi.shape[-1]).squeeze(0)

        return pi, log_mu

    def create_p_xgivenz(self, params: tuple[Tensor, Tensor]) -> Distribution:
        """
        Create distribution p(x|z) from parameters.
        """
        pi, log_mu = params
        distr = ZeroInflatedNegativeBinomial(
            total_count=self.log_theta.exp(),
            logits=log_mu.sub(self.log_theta),
            gate=pi,
            validate_args=False
        )
        out = Independent(distr, 1)
        return out
