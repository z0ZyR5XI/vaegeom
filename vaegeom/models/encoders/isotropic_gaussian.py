import torch
from torch import nn
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.multivariate_normal import MultivariateNormal

from vaegeom.models.linear_layer import LinearLayers
from vaegeom.models.linear_layer import LinearLayerConfig

class IsotropicGaussianEncoder(nn.Module):
    """
    Encoder of Variational Autoencoder (VAE)
    mu, logvar = f(x)
    z ~ N(mu, diag(exp(logvar)))
    """

    def __init__(
        self,
        dim_input: int,
        hidden: dict,
        latent_mu: dict,
        *args,
        latent_logvar: dict | None = None,
        **kwargs):
        super().__init__()
        self.dim_input = dim_input
        self._cfg_hidden = LinearLayerConfig(**hidden)
        self.dims_hidden = self._cfg_hidden.dims
        self.hidden = LinearLayers(dim_input, self._cfg_hidden)
        self._cfg_latent_mu = LinearLayerConfig(**latent_mu)
        self.dims_latent_mu = self._cfg_latent_mu.dims
        self.latent_mu = LinearLayers(
            self.dims_hidden[-1],
            self._cfg_latent_mu
        )
        if latent_logvar is None:
            self._cfg_latent_logvar = LinearLayerConfig(**latent_mu)
        else:
            self._cfg_latent_logvar = LinearLayerConfig(**latent_logvar)
        self.latent_logvar = LinearLayers(
            self.dims_hidden[-1],
            self._cfg_latent_logvar
        )
        self.dim_latent = self.dims_latent_mu[-1]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Returns
        -------
        mu: (n_batch, latent_dim), mean of Gaussian
        cov: (n_batch, latent_dim, latent_dim), covariance of Gaussian
        """
        x = self.hidden(x)
        mu = self.latent_mu(x)
        logvar = self.latent_logvar(x)
        cov = torch.diag_embed(logvar.exp())
        return mu, cov

    def create_cov(self, params: tuple[Tensor, Tensor]) -> Tensor:
        mu, cov = params
        return mu, cov

    def create_q_z(self, params: tuple[Tensor, Tensor]) -> Distribution:
        """
        Create distribution q(z|x) from parameters.
        """
        loc, covariance_matrix = params
        return MultivariateNormal(
            loc=loc,
            covariance_matrix=covariance_matrix
        )