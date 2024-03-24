import math

import torch
from torch import nn
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.multivariate_normal import MultivariateNormal

from .isotropic_gaussian import IsotropicGaussianEncoder
from vaegeom.models.linear_layer import LinearLayers
from vaegeom.models.linear_layer import LinearLayerConfig

class GaussianEncoder(IsotropicGaussianEncoder):
    """
    Encoder of Variational Autoencoder (VAE)
    mu, logvar, L' = f(x)
    L = diag(exp(0.5 * logvar)) + L' 
    z ~ N(mu, L @ L.T)
    """

    def __init__(
        self,
        dim_input: int,
        hidden: dict,
        latent_mu: dict,
        *args,
        latent_logvar: dict | None = None,
        latent_offdiag: dict | None = None,
        **kwargs):
        super().__init__(
            dim_input=dim_input,
            hidden=hidden,
            latent_mu=latent_mu,
            latent_logvar=latent_logvar,
            **kwargs
        )
        if latent_offdiag is None:
            self._cfg_latent_offdiag = LinearLayerConfig(**latent_mu)
        else:
            self._cfg_latent_offdiag = LinearLayerConfig(**latent_offdiag)
        self._cfg_latent_offdiag.dims[-1] = self.dim_offdiag
        self.latent_offdiag = LinearLayers(
            self.dims_hidden[-1],
            self._cfg_latent_offdiag
        )

    @property
    def dim_offdiag(self) -> int:
        return self.dim_latent * (self.dim_latent - 1) // 2

    def forward(self, x: Tensor) -> Distribution:
    #def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Returns
        -------
        mu: (n_batch, latent_dim), mean of Gaussian
        scale_tril: (n_batch, latent_dim, latent_dim)
            Cholesky factor of Gaussian
        """
        x = self.hidden(x)
        mu = self.latent_mu(x)
        logvar = self.latent_logvar(x)
        L_vec = self.latent_offdiag(x)
        scale_tril = self._create_scale_tril(logvar, L_vec)
        return self.create_q_z((mu, scale_tril))
        #return mu, scale_tril

    def _create_scale_tril(self, logvar: Tensor, L_vec: Tensor) -> Tensor:
        scale_tril = torch.diag_embed(torch.exp(0.5 * logvar))
        scale_tril = scale_tril.add_(self._vec_to_tril(L_vec))
        return scale_tril

    #def create_cov(self, params: tuple[Tensor, Tensor]) -> Tensor:
    #    mu, scale_tril = params
    #    cov = scale_tril.matmul(scale_tril.transpose(-2, -1))
    #    return mu, cov

    def create_q_z(self, params: tuple[Tensor, Tensor]) -> Distribution:
        """
        Create distribution q(z|x) from parameters.
        """
        loc, scale_tril = params
        return MultivariateNormal(
            loc=loc,
            scale_tril=scale_tril
        )

    def _vec_to_tril(self, value: Tensor) -> Tensor:
        """
        Parameters (required)
        ---------------------
        value: (N, d * (d - 1) / 2)

        Returns
        -------
        out: (N, d, d)
        """
        N, e = value.shape
        d = int(math.ceil(math.sqrt(2 * e)))
        tril_idx = torch.tril_indices(d, d, -1, device=value.device) # (2, d * (d - 1) / 2)
        tril_idx = tril_idx.repeat(1, N) # (2, N * d * (d - 1) / 2)
        batch_idx = torch.arange(N, device=value.device).repeat_interleave(e) # (N * d * (d - 1) / 2,)
        out = torch.zeros(N, d, d, dtype=value.dtype, device=value.device)
        out[batch_idx, *tril_idx.unbind()] = value.ravel()

        return out