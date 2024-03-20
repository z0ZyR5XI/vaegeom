import math

import torch
from torch import nn
from torch import Tensor

from .centroids import build_centroid
from vaegeom.models.functions import square_cdist
from vaegeom.models.linear_layer import LinearLayers
from vaegeom.models.linear_layer import LinearLayerConfig
from vaegeom.utils import load_config

class MetricNetPyraug(nn.Module):
    """
    Metric network from pyraug
    (https://github.com/clementchadebec/pyraug)
    """

    def __init__(
        self,
        dim_input: int,
        hidden: dict,
        centroids_config: dict | str,
        temperature: float
        *args,
        lmd: float = 1e-3,
        eps: float = 1e-2,
        **kwargs):
        super().__init__()
        self.dim_input = dim_input
        self._cfg_hidden = LinearLayerConfig(**hidden)
        self.dim_latent = self._cfg_hidden.dims[-1]
        self._cfg_hidden.dims[-1] = self.dim_offdiag()
        self.dims_hidden = self._cfg_hidden.dims
        self.hidden = LinearLayers(dim_input, self._cfg_hidden)
        self.centroid_fn = build_centroid(**load_config(centroids_config))
        self.temperature = temperature
        self.lmd = lmd
        self.eps = eps

    @property
    def dim_offdiag(self) -> int:
        return self.dim_latent * (self.dim_latent + 1) // 2

    def calc_G_inv(self, z: Tensor, c: Tensor, M: Tensor) -> Tensor:
        """
        Parameters (required)
        ---------------------
        z: (*, n_batch, n_dim), points in latent space
        c: (n_centroids, n_dim), centroids in latent space
        M: (n_centroids, n_dim, n_dim), G_inv elements on centroids

        Returns
        -------
        out: (n_batch, n_dim, n_dim), inverse metric tensor on points
        """
        weight = square_cdist(z, c).div(self.temperature ** 2).neg().exp()
        weight = weight.unsqueeze(-1).unsqueeze(-1) # (*, n_batch, n_centroids, 1, 1)
        out = M.mul(weight).sum(dim=-3) # (*, n_batch, n_dim, n_dim)
        I = torch.eye(z.shape[-1], dtype=z.dtype, device=z.device)
        return out + self.lmd * I

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
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
        return mu, scale_tril

    def _hamiltonian(
        self, 
        output: 'VAEOutput',
        x: Tensor,
        z: Tensor,
        rho: Tensor,
        G_inv: Tensor) -> Tensor:
        # Potential: U = -log(p(x|z)) - log(p(z))
        log_p_xgivenz = output.distr_p_xgivenz.log_prob(x) # (S, B)
        log_p_z = output.distr_p_z.log_prob(z) # (S, B)
        U = -(log_p_xgivenz + log_p_z)
        # Kinetic: K = -log(p(rho, G))
        K = self._gaussian_neg_log_prob(rho, G_inv) # (S, B)

    def _vec_to_tril(self, value: Tensor) -> Tensor:
        """
        Parameters (required)
        ---------------------
        value: (N, d * (d + 1) / 2)

        Returns
        -------
        out: (N, d, d)
        """
        N, e = value.shape
        d = int(math.floor(math.sqrt(2 * e)))
        tril_idx = torch.tril_indices(d, d, 0, device=value.device) # (2, d * (d + 1) / 2)
        tril_idx = tril_idx.repeat(1, N) # (2, N * d * (d + 1) / 2)
        batch_idx = torch.arange(N, device=value.device).repeat_interleave(e) # (N * d * (d + 1) / 2,)
        out = torch.zeros(N, d, d, dtype=value.dtype, device=value.device)
        out[batch_idx, *tril_idx.unbind()] = value.ravel()

        return out

    def _gaussian_log_prob(self, rho: Tensor, G_inv: Tensor) -> Tensor:
        """
        Log probability of p(rho) = Normal(0, G) from G_inv
        """
        # Mahalanobis distance: rho^T @ G_inv @ rho
        mahal = rho.unsqueeze(-2) @ G_inv @ rho.unsqueeze(-1)
        mahal = mahal.squeeze((-2, -1))
        # Partition function: log((2 * pi)^D * det(G))
        logZinv = G_inv.logdet() - rho.shape[-1] * math.log(2 * math.pi)

        out = 0.5 * (logZinv - mahal)
        return out