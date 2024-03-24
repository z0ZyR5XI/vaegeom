import torch
from torch import nn
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal

from vaegeom.models.linear_layer import LinearLayers
from vaegeom.models.linear_layer import LinearLayerConfig

###############################################################################
#### Define Class #############################################################

class GaussianDecoder(nn.Module):
    """
    Decoder of Variational Autoencoder (VAE)
    p(x | z) = Gaussian(y, I)
    """

    def __init__(
        self,
        dim_output: int,
        hidden: dict,
        output: dict,
        *args, **kwargs):
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
        self._cfg_output = LinearLayerConfig(**output)
        self.output = LinearLayers(
            self._cfg_hidden.dims[-1],
            self._cfg_output,
            dims=[dim_output]
        )

    def forward(self, z: Tensor, sampling: int) -> Distribution:
    #def forward(self, z: Tensor, sampling: int) -> tuple[Tensor]:
        """
        Parameters (required)
        ---------------------
        z: Tensor, (S * B, D)
        sampling: int

        Returns
        -------
        y: (S, B, D) or (B, D) if sampling == 1
        """
        z = self.hidden(z)
        y = self.output(z)
        y = y.view(sampling, -1, y.shape[-1])
        return self.create_p_xgivenz((y,))
        #return (y,)

    def create_p_xgivenz(self, params: tuple[Tensor]) -> Distribution:
        """
        Create distribution p(x|z) from parameters.
        """
        loc = params[0]
        scale = torch.ones_like(loc)        
        distr = Normal(loc=loc, scale=scale)
        out = Independent(distr, 1)
        return out
