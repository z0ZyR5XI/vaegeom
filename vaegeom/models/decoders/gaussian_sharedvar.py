import torch
from torch import nn
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal

from .gaussian import GaussianDecoder

###############################################################################
#### Define Class #############################################################

class GaussianSharedVarDecoder(GaussianDecoder):
    """
    Decoder of Variational Autoencoder (VAE)
    p(x | z) = Gaussian(y, I)
    """

    def __init__(
        self,
        dim_output: int,
        hidden: dict,
        output: dict,
        *args, 
        scaler: bool = True,
        init_logvar: float = 1.0,
        **kwargs):
        super().__init__(dim_output, hidden, output)
        self._var_dim = 1 if scaler else self.dim_output
        self.logvar = nn.Parameter(torch.empty(self._var_dim))
        nn.init.uniform_(self.logvar, -init_logvar, init_logvar)

    def create_p_xgivenz(self, params: tuple[Tensor]) -> Distribution:
        """
        Create distribution p(x|z) from parameters.
        """
        loc = params[0]
        scale = self.logvar.exp().expand_as(loc)
        distr = Normal(loc=loc, scale=scale)
        out = Independent(distr, 1)
        return out
