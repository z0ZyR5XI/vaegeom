from dataclasses import dataclass
from dataclasses import field

#import lightning as L
import torch
from torch import nn
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.distributions.distribution import Distribution
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal

from .vae_module import VAEModule

###############################################################################
#### Define Class #############################################################

@dataclass
class VAEVampPriorConfig:
    n_components: int
    sampling: int = 1
    lr: float = 1e-3
    optim: str = 'Adam'
    optim_kw: dict = field(default_factory=dict)
    train_mixture: bool = False

class VAEVampPrior(VAEModule):
    """Variational Mixture of Posterior (Vamp) prior"""

    def __init__(
        self,
        model: nn.Module,
        config: VAEVampPriorConfig,
        *args, **kwargs):
        super().__init__(model, config)
        self.n_components = self.config.n_components
        self.u = nn.Parameter(
            torch.randn(self.n_components, self.model.dim_input)
        )
        # Logit of mixture
        if config.train_mixture:
            _w = torch.zeros(self.n_components)
        else:
            _w = torch.randn(self.n_components)
        self.p_z_w = nn.Parameter(_w, requires_grad=config.train_mixture)

    @classmethod
    def create_config(cls, *args, **kwargs) -> VAEVampPriorConfig:
        return VAEVampPriorConfig(**kwargs)

    def create_p_z(self, x: Tensor, q: Distribution) -> Distribution:
        """
        Create distribution p(z) from Gaussian mixture model.
        """
        mix = Categorical(logits=self.p_z_w)
        comp = self.model.encoder(self.u)

        return MixtureSameFamily(mix, comp)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Get encoder outputs.
        
        Returns
        -------
        recon_x: (n_batch, input_dim), reconstructed input
        q_mu: (n_batch, n_dim), mean of posterior
        q_cov: (n_batch, n_dim, n_dim), covariance of posterior
        """
        output = self.forward(batch, self.config.sampling)
        recon_x = output.p_xgivenz.mean.mean(dim=0)
        q_mu, q_cov = self.model.encoder.create_cov(output.q_z)
        p_mu, p_cov = self.model.encoder.create_cov(
            output.p_z.component_distribution
        )
        p_w = output.p_z.mixture_distribution.probs
        return (
            recon_x,
            q_mu,
            q_cov,
            p_mu,
            p_cov,
            p_w
        )

    @property
    def name_predicts(self):
        return (
            'recon_x',
            'q_mu',
            'q_cov',
            'p_mu',
            'p_cov',
            'p_w'
        )
