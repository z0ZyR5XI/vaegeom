from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

import lightning as L
import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.distributions.distribution import Distribution

from .outputs import VAEModuleOutput
from vaegeom.models.outputs import VAEOutput

###############################################################################
#### Define Class #############################################################

OPTIMS = {
    'AdamW': optim.AdamW,
    'Adam': optim.Adam
}

@dataclass
class VAEConfig:
    sampling: int = 1
    lr: float = 1e-3
    optim: str = 'Adam'
    optim_kw: dict = field(default_factory=dict)

class VAEModule(L.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        config: VAEConfig,
        *args, **kwargs):
        super().__init__()
        self.model = model
        self.config = config
        self.name_predicts = ('recon_x', 'q_mu', 'q_cov')

    def create_p_z(self, x: Tensor, output: VAEOutput) -> Distribution:
    #def create_p_z(self, x: Tensor, q: Distribution) -> Distribution:
    #def create_p_z(self, q: Distribution) -> Distribution:
        """
        Create distribution p(z) from q(z|x).
        """
        loc = torch.zeros_like(output.q_z.mean) # (N, K)
        #loc = torch.zeros_like(q.mean) # (N, K)
        cov = torch.diag_embed(torch.ones_like(loc))
        out = self.model.encoder.create_q_z((loc, cov))
        return out

    def forward(
        self,
        x: Tensor, sampling: int | None = None) -> VAEModuleOutput:
        #x: Tensor, sampling: int | None = None) -> VAEOutput:
        output = self.model(x, sampling)
        #p_xgivenz, q_z = self.model(x, sampling)
        p_z = self.create_p_z(x, output)
        #p_z = self.create_p_z(x, q_z)
        return VAEModuleOutput(output=output, p_z=p_z)
        #return VAEModuleOutput(p_z=p_z, **asdict(output))
        #return VAEModuleOutput(p_xgivenz, q_z, p_z)
        #return VAEOutput(p_xgivenz, q_z, p_z)
        #return self.model(x, sampling)
    
    def _calc_record_loss(self, batch, batch_idx, state):
        output = self.forward(batch, self.config.sampling)
        list_loss = self.model.calc_loss(batch, output)
        self.log_dict({
            f'{state}_{key}': value.item()
            for key, value in zip(self.model.keys_loss, list_loss)
        })
        loss = list_loss.sum()
        self.log(f'{state}_loss', loss.item())
        return loss

    def training_step(self, batch, batch_idx):
        return self._calc_record_loss(batch, batch_idx, state='train')

    def validation_step(self, batch, batch_idx):
        return self._calc_record_loss(batch, batch_idx, state='val')

    def test_step(self, batch, batch_idx):
        return self._calc_record_loss(batch, batch_idx, state='test')

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
        #output = self.model(x, sampling)
        #p_xgivenz, q_z = self.model(batch)
        recon_x = output.p_xgivenz.mean.mean(dim=0)
        #recon_x = p_xgivenz.mean.mean(dim=0)
        q_mu, q_cov = self.model.encoder.create_cov(output.q_z)
        #q_mu, q_cov = self.model.encoder.create_cov(q_z)
        return recon_x, q_mu, q_cov
        #enc = self.model.encoder(batch)
        #return self.model.encoder.create_cov(enc)

    def configure_optimizers(self):
        optimizer = OPTIMS[self.config.optim](
            self.model.parameters(),
            lr=self.config.lr,
            **self.config.optim_kw
        )
        return optimizer
