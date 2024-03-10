from dataclasses import dataclass
from dataclasses import field

import lightning as L
import torch
from torch import nn
from torch import optim
from torch import Tensor

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
        self.name_predicts = ('q_mu', 'q_cov')

    def forward(self, x: Tensor, sampling: int | None = None):
        return self.model(x, sampling)
    
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
        """
        enc = self.model.encoder(batch)
        return self.model.encoder.create_cov(enc)

    def configure_optimizers(self):
        optimizer = OPTIMS[self.config.optim](
            self.model.parameters(),
            lr=self.config.lr,
            **self.config.optim_kw
        )
        return optimizer
