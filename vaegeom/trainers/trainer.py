import torch


class Trainer(object):

    def __init__(
        self,
        dir: 'Path',
        logger: 'Logger',
        name: str,
        inst_earlystop: 'Earlystop',
        max_epochs: int,
        *args,
        version: int = 0,
        accelerator: str = 'cpu',
        precision: str = '32-true',
        **kwargs):
        self.logger = logger
        self.dir = self._mkdir_if_not_exist(
            dir.joinpath(f'{name}/version_{version}')
        )
        self.dir_ckpt = self._mkdir_if_not_exist(
            self.dir.joinpath('checkpoints')
        )
        self.earlystop = inst_earlystop
        self.max_epochs = max_epochs
        _device = 'cuda:0' if accelerator == 'gpu' else 'cpu'
        self.device = torch.device(_device)
        if precision.startswith('64'):
            self.dtype = torch.float64
        else:
            self.dtype = torch.float32

    def _mkdir_if_not_exist(self, dir: 'pathlib.Path') -> 'pathlib.Path':
        if not dir.exists():
            self.logger.info(f'Create: {dir}')
            dir.mkdir()
        return dir

    def initialize_metric(self, model: 'nn.Module'):
        list_keys = ['state', 'epoch', 'step', *model.model.keys_loss]
        self.metric = {k: [] for k in list_keys}

    def set_metric(
        self,
        state: str,
        epoch: int,
        step: int,
        list_loss: Tensor):
        list_values = [state, epoch, step, *list_loss.tolist()]
        for k, v in zip(self.metric, list_values):
            self.metric[k].append(v)

    def fit(
        self,
        model: 'nn.Module',
        train_dataloaders: 'DataLoader',
        val_dataloaders: 'DataLoader',
        *args, **kwargs):
        self.initialize_metric(model)
        optimizer = model.configure_optimizers()
        self.logger('Start: Training.')
        for epoch in range(self.max_epochs):
            _state = 'train'
            model.train()
            train_loss = 0.0
            for batch_idx, batch in enumerate(train_dataloaders):
                optimizer.zero_grad()
                list_loss = model.training_step(batch, batch_idx)
                loss = list_loss.sum()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                self.set_metric(_state, epoch, batch_idx, list_loss)
            train_loss /= len(train_dataloaders)
            _state = 'val'
            model.eval()
            val_loss = 0.0
            for batch_idx, batch in enumerate(val_dataloaders):
                list_loss = model.validation_step(batch, batch_idx)
                loss = list_loss.sum()
                val_loss += loss.item()
                self.set_metric(_state, epoch, batch_idx, list_loss)
            val_loss /= len(val_dataloaders)

                


            