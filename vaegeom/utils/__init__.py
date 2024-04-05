import argparse
import dataclasses
import json
from logging import DEBUG, INFO
from logging import FileHandler
from logging import Formatter
from logging import getLogger
from logging import Logger
from logging import StreamHandler
import os
import pathlib
from typing import Callable

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .logger.show_array import ShowArray
from vaegeom.callbacks import ProgressMsg
from vaegeom.models import build_model
from vaegeom.modules import build_module


def load_config(config: dict | str) -> dict:
    if isinstance(config, str):
        assert os.path.exists(config), f'{config} is not existed.'
        with open(config, 'r') as f:
            out = json.load(f)
    else:
        out = config
    return out

@dataclasses.dataclass
class TorchConfig:
    seed: int
    show_array: ShowArray
    dtype: torch.dtype
    device: torch.device

def set_seed_dtype_device(
    logger: Logger,
    double: bool = False,
    cuda: bool = False,
    seed: int | None = None,
    *args, **kwargs) -> TorchConfig:
    # Set seed
    if seed is None:
        seed = np.random.randint(1e7, 1e8)
    L.seed_everything(seed)
    #torch.manual_seed(seed)
    logger.info(f'seed: {seed}')
    # Display array instance
    show_array = ShowArray(logger)
    # Device and dtype
    dtype = torch.float64 if double else torch.float32
    device = torch.device('cpu')
    if torch.cuda.is_available() and cuda:
        device = torch.device('cuda:0')
    logger.info(f'device: {device}')

    return TorchConfig(seed, show_array, dtype, device)

def prepare_data(
    path_data: str,
    inst_torchconfig: TorchConfig,
    data_normalization: bool) -> Tensor:
    data = torch.as_tensor(
        np.load(path_data),
        dtype=inst_torchconfig.dtype,
        device=inst_torchconfig.device
    )
    inst_torchconfig.show_array(data, 'data')
    if data_normalization:
        data = data - data.mean(dim=0)
    return data

def set_dataloader(
    data: Tensor,
    batch_size: int,
    *args, **kwargs) -> DataLoader:
    if batch_size < 0:
        batch_size = len(data)
    return DataLoader(data, batch_size=batch_size, **kwargs)

def set_trainer(
    dir: pathlib.Path,
    condition: str,
    logger: Logger,
    dim_input: int,
    model: str,
    encoder_config: dict | str,
    decoder_config: dict | str,
    module_config: dict | str,
    ckpt_config: dict | str,
    earlystop_config: dict | str,
    trainer_config: dict | str,
    inst_torchconfig: TorchConfig,
    *args,
    module: str = 'VAEModule',
    **kwargs):
    # Model
    encoder_kw = load_config(encoder_config)
    decoder_kw = load_config(decoder_config)
    inst_model = build_model(
        model=model,
        dim_input=dim_input,
        encoder_kw=encoder_kw,
        decoder_kw=decoder_kw
    ).to(
        dtype=inst_torchconfig.dtype,
        device=inst_torchconfig.device
    )
    # Module
    module_kw = load_config(module_config)
    inst_module = build_module(
        module=module,
        model=inst_model,
        **module_kw
    )
    logger.debug(inst_module)
    logger.debug(inst_module.config)
    for _n, _p in inst_module.named_parameters():
    #for _n, _p in inst_model.named_parameters():
        inst_torchconfig.show_array.head(_p, _n)
    # AUC (log)
    name = '_'.join([
        s for s in condition.split('_')
        if not s.startswith('seed')
    ])
    logger.info(f'Output: {dir}/{name}')
    auc = CSVLogger(dir, name=name, version=inst_torchconfig.seed)
    # Callbacks
    ckpt_kw = load_config(ckpt_config) 
    #inst_ckpt = ModelCheckpoint(**ckpt_kw)
    earlystop_kw = load_config(earlystop_config)
    #inst_earlystop = EarlyStopping(**earlystop_kw)
    list_callbacks = [
        ProgressMsg(logger),
        ModelCheckpoint(**ckpt_kw),
        EarlyStopping(**earlystop_kw)
    ]
    # Trainer
    trainer_kw = load_config(trainer_config)
    inst_trainer = L.Trainer(
        logger=auc,
        callbacks=list_callbacks,
        #callbacks=[inst_ckpt, inst_earlystop],
        **trainer_kw
    )
    return inst_module, inst_trainer

def set_ckpt_path(inst_trainer: 'L.Trainer') -> 'pathlib.Path':
    dir_ckpt = pathlib.Path(f'{inst_trainer.log_dir}/checkpoints')
    ckpt_path = list(dir_ckpt.glob('*.ckpt'))[0]
    assert ckpt_path.exists(), f'Checkpoint {ckpt_path} is not existed.'

    return ckpt_path

def save_outputs(
    logger,
    outputs,
    inst_module,
    inst_torchconfig,
    ckpt_path):
    for i, _n in enumerate(inst_module.name_predicts):
        _z = torch.cat([_tpl[i] for _tpl in outputs])
        inst_torchconfig.show_array.head(_z, _n)
        _path = ckpt_path.with_name(f'{_n}.npy')
        np.save(_path, _z.clone().detach().cpu().numpy())
        logger.info(f'Save to: {_path}')
    return None

def set_logger(
    dir: pathlib.Path,
    condition: str,
    name: str,
    debug: bool = False) -> Logger:
    # Set Logger
    dir_log = dir.with_name('logs')
    assert dir_log.exists(), f'Directory {dir_log} is not existed.'
    path_log = dir_log.joinpath(condition + '.txt')
    logger = getLogger(name)
    level = DEBUG if debug else INFO
    logger.setLevel(level)
    handler = FileHandler(path_log)
    formatter = Formatter(
        '%(asctime)s : %(name)s : %(levelname)s : %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Remove stream handler
    for _hdlr in logger.handlers:
        if isinstance(_hdlr, StreamHandler):
            logger.removeHandler(_hdlr)

    return logger

def exec_func(
    args: argparse.ArgumentParser,
    script: str,
    name: str,
    func: Callable) -> None:
    # Config file
    path_config = pathlib.Path(args.config)
    with open(path_config, 'r') as f:
        config = json.load(f)
    debug = config.get('debug', False)
    # Logger
    dir = path_config.parent.joinpath('results')
    assert dir.exists(), f'Directory {dir} is not existed.'
    stem = pathlib.Path(script).stem
    condition = path_config.stem.replace(f'config_{stem}_', '')
    if name == '__main__':
        name = stem
    logger = set_logger(dir, f'{stem}_{condition}', name, debug)
    logger.info(logger.handlers)
    # Run
    logger.info(f'Load config: {path_config}')
    for k, v in config.items():
        logger.debug(f'{k}: {type(v)}')
    func(dir, condition, logger, **config)


def create_df_loss(df_log: pd.DataFrame) -> pd.DataFrame:
    cols = ['loss_recon', 'loss_reg', 'loss']
    cols_train = ['epoch', *['train_' + c for c in cols]]
    train = df_log[df_log['train_loss'] > 0][cols_train]
    train = train.groupby('epoch').mean().reset_index()
    cols_val = ['epoch', *['val_' + c for c in cols]]
    val = df_log[df_log['val_loss'] > 0][cols_val]
    out = train.merge(val, how='inner', on='epoch')
    return out