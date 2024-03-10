import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vaegeom.utils import exec_func
from vaegeom.utils import load_config
from vaegeom.utils import prepare_data
from vaegeom.utils import save_outputs
from vaegeom.utils import set_ckpt_path
from vaegeom.utils import set_seed_dtype_device
from vaegeom.utils import set_trainer

DESCRIPTION = 'Infer VAE model.'

###############################################################################
#### Main Function ############################################################

def main(
    dir: 'pathlib.Path',
    condition: str,
    logger: 'Logger',
    path_data: str,
    *args, 
    train_config: str | None = None,
    batch_size: int | None = None,
    version: int = 0,
    **kwargs) -> None:
    # Train config
    if train_config is None:
        train_config = f'config_train_{condition}.json'
    train_kw = load_config(train_config)
    # Set seed, device and dtype
    inst_torchconfig = set_seed_dtype_device(logger, **train_kw)
    # Data: (n_samples, n_features)
    data_normalization = train_kw.get('data_normalization', True)
    data = prepare_data(path_data, inst_torchconfig, data_normalization)
    dim_input = data.shape[-1]
    # Dataloader
    if batch_size is None:
        batch_size = len(data)
    num_workers = train_kw.get('num_workers', None)
    loader = DataLoader(
        data, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    # Trainer
    inst_module, inst_trainer = set_trainer(
        dir,
        condition,
        logger,
        dim_input,
        inst_torchconfig=inst_torchconfig,
        **train_kw
    )
    # Load checkpoint
    ckpt_path = set_ckpt_path(inst_trainer)
    logger.info(f'Load checkpoint from {ckpt_path}')
    # Inference
    logger.info('Start Inference.')
    outputs = inst_trainer.predict(
        model=inst_module,
        dataloaders=loader,
        ckpt_path=ckpt_path
    )
    # Gather outputs and Save
    save_outputs(logger, outputs, inst_module, inst_torchconfig, ckpt_path)
    logger.info('Finish Inference.')


###############################################################################
#### Execution ################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('config', help='config .json file')
    args = parser.parse_args()
    # Run
    exec_func(args, __file__, __name__, main)