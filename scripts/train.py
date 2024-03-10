import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vaegeom.utils import exec_func
from vaegeom.utils import load_config
from vaegeom.utils import prepare_data
from vaegeom.utils import set_seed_dtype_device
from vaegeom.utils import set_trainer

DESCRIPTION = 'Train VAE model.'

###############################################################################
#### Main Function ############################################################

def main(
    dir: 'pathlib.Path',
    condition: str,
    logger: 'Logger',
    path_data: str,
    path_train_test: str,
    model: str,
    encoder_config: dict | str,
    decoder_config: dict | str,
    module_config: dict | str,
    ckpt_config: dict | str,
    earlystop_config: dict | str,
    batch_size: int,
    trainer_config: dict | str,
    *args, 
    module: str = 'VAEModule',
    data_normalization: bool = True,
    num_workers: int | None = None,
    double: bool = False,
    cuda: bool = False,
    seed: int | None = None,
    **kwargs) -> None:
    # Set seed, device and dtype
    inst_torchconfig = set_seed_dtype_device(logger, double, cuda, seed)
    # Sample with train/ test
    df_samples = pd.read_csv(path_train_test, index_col=0)
    logger.info(f'Samples:\n{df_samples}')
    ps_train_test = df_samples['train_test']
    # Data: (n_samples, n_features)
    data = prepare_data(path_data, inst_torchconfig, data_normalization)
    dim_input = data.shape[-1]
    # DataLoader
    loader_train = DataLoader(
        data[ps_train_test == 0], 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    loader_val = DataLoader(
        data[ps_train_test == 1], 
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
        model,
        encoder_config,
        decoder_config,
        module_config,
        ckpt_config,
        earlystop_config,
        trainer_config,
        inst_torchconfig,
        module=module
    )
    # Training
    logger.info('Start training.')
    inst_trainer.fit(
        model=inst_module,
        train_dataloaders=loader_train,
        val_dataloaders=loader_val
    )
    logger.info('Finish training.')


###############################################################################
#### Execution ################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('config', help='config .json file')
    args = parser.parse_args()
    # Run
    exec_func(args, __file__, 'lightning.pytorch', main)